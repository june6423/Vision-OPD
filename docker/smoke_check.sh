#!/bin/bash
# Smoke check of the Vision-OPD container (VLM-VISIONOPD-REPRO-001). Not run in revision R2.
# Usage, inside the container from /workspace/harness/vision-opd, with at least 1 GPU:
#   bash docker/smoke_check.sh [model_dir]
# Prints versions, the GPU and driver view, nvcc, the Qwen3.5-4B config, and the chat template check.
# The kernel checks run with CUDA_DISABLE_PTX_JIT=1, so a kernel without sm_80 SASS fails.
set -uo pipefail

model="${1:-/workspace/harness/data/vision-opd/Qwen3.5-4B}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
status=0

echo "== nvcc"; nvcc --version || status=1
echo "== nvidia-smi"; nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv || status=1

echo "== python"
CUDA_DISABLE_PTX_JIT=1 python - "$model" <<'EOF' || status=1
import importlib
import json
import sys

ok = True
for name in ("torch", "vllm", "flash_attn", "causal_conv1d", "fla", "transformers", "ray", "triton", "flashinfer"):
    try:
        module = importlib.import_module(name)
        print("import", name, getattr(module, "__version__", "no __version__"))
    except Exception as e:
        ok = False
        print("import", name, "FAILED", type(e).__name__, e)

import torch

print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("driver_version", torch._C._cuda_getDriverVersion())
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    with open("/proc/self/maps") as maps:
        print("libcuda", sorted({line.split()[-1] for line in maps if "libcuda.so" in line}))
    x = torch.randn(256, 256, device="cuda")
    print("matmul", float((x @ x).sum()))
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        print("flash_attn_func", tuple(flash_attn_func(q, q, q, causal=True).shape))
    except Exception as e:
        ok = False
        print("flash_attn_func FAILED", type(e).__name__, e)
    try:
        from causal_conv1d import causal_conv1d_fn
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
        print("causal_conv1d_fn", tuple(causal_conv1d_fn(x, w, activation="silu").shape))
    except Exception as e:
        ok = False
        print("causal_conv1d_fn FAILED", type(e).__name__, e)
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def add_one(ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(ptr + offs, tl.load(ptr + offs, mask=mask) + 1, mask=mask)

        y = torch.zeros(1000, device="cuda")
        add_one[(4,)](y, 1000, BLOCK=256)
        print("triton kernel", float(y.sum()))
    except Exception as e:
        ok = False
        print("triton kernel FAILED", type(e).__name__, e)
else:
    ok = False

from transformers import AutoConfig

model = sys.argv[1]
config = AutoConfig.from_pretrained(model)
print("model_type", config.model_type)
with open(f"{model}/config.json") as f:
    print("config.json model_type", json.load(f).get("model_type"))
ok = ok and config.model_type == "qwen3_5"
sys.exit(0 if ok else 1)
EOF

echo "== sm_80 SASS in the source-built extensions"
for so in $(python -c 'import flash_attn_2_cuda, causal_conv1d_cuda; print(flash_attn_2_cuda.__file__, causal_conv1d_cuda.__file__)'); do
  echo "$so"; cuobjdump --list-elf "$so" | grep -o 'sm_[0-9]*' | sort | uniq -c || status=1
done

echo "== chat template of scripts/run_vision_opd.sh"
template="$repo/chat_templates/perception_chat_template_qwen35.jinja"
grep -n 'CUSTOM_CHAT_TEMPLATE_FILE=' "$repo/scripts/run_vision_opd.sh"
if [[ -f "$template" ]]; then echo "present $template"; sha256sum "$template"; else echo "missing $template"; status=1; fi

echo "== smoke status $status"
exit "$status"
