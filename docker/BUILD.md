# Vision-OPD container on a100 (VLM-VISIONOPD-REPRO-001)

This file belongs to HANDOFF-VLM-VISIONOPD-REPRO-001-R2. ET wrote the Dockerfile. The researcher builds the image and starts the container. Revision R2 builds no image and runs no container.

## Image contents

`docker/Dockerfile` runs the README "Environment Setup" steps in this order:

1. `conda create -n vision-opd python=3.12 -y` (Miniforge `24.11.3-0` in `/opt/conda`).
2. `conda activate vision-opd`: the environment `bin` directory is first on `PATH`.
3. `pip install --upgrade pip`.
4. `pip install --no-deps -r requirements.txt`. The file is unchanged.
5. `pip install -e . --no-deps`.
6. `pip install flash-attn --no-build-isolation`, with `TORCH_CUDA_ARCH_LIST=8.0`, `FLASH_ATTN_CUDA_ARCHS=80`, and `FLASH_ATTENTION_FORCE_BUILD=TRUE`.
7. `pip install causal-conv1d==1.6.1 --no-build-isolation`, with `TORCH_CUDA_ARCH_LIST=8.0` and `CAUSAL_CONV1D_FORCE_BUILD=TRUE`.

The image does not install trl. The last build step fails if `pip freeze` lists trl. `/opt/vision-opd/pip-freeze.txt` records the installed versions. The README does not pin the flash-attn version, so pip installs the newest release at build time.

Base image: `nvidia/cuda:12.8.1-devel-ubuntu22.04`. It contains the CUDA 12.8 toolkit with `nvcc`. To pin the digest, run `docker pull nvidia/cuda:12.8.1-devel-ubuntu22.04`, then `docker image inspect --format '{{index .RepoDigests 0}}' nvidia/cuda:12.8.1-devel-ubuntu22.04`. Then build with `--build-arg BASE_IMAGE=nvidia/cuda@sha256:<digest>`. Record the digest in the experiment report.

## Editable install: choice

The repository stays out of the image. Step 5 uses a BuildKit bind mount of the build context at `/workspace/harness/vision-opd`, which is the runtime mount path of the server clone. The image keeps only the editable-install metadata in site-packages. At run time, `import verl` resolves to the mounted clone. `scripts/run_vision_opd.sh` also sets `PYTHONPATH` to the repository root.

Do not move the clone. If the clone path changes, run `pip install -e . --no-deps` again in the container.

## Build

Run on the a100 host from the server clone. BuildKit is necessary (Docker 23 or later uses it by default).

```bash
cd /home/vision/dongjun/harness/vision-opd
git rev-parse HEAD
docker pull nvidia/cuda:12.8.1-devel-ubuntu22.04
docker image inspect --format '{{index .RepoDigests 0}}' nvidia/cuda:12.8.1-devel-ubuntu22.04
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=nvidia/cuda@sha256:<digest> \
  --build-arg MAX_JOBS=8 \
  -t vision-opd:exp-VLM-VISIONOPD-REPRO-001-$(git rev-parse --short HEAD) .
```

The build needs no GPU. `MAX_JOBS` limits the parallel compile jobs of flash-attn and causal-conv1d. Each job can use several GB of memory.

## Run

```bash
docker run -d --name <container name> \
  --gpus all --ipc=host --shm-size=64g \
  -v /home/vision/dongjun:/workspace \
  -w /workspace/harness/vision-opd \
  -e HF_HOME=/workspace/harness/data/vision-opd/hf_cache \
  -e HF_HUB_OFFLINE=0 \
  vision-opd:exp-VLM-VISIONOPD-REPRO-001-<short sha> sleep infinity
```

1. `--ipc=host` gives the container the host IPC namespace and `/dev/shm`. With `--ipc=host`, Docker does not apply `--shm-size`. Without `--ipc=host`, `--shm-size=64g` sets `/dev/shm`.
2. `--gpus all` shows all 8 GPUs. Use `--gpus '"device=<list>"'` for a subset.
3. The dataset is in `/workspace/harness/data/vision-opd/Vision-OPD-6K`. The model is in `/workspace/harness/data/vision-opd/Qwen3.5-4B`.
4. If the container does not start with the message "unsatisfied condition: cuda>=12.8", add `-e NVIDIA_DISABLE_REQUIRE=1`. The base image sets `NVIDIA_REQUIRE_CUDA`, and the NVIDIA Container Toolkit compares it with the host driver.

Smoke check in the container (1 GPU or more): `bash docker/smoke_check.sh`.

## Driver compatibility (gate G7)

Host: NVIDIA driver 535.274.02, CUDA 12.2 (TASK-VLM-VISIONOPD-REPRO-001-002, `ENV_CHECK.txt`). GPU: A100-SXM4-80GB, compute capability 8.0.

### CUDA minor version compatibility

1. An application built with any CUDA 12.x toolkit runs on a Linux driver 525.60.13 or later [S1, S2]. The driver 535.274.02 meets this condition.
2. Limitation: the driver cannot JIT-compile PTX that a newer toolkit made. Such code fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` [S1]. Each GPU kernel must therefore contain SASS for sm_80, or it must be compiled to a cubin in user space.
3. Limitation: an API that needs a newer driver returns an error only when the program calls it [S1].

### Component analysis

| Component (pin) | GPU code | Expected under driver 535 |
|---|---|---|
| torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0 (cu128 wheels, CUDA runtime 12.8.90, cuBLAS 12.8.4.1, cuDNN 9.10.2.21) | Prebuilt, SASS for sm_80 in the cu128 arch list [S3] | Runs through minor version compatibility |
| nvidia-nccl-cu12 2.27.5 | Prebuilt | Runs through minor version compatibility |
| vllm 0.18.0 | Prebuilt kernels with sm_80 SASS [S4] | Runs through minor version compatibility |
| flashinfer-python 0.6.6 | JIT compile with the image `nvcc` 12.8 to a cubin | Runs. It needs `CUDA_HOME` and `nvcc` in the image. The base image has both |
| flash-attn (unpinned), causal-conv1d 1.6.1 | Source build in the image with `nvcc` 12.8, SASS sm_80 only [S5, S6] | Runs. No PTX in the binary |
| triton 3.6.0 | JIT: Triton compiles PTX to a cubin with its bundled `ptxas`, then loads the cubin | Runs if the bundled `ptxas` belongs to CUDA 12.x. If it belongs to CUDA 13, the cubin needs a newer driver. Then set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (the image `ptxas` 12.8) [S7] |
| cuda-python, cuda-bindings 12.9.4 | Driver API bindings | Import works on a 12.x driver. A call to an API that driver 535 does not have fails at call time [S8] |
| cupy-cuda12x 14.0.1 | NVRTC compile to a cubin | Runs through minor version compatibility |
| nvidia-cutlass-dsl 4.4.2, quack-kernels 0.3.9 | CuTe DSL JIT. The kernels target sm_90 and later | Not expected on A100. Not verified |

ET found no pinned component that needs a driver upgrade or a different wheel index. ET did not run the stack. `docker/smoke_check.sh` checks the open items at run time:

1. It sets `CUDA_DISABLE_PTX_JIT=1`. A kernel without sm_80 SASS then fails, also when the driver could JIT it.
2. It runs one flash-attn, one causal-conv1d, and one Triton kernel.
3. It lists the SASS architectures of the two source-built extensions with `cuobjdump --list-elf`.
4. It prints the loaded `libcuda` path and the driver version that torch sees.

### Fallback without a driver upgrade: CUDA forward compatibility

The `nvidia/cuda` base images install the `cuda-compat` package (user-mode driver of CUDA 12.8 in `/usr/local/cuda/compat`) [S9]. Forward compatibility is supported on data center GPUs such as A100 [S1]. On this a100 host, vLLM with a CUDA 13 toolchain failed with "The NVIDIA driver on your system is too old (found version 12020)". It ran after the cuda-compat 13.0 libraries were added to `LD_LIBRARY_PATH` (faculty-opsd commit 579d4e5, TASK-VLM-COMPODISTILL-SD-001-018 to -020). If a component fails under minor version compatibility, start the process with `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH`. Recent NVIDIA Container Toolkit versions can add these libraries automatically. The smoke check prints which `libcuda` the process loads.

### Sources

ET could not open web pages in the R2 session. The URLs below are the sources of the statements. ET did not read them again in R2.

- [S1] NVIDIA, CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/
- [S2] NVIDIA, CUDA Toolkit Release Notes, table "CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility": https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- [S3] PyTorch, CUDA arch list of the manywheel builds: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh
- [S4] vLLM, CUDA architectures of the wheel build: https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt
- [S5] flash-attention `setup.py` (`FLASH_ATTN_CUDA_ARCHS`, `FLASH_ATTENTION_FORCE_BUILD`): https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
- [S6] causal-conv1d `setup.py` (`CAUSAL_CONV1D_FORCE_BUILD`, gencode list): https://github.com/Dao-AILab/causal-conv1d/blob/main/setup.py
- [S7] Triton, `TRITON_PTXAS_PATH`: https://github.com/triton-lang/triton
- [S8] NVIDIA cuda-python, cuda.bindings documentation: https://nvidia.github.io/cuda-python/
- [S9] NVIDIA container images for CUDA (`cuda-compat`, `NVIDIA_REQUIRE_CUDA`): https://gitlab.com/nvidia/container-images/cuda
