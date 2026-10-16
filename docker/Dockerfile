# syntax=docker/dockerfile:1.7
# Vision-OPD environment for VLM-VISIONOPD-REPRO-001 (HANDOFF-VLM-VISIONOPD-REPRO-001-R2).
# Runs the README "Environment Setup" steps in order. Build and run commands: docker/BUILD.md.
# The repository stays out of the image. The container mounts the server clone at
# /workspace/harness/vision-opd, and the editable install below points to that path.
# Build from the repository root: docker build -f docker/Dockerfile -t <tag> .

# CUDA 12.8 toolkit with nvcc, for the flash-attn and causal-conv1d source builds.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG MINIFORGE_VERSION=24.11.3-0
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget git build-essential \
 && rm -rf /var/lib/apt/lists/*

# conda for the README "conda create" step.
RUN wget -q -O /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh

# README: conda create -n vision-opd python=3.12 -y
RUN /opt/conda/bin/conda create -n vision-opd python=3.12 -y && /opt/conda/bin/conda clean -afy

# README: conda activate vision-opd (the environment is first on PATH)
ENV CONDA_DEFAULT_ENV=vision-opd
ENV PATH=/opt/conda/envs/vision-opd/bin:/opt/conda/bin:${PATH}

# README: pip install --upgrade pip
RUN pip install --upgrade pip

# README: pip install --no-deps -r requirements.txt
RUN --mount=type=bind,source=requirements.txt,target=/tmp/vision-opd/requirements.txt \
    pip install --no-deps -r /tmp/vision-opd/requirements.txt

# README: pip install -e . --no-deps
# The bind mount puts the build context at the runtime mount path. The image keeps only the
# editable-install metadata in site-packages, not the repository.
RUN --mount=type=bind,source=.,target=/workspace/harness/vision-opd,rw \
    cd /workspace/harness/vision-opd && pip install -e . --no-deps

# README: pip install flash-attn --no-build-isolation
# Source build with SASS for sm_80 (A100) only. FLASH_ATTN_CUDA_ARCHS sets the flash-attn gencode list.
RUN CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 FLASH_ATTN_CUDA_ARCHS=80 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS} \
    pip install flash-attn --no-build-isolation

# README: pip install causal-conv1d==1.6.1 --no-build-isolation
RUN CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS} \
    pip install causal-conv1d==1.6.1 --no-build-isolation

# Record the installed versions. trl is not installed.
RUN mkdir -p /opt/vision-opd && pip freeze > /opt/vision-opd/pip-freeze.txt \
 && ! grep -qi '^trl==' /opt/vision-opd/pip-freeze.txt

WORKDIR /workspace/harness/vision-opd
