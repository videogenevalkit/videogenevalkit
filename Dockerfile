# videogenevalkit Docker image — CUDA 12.1 + conda-pack'd env + toolkit
#
# Build:
#   1. First build the env tarball on a machine where the toolkit is installed:
#        conda-pack -p /path/to/working/env -o /path/to/videvalkit-env.tar.gz
#      (or fetch it from HF: hf download videogenevalkit/env-tarball videvalkit-env.tar.gz)
#   2. Put the tarball in the repo root next to this Dockerfile (or pass --build-arg ENV_TARBALL=...)
#   3. docker build -t videogenevalkit/videogenevalkit:0.1.0 .
#
# Run (single GPU):
#   docker run --rm --gpus all -v $PWD:/workspace \
#     videogenevalkit/videogenevalkit:0.1.0 \
#     videvalkit eval --bench worldjen --videos /workspace/videos --workspace /workspace/runs/first
#
# Run (specific GPUs):
#   docker run --rm --gpus '"device=2,3"' ... videogenevalkit/videogenevalkit:0.1.0 ...

# --- Base: CUDA 12.1 runtime on Ubuntu 22.04 (matches torch 2.3.1+cu121) ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Build args: where the conda-packed env tarball is + where to unpack it inside the image
ARG ENV_TARBALL=videvalkit-env.tar.gz
ARG ENV_PREFIX=/opt/videvalkit-env

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIDEVALKIT_CACHE_HOME=/root/.cache/videvalkit \
    PATH=${ENV_PREFIX}/bin:${PATH} \
    LD_LIBRARY_PATH=${ENV_PREFIX}/lib:${LD_LIBRARY_PATH}

# --- System libs needed at runtime (ffmpeg already in env; this is for native deps) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Unpack the conda-pack tarball into ENV_PREFIX ---
# Layered so re-running the build with a different src/ doesn't re-extract this.
COPY ${ENV_TARBALL} /tmp/env.tar.gz
RUN mkdir -p ${ENV_PREFIX} \
    && tar xzf /tmp/env.tar.gz -C ${ENV_PREFIX} \
    && rm /tmp/env.tar.gz \
    && ${ENV_PREFIX}/bin/conda-unpack \
    && echo "env unpacked to ${ENV_PREFIX}"

# --- Copy + install the toolkit source ---
WORKDIR /opt/videogenevalkit
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY examples/ ./examples/
COPY envs/ ./envs/
COPY validation/expected/ ./validation/expected/

RUN ${ENV_PREFIX}/bin/pip install --no-deps -e . \
    && echo "videvalkit installed"

# --- Smoke check (cached at build time) ---
RUN ${ENV_PREFIX}/bin/python -c "import torch, videvalkit; \
    print(f'torch={torch.__version__} cuda_built={torch.cuda.is_available()}'); \
    print(f'videvalkit={videvalkit.__version__}')"

# --- Verify CLI entry point ---
RUN videvalkit list benchmarks | head -10

# --- Default entrypoint exposes `videvalkit` ---
ENTRYPOINT ["videvalkit"]
CMD ["--help"]

# Image labels (cosmetic — show up in `docker inspect`)
LABEL org.opencontainers.image.title="videogenevalkit" \
      org.opencontainers.image.description="Unified evaluation toolkit for text-to-video generation" \
      org.opencontainers.image.source="https://github.com/videogenevalkit/videogenevalkit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="0.1.0"
