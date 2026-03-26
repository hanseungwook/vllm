# docker/

This directory contains the Dockerfiles and build configuration for vLLM across
all supported platforms and accelerators.

## Directory contents

| File | Platform / Purpose |
|---|---|
| `Dockerfile` | **NVIDIA CUDA (GPU)** -- the primary Dockerfile. Produces images for the OpenAI-compatible server, testing, and development. |
| `Dockerfile.cpu` | **CPU** (x86_64 and aarch64). Includes a ZenDNN/zentorch variant for AMD CPUs. |
| `Dockerfile.rocm_base` | **AMD ROCm** base image -- builds PyTorch, Triton, Flash Attention, AITER, and other ROCm-specific dependencies from source. |
| `Dockerfile.rocm` | **AMD ROCm** vLLM image -- builds on top of the ROCm base image to compile and install vLLM. |
| `Dockerfile.xpu` | **Intel XPU** (Data Center and Arc GPUs via oneAPI/SYCL). |
| `Dockerfile.tpu` | **Google Cloud TPU** via PyTorch/XLA. |
| `Dockerfile.ppc64le` | **IBM POWER** (ppc64le), UBI9-based. |
| `Dockerfile.s390x` | **IBM Z** (s390x), UBI9-based. |
| `Dockerfile.nightly_torch` | **(Deprecated)** Nightly PyTorch builds. Use the main `Dockerfile` with `--build-arg PYTORCH_NIGHTLY=1` instead. |
| `docker-bake.hcl` | Docker Buildx Bake configuration for the CUDA Dockerfile. Defines build groups, variables, and OCI labels. |
| `versions.json` | Auto-generated file that mirrors the `ARG` defaults from `Dockerfile` for use with `docker buildx bake`. Do not edit manually. |

## Build targets (main Dockerfile)

The primary `Dockerfile` is a multi-stage build with the following key targets:

| Target | Description |
|---|---|
| `vllm-openai` | **(default)** Production image with the OpenAI-compatible API server. Entrypoint: `vllm serve`. |
| `vllm-sagemaker` | Variant of the OpenAI image configured for AWS SageMaker. |
| `test` | Image with test dependencies and the test suite included. Used in CI. |
| `dev` | Development image with the full source tree, dev tools, and an editable install. |
| `build` | Intermediate stage that builds the vLLM wheel. |

The CPU Dockerfile (`Dockerfile.cpu`) provides its own set of targets:

| Target | Description |
|---|---|
| `vllm-openai` | Production CPU inference server. |
| `vllm-openai-zen` | CPU image with AMD ZenDNN (`vllm[zen]`). x86_64 only. |
| `vllm-test` | CPU test image. |
| `vllm-dev` | CPU development image. |

## Quick start

All commands assume you are at the repository root.

### CUDA (GPU)

```bash
# Build the default OpenAI server image
docker build -t vllm:openai -f docker/Dockerfile --target vllm-openai .

# Build the test image
docker build -t vllm:test -f docker/Dockerfile --target test .
```

### Using Docker Buildx Bake

Bake provides pre-configured build targets with sensible defaults for parallel
jobs, CUDA architectures, and OCI labels.

```bash
# Build the default target (openai)
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json

# Build a specific target
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json test

# Build the Ubuntu 24.04 variant
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json openai-ubuntu2404

# Preview the resolved build configuration without building
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json --print
```

Bake variables can be overridden on the command line:

```bash
docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json \
  --set '*.args.MAX_JOBS=8' \
  --set '*.args.TORCH_CUDA_ARCH_LIST=9.0 10.0'
```

### CPU

```bash
# Build for the current platform
docker build -t vllm:cpu -f docker/Dockerfile.cpu --target vllm-openai .

# Cross-build for ARM64
docker buildx build --platform=linux/arm64 -t vllm:cpu-arm64 \
  -f docker/Dockerfile.cpu --target vllm-openai .
```

### ROCm (AMD GPU)

```bash
# Build the base image first
docker build -t rocm/vllm-dev:base -f docker/Dockerfile.rocm_base .

# Then build vLLM on top of it
docker build -t vllm:rocm -f docker/Dockerfile.rocm .
```

### Intel XPU

```bash
docker build -t vllm:xpu -f docker/Dockerfile.xpu --target vllm-openai .
```

### TPU

```bash
docker build -t vllm:tpu -f docker/Dockerfile.tpu .
```

### ppc64le / s390x

```bash
docker build -t vllm:ppc64le -f docker/Dockerfile.ppc64le --target vllm-openai .
docker build -t vllm:s390x  -f docker/Dockerfile.s390x .
```

## Key build arguments (CUDA Dockerfile)

| Argument | Default | Description |
|---|---|---|
| `CUDA_VERSION` | `12.9.1` | CUDA toolkit version. |
| `PYTHON_VERSION` | `3.12` | Python version. |
| `UBUNTU_VERSION` | `22.04` | Ubuntu version for the final image. The build stage uses Ubuntu 20.04 for glibc compatibility. |
| `TORCH_CUDA_ARCH_LIST` | `7.0 7.5 8.0 8.9 9.0 10.0 12.0` | CUDA compute capabilities to compile for. |
| `MAX_JOBS` | `2` | Parallel compilation jobs. |
| `NVCC_THREADS` | `8` | Threads per nvcc invocation. |
| `PYTORCH_NIGHTLY` | (unset) | Set to `1` to build against PyTorch nightly instead of the stable release. |
| `BUILD_BASE_IMAGE` | `nvidia/cuda:...-devel-ubuntu20.04` | Override the build base image (useful for private registries). |
| `FINAL_BASE_IMAGE` | `nvidia/cuda:...-base-ubuntu22.04` | Override the runtime base image. |
| `INSTALL_KV_CONNECTORS` | `false` | Include KV-connector dependency libraries. |

## versions.json

`versions.json` is auto-generated from the `ARG` defaults in `Dockerfile` by
the script `tools/generate_versions_json.py`. It is consumed by
`docker buildx bake` as a variable file so that bake builds stay in sync with
the Dockerfile without manual duplication.

To regenerate after editing Dockerfile ARGs:

```bash
python tools/generate_versions_json.py
```

To verify it is up to date (used in CI):

```bash
python tools/generate_versions_json.py --check
```

## Further reading

- Dockerfile stage diagram: `docs/assets/contributing/dockerfile-stages-dependency.png`
- Dockerfile documentation: `docs/contributing/dockerfile/dockerfile.md`
