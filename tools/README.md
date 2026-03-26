# tools/

Utility scripts and helpers for building, installing, profiling, and linting vLLM.

## Directory Structure

```
tools/
  check_repo.sh
  flashinfer-build.sh
  generate_cmake_presets.py
  generate_versions_json.py
  install_deepgemm.sh
  install_gdrcopy.sh
  install_nixl_from_source_ubuntu.py
  install_torchcodec_rocm.sh
  report_build_time_ninja.py
  ep_kernels/           -- Expert-parallel kernel build scripts (DeepEP, NVSHMEM)
  pre_commit/           -- Pre-commit hook scripts for linting and code generation
  profiler/             -- GPU profiling and visualization tools
  vllm-rocm/            -- ROCm wheel packaging helpers
  vllm-tpu/             -- TPU wheel build script
```

## Top-Level Scripts

### check_repo.sh

Validates that the git repo is clean (no uncommitted changes) and that tags are available. Used to ensure the vLLM version can be determined at build time.

```bash
bash tools/check_repo.sh
```

### flashinfer-build.sh

Builds FlashInfer wheels with ahead-of-time (AOT) compiled kernels. Requires `FLASHINFER_GIT_REF` and `CUDA_VERSION` environment variables. Set `BUILD_WHEEL=false` to install directly instead of producing a wheel.

```bash
FLASHINFER_GIT_REF=v0.2.2 CUDA_VERSION=12.8 bash tools/flashinfer-build.sh
```

### generate_cmake_presets.py

Auto-detects NVCC, Python, compiler caches (ccache/sccache), and CPU core count, then generates a `CMakeUserPresets.json` for building vLLM's C++/CUDA extensions with CMake.

```bash
python tools/generate_cmake_presets.py [--force-overwrite]
```

### generate_versions_json.py

Parses Dockerfile ARG defaults and writes `docker/versions.json` for use with `docker buildx bake`. Pass `--check` to verify the file is in sync (used in CI).

```bash
python tools/generate_versions_json.py          # generate
python tools/generate_versions_json.py --check   # CI validation
```

### install_deepgemm.sh

Clones and builds DeepGEMM from source. Requires CUDA 12.8+. Supports `--wheel-dir` to build a wheel without installing, `--ref` to pin a commit, and `--cuda-version` to override auto-detection.

```bash
bash tools/install_deepgemm.sh
bash tools/install_deepgemm.sh --wheel-dir /tmp/wheels
```

### install_gdrcopy.sh

Downloads and installs the `libgdrapi` package from NVIDIA for GDRCopy support. Must be run as root.

```bash
sudo bash tools/install_gdrcopy.sh Ubuntu22_04 12.8 x64
```

### install_nixl_from_source_ubuntu.py

Builds UCX and NIXL from source, producing a self-contained wheel with bundled UCX libraries (via `auditwheel repair`). Handles caching to avoid redundant rebuilds. Supports `--force-reinstall`.

```bash
python tools/install_nixl_from_source_ubuntu.py
```

### install_torchcodec_rocm.sh

Installs TorchCodec from source for ROCm compatibility. Installs FFmpeg system dependencies automatically if running as root, then builds and verifies the installation.

```bash
bash tools/install_torchcodec_rocm.sh
```

### report_build_time_ninja.py

Parses a `.ninja_log` file and prints a summary of the longest build steps, grouped by file extension (`.cu.o`, `.cpp.o`, `.so`). Shows weighted duration to estimate each step's impact on total wall-clock build time.

```bash
python tools/report_build_time_ninja.py -C cmake-build-release
```

---

## ep_kernels/

Scripts for building expert-parallel (EP) kernels, primarily [DeepEP](https://github.com/deepseek-ai/DeepEP), used for large-scale Mixture-of-Experts deployment as described in the DeepSeek-V3 paper.

See `ep_kernels/README.md` for full details.

| File | Description |
|------|-------------|
| `install_python_libraries.sh` | Downloads NVSHMEM, clones DeepEP, and builds/installs or produces wheels. Accepts `--workspace`, `--mode` (install or wheel), `--deepep-ref`, and `--nvshmem-ver`. |
| `configure_system_drivers.sh` | Configures NVIDIA driver options to enable IBGDA for multi-node EP. Requires root and a reboot. |
| `elastic_ep/install_eep_libraries.sh` | Builds NVSHMEM from source with Elastic EP patches, for the elastic expert-parallel variant. |
| `elastic_ep/eep_nvshmem.patch` | Patch for NVSHMEM to fix re-initialization and double-free issues in Elastic EP mode. |

```bash
# Build DeepEP for Hopper
TORCH_CUDA_ARCH_LIST="9.0" bash tools/ep_kernels/install_python_libraries.sh

# Multi-node setup (root required, then reboot)
sudo bash tools/ep_kernels/configure_system_drivers.sh
```

---

## pre_commit/

Scripts invoked by pre-commit hooks. These are referenced from `.pre-commit-config.yaml` and run automatically on staged files.

| File | Description |
|------|-------------|
| `check_boolean_context_manager.py` | Detects `with a() and b():` -- a common bug where only one context manager is entered. |
| `check_forbidden_imports.py` | Blocks direct use of `pickle`, `base64`, `re`, and `triton` imports. Enforces project alternatives (`pybase64`, `regex`, `vllm.triton_utils`). |
| `check_init_lazy_imports.py` | Ensures `vllm/__init__.py` only imports internal modules inside `typing.TYPE_CHECKING` guards, enforcing lazy loading. |
| `check_spdx_header.py` | Checks and auto-adds the required SPDX license and copyright header to Python files. |
| `check_torch_cuda.py` | Flags direct `torch.cuda.*` API calls. Enforces use of the platform-agnostic `torch.accelerator` API instead. |
| `generate_attention_backend_docs.py` | Parses all attention backend classes via AST and generates a markdown feature-support table for documentation. |
| `generate_nightly_torch_test.py` | Splits `requirements/test.in` into a file without PyTorch-related packages, for nightly PyTorch testing. |
| `mypy.py` | Runs mypy on changed files, grouping them by directory to handle different `follow_imports` settings. |
| `png-lint.sh` | Validates that `*.excalidraw.png` files have embedded Excalidraw scene metadata (for future editing). |
| `shellcheck.sh` | Runs ShellCheck on all `.sh` files in the repo. Auto-installs ShellCheck on Linux x86_64 if not present. |
| `update-dockerfile-graph.sh` | Regenerates the Dockerfile stage dependency graph PNG when `docker/Dockerfile` changes. Requires Docker. |
| `validate_config.py` | Validates that `@config` dataclasses have default values and docstrings for every field. |

---

## profiler/

Tools for profiling GPU kernel execution in vLLM.

### nsys_profile_tools/

Processes NVIDIA Nsight Systems trace files (`.nsys-rep`) and generates kernel-level summaries. Classifies GPU kernels into categories (gemm, attention, MoE, normalization, etc.) and produces CSV and HTML visualizations.

See `profiler/nsys_profile_tools/README.md` for detailed usage and examples.

| File | Description |
|------|-------------|
| `gputrc2graph.py` | Main script. Takes `.nsys-rep` files and produces categorized stacked bar charts (HTML) and kernel-to-category mappings (CSV). |
| `vllm_engine_model.json` | Kernel classification rules for vLLM models (llama, ds, gpt-oss). Maps kernel name patterns to categories. |

### Layerwise Profiling

| File | Description |
|------|-------------|
| `print_layerwise_table.py` | Prints a table of per-layer profiling statistics from a JSON profile dump. |
| `visualize_layerwise_profile.py` | Generates matplotlib visualizations of per-layer profiling data from a JSON profile dump. |

---

## vllm-rocm/

Helpers for building and publishing ROCm-specific vLLM wheels.

| File | Description |
|------|-------------|
| `generate-rocm-wheels-root-index.sh` | Generates a PEP 503 compatible PyPI index on S3 pointing to the latest ROCm wheel version. Supports `--dry-run` and `--version`. |
| `pin_rocm_dependencies.py` | Pins custom ROCm wheel versions (torch, triton, torchvision, amdsmi, etc.) in requirements files so that `pip install vllm` fetches the correct builds. |

```bash
# Preview the index without uploading
bash tools/vllm-rocm/generate-rocm-wheels-root-index.sh --dry-run

# Pin dependencies from built wheels
python tools/vllm-rocm/pin_rocm_dependencies.py /install requirements/rocm.txt
```

---

## vllm-tpu/

| File | Description |
|------|-------------|
| `build.sh` | Builds a `vllm-tpu` wheel. Patches `pyproject.toml` and source files to rename the package from `vllm` to `vllm-tpu`, builds the wheel, then restores all files. Accepts an optional version override argument. |

```bash
bash tools/vllm-tpu/build.sh           # default version
bash tools/vllm-tpu/build.sh 0.8.0     # override version
```
