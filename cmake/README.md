# cmake/ Directory

This directory contains CMake modules and helper scripts used by the top-level
`CMakeLists.txt` to build vLLM's native extensions. It is split into two
top-level modules, one helper script, and a subdirectory for external
(vendored/fetched) project definitions.

## Directory Structure

```
cmake/
  cpu_extension.cmake          # CPU-only backend build logic
  utils.cmake                  # Shared CMake utility functions and macros
  hipify.py                    # CUDA-to-HIP source conversion script
  external_projects/
    flashmla.cmake             # FlashMLA attention library
    qutlass.cmake              # QuTLASS quantized GEMM library
    triton_kernels.cmake       # OpenAI Triton kernels
    vllm_flash_attn.cmake      # vLLM fork of Flash Attention (FA2/FA3/FA4)
```

## Top-Level Modules

### utils.cmake

Shared utility functions and macros used by both the GPU and CPU build paths.
Included early from the top-level `CMakeLists.txt` (`include(cmake/utils.cmake)`).

Key contents:

- **`find_python_from_executable`** -- Locates a Python interpreter matching a
  given executable and validates it against a list of supported versions.
- **`run_python`** -- Executes an arbitrary Python expression and captures its
  stdout.
- **`append_cmake_prefix_path`** -- Imports a Python package path into
  `CMAKE_PREFIX_PATH` (used to find the torch CMake config).
- **`hipify_sources_target`** -- Creates a custom target that runs `hipify.py`
  to convert CUDA `.cu` files into HIP `.hip` files for ROCm builds.
- **`get_torch_gpu_compiler_flags`** -- Queries PyTorch for the recommended
  NVCC or HIP compiler flags.
- **`vllm_prepare_torch_gomp_shim`** -- Finds the `libgomp` shipped inside the
  PyTorch wheel and creates a shim directory with conventional symlink names so
  the linker can find it.
- **CUDA architecture helpers** -- A collection of functions for manipulating
  `-gencode` flags:
  - `clear_cuda_arches` / `extract_unique_cuda_archs_ascending`
  - `set_gencode_flag_for_srcs` / `set_gencode_flags_for_srcs`
  - `cuda_archs_loose_intersection` -- Computes a "loose intersection" between
    the architectures a kernel source supports and the architectures the user is
    building for, handling `+PTX` and architecture-family suffixes (`a`, `f`).
- **`override_gpu_arches`** -- Filters detected GPU architectures against the
  set supported by vLLM (used for HIP/ROCm).
- **`define_extension_target`** -- Central function that defines a Python
  extension module target (`.so`). Handles language selection (CUDA/HIP/CXX),
  hipification, stable ABI, architecture properties, and install rules.

### cpu_extension.cmake

Included when `VLLM_TARGET_DEVICE` is `cpu`. Handles the entire CPU backend
build:

1. **ISA detection** -- Reads `/proc/cpuinfo` (Linux) or `sysctl` (macOS) to
   detect the instruction sets available on the host: x86-64 (AVX2, AVX-512,
   AMX), ARMv8/NEON (with optional BF16), Apple Silicon, IBM POWER9/10/11,
   IBM S390x, and RISC-V Vector.
2. **Compiler flag selection** -- Assembles per-ISA compile flag lists
   (`CXX_COMPILE_FLAGS_AVX512_AMX`, `CXX_COMPILE_FLAGS_AVX2`, etc.).
3. **oneDNN** -- Fetches Intel oneDNN (v3.10 for x86, a pinned commit for
   AArch64) via `FetchContent` and builds it as a static library for GEMM
   kernels. On AArch64 it also fetches and builds the Arm Compute Library (ACL
   v52.6.0) as oneDNN's backend.
4. **Extension targets** -- Defines up to three `_C` extension libraries on
   x86 (`_C` for AMX, `_C_AVX512`, `_C_AVX2`) or a single `_C` on other
   architectures, each compiled with the appropriate flags.

### hipify.py

A command-line wrapper around PyTorch's `hipify_python` preprocessor. It is
invoked as a build step (via the `hipify_sources_target` function in
`utils.cmake`) to translate CUDA source files into HIP source files when
building for ROCm/AMD GPUs.

Usage (called automatically by CMake):

```
python cmake/hipify.py -p <project_dir> -o <output_dir> <source files...>
```

## External Projects (cmake/external_projects/)

Each file in this subdirectory uses CMake `FetchContent` to download (or use a
local checkout of) a third-party library, compile its CUDA/C++ sources, and
install the resulting artifacts into the vLLM wheel. All of them support a
`*_SRC_DIR` environment variable for local development.

### vllm_flash_attn.cmake

Fetches the [vLLM fork of Flash Attention](https://github.com/vllm-project/flash-attention)
(pinned to a specific commit). Builds Flash Attention 2 and 3 as native
extension components (`_vllm_fa2_C`, `_vllm_fa3_C`) and installs FA4 CuteDSL
Python files (`_vllm_fa4_cutedsl_C`) with import-path rewriting. This file must
be included last because Flash Attention's CMake definitions share the global
macro namespace with vLLM.

Local override: `VLLM_FLASH_ATTN_SRC_DIR`

### flashmla.cmake

Fetches [FlashMLA](https://github.com/vllm-project/FlashMLA) (Multi-head
Latent Attention). Requires NVIDIA Hopper (sm90a) or newer and CUDA >= 12.3.
Builds two extension targets (`_flashmla_C`, `_flashmla_extension_C`) covering
dense decode, sparse decode, and sparse prefill kernels for sm90 and sm100.
Vendors the Python interface file with import rewrites.

Local override: `FLASH_MLA_SRC_DIR`

### qutlass.cmake

Fetches [QuTLASS](https://github.com/IST-DASLab/qutlass), a quantized GEMM
library built on CUTLASS. Requires CUDA >= 12.8 and sm120a or sm100a/sm100f
architectures. Rather than creating a separate extension target, it appends its
sources directly to the main `_C` target.

Local override: `QUTLASS_SRC_DIR`

### triton_kernels.cmake

Fetches the `triton_kernels` Python package from the
[Triton repository](https://github.com/triton-lang/triton) (default tag
`v3.6.0`). This is a pure-Python install -- no native compilation. The `.py`
files are copied into `vllm/third_party/triton_kernels/` at install time.

Local override: `TRITON_KERNELS_SRC_DIR`
