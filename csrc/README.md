# csrc -- vLLM Native Kernels

This directory contains the C, C++, and CUDA/HIP source code for vLLM's
performance-critical operations. These kernels are compiled into shared
libraries (via PyTorch's extension mechanism) and exposed to Python through
`torch.ops` bindings.

## How Ops Are Registered

Each sub-component has a `torch_bindings.cpp` that registers operations with
PyTorch using `TORCH_LIBRARY_EXPAND`. The main entry point is the top-level
`torch_bindings.cpp`, which registers the bulk of the CUDA ops across several
namespaces:

| Namespace | Purpose |
|-----------|---------|
| `TORCH_EXTENSION_NAME` | Core ops (attention, activations, norms, quantization, sampling, etc.) |
| `_cache_ops` | KV cache management (reshape, swap, concat, gather, quantize) |
| `_cuda_utils` | Device attribute queries |
| `_custom_ar` | Custom all-reduce / collective communication |

Additional `torch_bindings.cpp` files exist in `moe/`, `cpu/`, `rocm/`, and
`libtorch_stable/` for their respective op sets.

Function signatures (declarations) live in `ops.h` and `cache.h`; each `.cu`
file provides the kernel implementation.

## Directory Structure

### Top-level files

| File(s) | Description |
|---------|-------------|
| `torch_bindings.cpp`, `ops.h`, `cache.h` | Op registration and C++ declarations for the main CUDA library. |
| `activation_kernels.cu` | Gated activation functions: SiLU, GELU (exact, tanh, fast, quick), FATReLU, SwigluOAI. Each has an `act_and_mul` fused variant. |
| `layernorm_kernels.cu` | RMS normalization and fused add-RMS-norm supporting 2D/3D/4D tensors. |
| `layernorm_quant_kernels.cu` | Fused RMS-norm + quantization (static FP8, dynamic per-token, per-block). Avoids a separate quantization pass. |
| `pos_encoding_kernels.cu` | Rotary positional embedding (GPT-NeoX and GPT-J styles). |
| `fused_qknorm_rope_kernel.cu` | Fused QK-norm + RoPE kernel that applies normalization and rotary embedding in a single pass. |
| `cache_kernels.cu` | KV cache operations: `swap_blocks`, `reshape_and_cache`, `reshape_and_cache_flash`, `concat_and_cache_mla`, `convert_fp8`, `gather_and_maybe_dequant_cache`, and context-parallelism gather variants. |
| `cache_kernels_fused.cu` | Fused MLA (Multi-head Latent Attention) cache kernels: combined RoPE + concat + cache write for DeepSeek-style MLA. |
| `concat_mla_q.cuh` | Utility for concatenating MLA query components (q_nope and q_pe). |
| `sampler.cu` | Repetition penalty application on logits. |
| `topk.cu` | GPU top-k selection for sparse attention and large-context scenarios. |
| `custom_all_reduce.cu`, `custom_all_reduce.cuh` | Custom all-reduce for multi-GPU tensor parallelism using IPC shared memory. |
| `custom_quickreduce.cu` | ROCm QuickReduce all-reduce wrapper. |
| `dsv3_fused_a_gemm.cu` | DeepSeek V3 fused absorption GEMM for ultra-low-latency decode (SM90+, bf16, 1-16 tokens). |
| `cuda_view.cu` | Creates a CUDA-addressable view of a CPU (pinned) tensor via UVA. |
| `cumem_allocator.cpp` | CUDAPluggableAllocator based on `cuMem*` virtual-memory APIs (supports both NVIDIA and ROCm). |
| `cuda_utils_kernels.cu` | Device attribute query helpers. |

### Utility headers

| File | Description |
|------|-------------|
| `cuda_compat.h` | Macros for NVIDIA/ROCm portability (`VLLM_LDG`, warp size, etc.). |
| `cuda_utils.h` | `get_device_attribute` and shared-memory query wrappers. |
| `cuda_vec_utils.cuh` | Packed vector types and element-wise math for vectorized kernels. |
| `type_convert.cuh` | Scalar type conversion utilities between CUDA/PyTorch types. |
| `dispatch_utils.h` | `VLLM_DISPATCH_FLOATING_TYPES` and related dispatch macros. |
| `cub_helpers.h` | CUB/hipCUB block-reduce wrappers. |
| `launch_bounds_utils.h` | Thread-block launch-bounds calculation helpers. |

### `attention/`

Paged attention kernels and supporting utilities.

- `paged_attention_v1.cu`, `paged_attention_v2.cu` -- PagedAttention V1 (single-pass) and V2 (two-pass with split-K reduction). Support FP16, BF16, and FP8 KV caches with optional ALiBi slopes and block-sparse patterns.
- `merge_attn_states.cu` -- Merges partial attention outputs from split-KV computations using the log-sum-exp correction (Section 2.2 of arXiv:2501.01005).
- `vertical_slash_index.cu` -- Index conversion for vertical-slash sparse attention patterns.
- `attention_kernels.cuh`, `attention_utils.cuh`, `attention_generic.cuh` -- Shared kernel templates and warp-reduce utilities.
- `dtype_*.cuh` -- Per-dtype specializations (float16, bfloat16, float32, fp8).

#### `attention/mla/`

CUTLASS-based Multi-head Latent Attention (MLA) decode kernel for SM100 (Blackwell). Uses TMA warp-specialized kernels with custom tile schedulers.

### `quantization/`

Weight and activation quantization kernels, organized by method.

| Subdirectory | Description |
|-------------|-------------|
| `w8a8/fp8/` | FP8 (E4M3/E5M2) quantization: static, dynamic per-tensor, dynamic per-token. Separate NVIDIA and AMD codepaths in `nvidia/` and `amd/`. |
| `w8a8/int8/` | INT8 symmetric quantization with static or dynamic scaling. |
| `w8a8/cutlass/` | CUTLASS-based scaled matrix multiply for W8A8. Covers SM75 through SM120 (C2x and C3x APIs). Includes `moe/` subdirectory for grouped GEMM used in fused MoE. |
| `fp4/` | NVFP4 (4-bit floating point) block-scaled quantization and GEMM kernels for SM100+. Includes expert-specialized MoE variants. |
| `cutlass_w4a8/` | CUTLASS W4A8 mixed-precision GEMM: 4-bit weights with 8-bit activations. |
| `awq/` | AWQ dequantization and GEMM kernels. |
| `gptq/` | GPTQ/ExLlama quantized GEMM (2/3/4/8-bit support). |
| `gptq_allspark/` | AllSpark W8A16 fused GEMM and weight reordering for Ampere. |
| `marlin/` | Marlin optimized quantized GEMM supporting GPTQ, AWQ, FP8, NVFP4, and MXFP4 formats. Includes weight repacking utilities. |
| `machete/` | Machete: CUTLASS-based mixed-precision GEMM for Hopper. Successor to Marlin with prepacked weight layouts and code-generated kernel instantiations. |
| `gguf/` | GGML/GGUF format dequantization, matrix-vector multiply, matrix-matrix multiply, and MoE support. |
| `hadamard/` | Hadamard transform kernels (used by HadaCore / QuIP# quantization). |
| `fused_kernels/` | Fused layernorm + dynamic per-token quantization in a single kernel launch. |
| `activation_kernels.cu` | Quantized activation functions (FP8 SiLU-and-mul, etc.). |

### `moe/`

Mixture-of-Experts kernels.

- `topk_softmax_kernels.cu` -- Top-k softmax and top-k sigmoid gating.
- `grouped_topk_kernels.cu` -- Grouped top-k routing (DeepSeek V3 style).
- `moe_align_sum_kernels.cu` -- Token-to-expert alignment, padding, and partial-result summation.
- `moe_permute_unpermute_op.cu` and `permute_unpermute_kernels/` -- Efficient token permutation/unpermutation for expert dispatch.
- `moe_wna16.cu` -- WnA16 quantized MoE GEMM.
- `marlin_moe_wna16/` -- Marlin-based MoE GEMM with WnA16 quantization (code-generated).
- `mxfp8_moe/` -- MXFP8 (microscaling FP8) block-scaled grouped GEMM and expert quantization for SM100+.
- `dsv3_router_gemm*.cu`, `router_gemm.cu`, `gpt_oss_router_gemm.cu` -- Optimized router GEMM kernels (SM90+).
- `torch_bindings.cpp` -- Op registration for the MoE extension.

### `cpu/`

CPU-optimized kernels for inference without a GPU.

- `cpu_attn.cpp`, `cpu_attn_*.hpp` -- CPU paged attention with backends for x86 (AVX2/AVX-512/AMX), ARM (NEON/BFMMLA), RISC-V (vector), and IBM (VSX/VXE).
- `layernorm.cpp` -- CPU RMS normalization.
- `activation.cpp` -- CPU activation functions.
- `pos_encoding.cpp` -- CPU rotary embedding.
- `cpu_fused_moe.cpp` -- CPU fused MoE.
- `cpu_wna16.cpp` -- CPU WnA16 quantized GEMM.
- `mla_decode.cpp` -- CPU MLA decode attention.
- `dnnl_helper.*`, `dnnl_kernels.cpp` -- oneDNN integration for INT8/FP8 scaled matmul.
- `micro_gemm/` -- Micro-kernel GEMM implementations (AMX, vectorized).
- `sgl-kernels/` -- SGL FP8/INT8 GEMM and MoE kernels for CPU.
- `shm.cpp` -- Shared-memory based collective operations (allreduce, gather, send/recv).
- `torch_bindings.cpp` -- Op registration for the CPU extension.

### `rocm/`

ROCm (AMD GPU) specific kernels.

- `attention.cu` -- ROCm paged attention implementation.
- `skinny_gemms.cu` -- Optimized skinny GEMM and matrix-vector kernels (`LLMM1`, `wvSplitK`, etc.).
- `torch_bindings.cpp` -- Op registration for the ROCm extension.

### `cutlass_extensions/`

Shared CUTLASS utilities used by the quantization and MoE kernels.

- `common.hpp`, `common.cpp` -- CUTLASS error-checking macros, SM version queries, and architecture-guarded kernel wrappers.
- `cute_utils.cuh` -- CuTe layout utilities.
- `vllm_numeric_conversion.cuh` -- Custom numeric type conversion (extends CUTLASS's built-in converters for vLLM's type pairs).
- `vllm_custom_types.cuh` -- Custom scalar type registration for CUTLASS.
- `vllm_collective_builder.cuh` -- Custom collective builder overrides.
- `torch_utils.hpp` -- PyTorch-to-CUTLASS type mapping.
- `epilogue/` -- Custom epilogues for scaled matmul with broadcast loads (C2x and C3x APIs).

### `libtorch_stable/`

Ops registered using PyTorch's stable ABI (`STABLE_TORCH_LIBRARY`) for
forward-compatible binary distribution.

- `permute_cols.cu` -- Column permutation kernel.
- `quantization/w8a8/` -- Per-token-group FP8 and INT8 quantization kernels with UE8M0 scale packing and TMA alignment for DeepGEMM compatibility.

### `quickreduce/`

ROCm QuickReduce all-reduce implementation using HIP IPC for fast multi-GPU
reductions.

### `core/`

Shared C++ utilities.

- `registration.h` -- `TORCH_LIBRARY_EXPAND` and `REGISTER_EXTENSION` macros.
- `scalar_type.hpp` -- `ScalarType` class representing sub-byte and exotic numeric types (used for quantization format dispatch).
- `math.hpp` -- Compile-time math helpers (ceil_div, etc.).
- `batch_invariant.hpp` -- Utilities for batch-invariant tensor operations.
- `exception.hpp` -- Custom exception types.

## Build Integration

These sources are compiled by `setup.py` / CMake at the project root. Key
build-time variables:

- `USE_ROCM` -- Defined when building for AMD GPUs; switches between CUDA and HIP codepaths.
- `TORCH_EXTENSION_NAME` -- Set by the build system; determines the Python module name for the compiled extension.
- Architecture flags (`-gencode`, `--offload-arch`) control which GPU architectures are targeted and which kernels are conditionally compiled.
