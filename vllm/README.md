# vllm/ -- Main Source Package

This is the core Python package for vLLM, a high-throughput and memory-efficient
inference engine for large language models. This document maps out the directory
structure so developers can quickly find the code they need.

## Top-Level Files

| File | Purpose |
|---|---|
| `__init__.py` | Package root. Exports the public API (`LLM`, `SamplingParams`, `RequestOutput`, engine classes, etc.) via lazy imports. |
| `envs.py` | Centralized registry of every `VLLM_*` environment variable with types and defaults. |
| `env_override.py` | Imported before all other modules to set CUDA compatibility paths and library overrides early. |
| `config/` | All configuration dataclasses (`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`, `LoRAConfig`, `SpeculativeConfig`, etc.), split into one file per concern. |
| `version.py` | Exposes `__version__` and `__version_tuple__`. |
| `sampling_params.py` | `SamplingParams` dataclass -- temperature, top-k/p, penalties, structured output options, and everything else that controls generation. |
| `pooling_params.py` | `PoolingParams` for embedding/classification/scoring requests. |
| `outputs.py` | Output dataclasses: `RequestOutput`, `CompletionOutput`, `EmbeddingOutput`, `ScoringOutput`, `ClassificationOutput`, and their request-level wrappers. |
| `sequence.py` | `IntermediateTensors` for passing hidden states between pipeline-parallel stages. |
| `tasks.py` | Type definitions for supported tasks: `GenerationTask`, `PoolingTask`, `FrontendTask`. |
| `inputs/` | Input type definitions (`PromptType`, `TextPrompt`, `TokensPrompt`) and preprocessing logic for the engine and LLM entrypoints. |
| `exceptions.py` | Custom exception hierarchy (`VLLMValidationError`, `VLLMNotFoundError`). |
| `logger.py` | Logging configuration, colored formatters, and the `init_logger` factory. |
| `logits_process.py` | `LogitsProcessor` type alias and built-in processors (e.g., bad-words filtering). |
| `logprobs.py` | Types for prompt and sample log-probability data. |
| `forward_context.py` | Thread-local context manager that carries batch metadata, attention state, and timing info through a forward pass. |
| `connections.py` | HTTP/S connection helpers with retry and backoff for downloading models and assets. |
| `scalar_type.py` | Python mirror of the C++ `ScalarType` class for custom quantization bit-widths. |
| `beam_search.py` | Beam search implementation on top of the engine. |
| `model_inspection.py` | Utilities for printing model architecture info (quantization schemes, layer structure). |
| `scripts.py` | Legacy CLI entry point; delegates to `entrypoints.cli.main`. |
| `_custom_ops.py`, `_oink_ops.py`, `_aiter_ops.py`, `_xpu_ops.py` | Thin Python wrappers around C++/CUDA custom operators. |

## Major Subsystems

### `entrypoints/`

User-facing APIs and servers.

- **`llm.py`** -- The offline `LLM` class for batch inference.
- **`openai/`** -- OpenAI-compatible HTTP server (`api_server.py`) with sub-packages for chat completions, text completions, embeddings, responses, speech-to-text, and real-time streaming.
- **`cli/`** -- The `vllm` command-line interface (`main.py`): `serve`, `run-batch`, `benchmark`, `collect-env`, `launch`, etc.
- **`grpc_server.py`** -- gRPC serving endpoint.
- **`mcp/`** -- Model Context Protocol (MCP) tool server integration.
- **`api_server.py`** -- Minimal demo/benchmark server (not for production).
- **`launcher.py`** -- Helpers for starting HTTP/gRPC servers.
- **`chat_utils.py`** -- Chat template application and message formatting.

### `engine/`

The inference engine layer that ties scheduling, model execution, and output processing together.

- **`llm_engine.py`** -- Synchronous `LLMEngine`.
- **`async_llm_engine.py`** -- `AsyncLLMEngine` for concurrent request handling.
- **`arg_utils.py`** -- `EngineArgs` / `AsyncEngineArgs` parsing and validation.
- **`protocol.py`** -- Engine protocol (interface) definitions.

### `v1/`

The next-generation (V1) engine architecture, designed for higher performance. Mirrors and replaces much of the legacy `engine/` path.

- **`engine/`** -- V1 async engine, coordinator, core scheduling loop, detokenizer, input/output processors.
- **`core/`** -- KV cache management (block pool, cache manager, cache coordinator), encoder cache, and the scheduler (`sched/`).
- **`worker/`** -- GPU/CPU/TPU/XPU model runners and workers, input batching, LoRA and KV connector mixins, CUDA graph support for encoders.
- **`executor/`** -- Process orchestration: uniproc, multiproc, and Ray-based distributed executors.
- **`attention/`** -- Attention backend abstraction and backend-specific implementations (FlashAttention, FlashInfer, etc.).
- **`sample/`** -- Sampling kernels, logits processors, rejection sampler for speculative decoding.
- **`spec_decode/`** -- Speculative decoding: EAGLE, Medusa, n-gram proposers, draft models, suffix decoding.
- **`structured_output/`** -- Guided generation backends (xgrammar, outlines, guidance, lm-format-enforcer).
- **`kv_offload/`** -- KV cache offloading to CPU or other devices.
- **`pool/`** -- Pooling/embedding-specific logic (late interaction, metadata).
- **`metrics/`** -- Request-level and system-level metrics collection.

### `model_executor/`

Everything related to loading, building, and running models.

- **`models/`** -- Model implementations (261+ files). Each file wires a HuggingFace architecture into vLLM's layer abstractions. Registered via `ModelRegistry`.
- **`layers/`** -- Reusable layer building blocks:
  - `linear.py`, `layernorm.py`, `activation.py`, `conv.py` -- basic layers.
  - `attention/` -- attention layer implementations.
  - `rotary_embedding/` -- RoPE variants.
  - `fused_moe/` -- Mixture-of-Experts kernels (Triton, CUTLASS, DeepGEMM, etc.).
  - `quantization/` -- Quantization methods: AWQ, GPTQ, FP8, BitsAndBytes, GGUF, compressed tensors, and more.
  - `vocab_parallel_embedding.py` -- Tensor-parallel embedding layers.
  - `mla.py` -- Multi-head Latent Attention (DeepSeek).
  - `mamba/`, `fla/` -- State-space model and linear-attention layers.
  - `pooler/` -- Pooling heads for embedding models.
  - `logits_processor.py` -- Final logits computation layer.
- **`model_loader/`** -- Weight loading backends: default (safetensors/PyTorch), GGUF, sharded state, BitsAndBytes, TensorRT, RunAI streamer, and hot-reload support.
- **`custom_op.py`** -- Base class for registering custom Torch ops.
- **`offloader/`** -- Layer-level weight offloading.
- **`warmup/`** -- Model warmup utilities.

### `distributed/`

Multi-GPU and multi-node support.

- **`parallel_state.py`** -- Global state for tensor/pipeline/expert parallelism groups.
- **`communication_op.py`** -- Collective operations (all-reduce, all-gather, broadcast).
- **`device_communicators/`** -- Backend-specific communicators: NCCL (PyNccl), custom all-reduce, FlashInfer all-reduce, symmetric memory, shared memory broadcast, Ray, XPU.
- **`kv_transfer/`** -- Disaggregated prefill / KV cache transfer between nodes.
- **`elastic_ep/`** -- Elastic expert parallelism.
- **`eplb/`** -- Expert-parallel load balancing.
- **`weight_transfer/`** -- Live weight updates (for online LoRA swaps, etc.).
- **`ec_transfer/`** -- Elastic compute transfer.

### `multimodal/`

Multimodal input processing for vision-language, audio-language, and video-language models.

- **`image.py`**, **`video.py`**, **`audio.py`** -- Media type plugins.
- **`processing/`** -- Per-model multimodal processors.
- **`registry.py`** -- Multimodal plugin registry.
- **`inputs.py`** -- Multimodal input data structures.
- **`cache.py`**, **`hasher.py`** -- Caching and deduplication of processed media.
- **`media/`** -- Media loading and fetching utilities.

### `lora/`

LoRA (Low-Rank Adaptation) support.

- **`lora_model.py`** -- LoRA model wrapper.
- **`model_manager.py`**, **`worker_manager.py`** -- LoRA adapter lifecycle management.
- **`layers/`** -- LoRA-aware linear, embedding, and attention layers.
- **`punica_wrapper/`** -- Punica-based batched LoRA kernels.
- **`request.py`** -- `LoRARequest` data structure.

### `platforms/`

Hardware platform abstraction layer.

- **`interface.py`** -- Base `Platform` interface.
- **`cuda.py`**, **`rocm.py`**, **`cpu.py`**, **`tpu.py`**, **`xpu.py`**, **`zen_cpu.py`** -- Platform-specific implementations for NVIDIA CUDA, AMD ROCm, CPU, Google TPU, Intel XPU, and AMD Zen CPU.

### `compilation/`

Graph compilation and CUDA graph capture.

- **`cuda_graph.py`** -- CUDA graph capture and replay.
- **`backends.py`**, **`piecewise_backend.py`** -- Torch compilation backends.
- **`passes/`** -- Compiler optimization passes.
- **`decorators.py`** -- Decorators for marking functions as compilable.
- **`monitor.py`** -- Compilation monitoring and diagnostics.

### `tokenizers/`

Tokenizer implementations and registry.

- **`hf.py`** -- HuggingFace tokenizer wrapper.
- **`mistral.py`** -- Mistral tokenizer support.
- **`registry.py`** -- Tokenizer registry and lookup.
- Specialized tokenizers for DeepSeek V3.2, Grok2, Kimi Audio, Qwen VL.

### `transformers_utils/`

Utilities for interfacing with HuggingFace Transformers.

- **`config.py`**, **`config_parser_base.py`** -- Model config loading and parsing.
- **`tokenizer.py`** -- Tokenizer loading with fallbacks.
- **`processor.py`**, **`processors/`** -- HuggingFace processor wrappers per model.
- **`configs/`** -- Custom config classes for models not yet in upstream Transformers.
- **`chat_templates/`** -- Chat template overrides.
- **`repo_utils.py`**, **`s3_utils.py`** -- Model repository and S3 access helpers.

### `reasoning/`

Chain-of-thought / reasoning token parsers for models that emit structured thinking (DeepSeek R1/V3, Qwen3, Granite, Mistral, etc.).

### `tool_parsers/`

Function-calling / tool-use output parsers. One parser per model family (Hermes, Llama, Mistral, Granite, etc.) plus a generic OpenAI-format parser.

### `renderers/`

Output rendering for models that produce non-text outputs (embeddings, multimodal, specialized formats). Includes HuggingFace, Mistral, Grok2, and DeepSeek V3.2 renderers.

## Smaller Modules

| Directory / File | Purpose |
|---|---|
| `kernels/` | Custom compute kernels, including Helion kernels. |
| `benchmarks/` | Built-in benchmarking tools for latency, throughput, and multimodal processor performance. |
| `ray/` | Ray integration utilities and lazy-import helpers. |
| `plugins/` | Plugin system based on Python entry points (general, IO processor, platform, stat logger plugin groups). |
| `profiler/` | Layer-wise profiling and profiler wrappers. |
| `tracing/` | OpenTelemetry tracing integration. |
| `usage/` | Anonymous usage statistics collection. |
| `utils/` | General-purpose utilities: async helpers, memory management, hashing, network, NCCL, FlashInfer, import guards, and more. |
| `assets/` | Asset loading (images, audio, video) for tests and examples. |
| `device_allocator/` | CUDA memory allocator (`cumem.py`) for fine-grained GPU memory control. |
| `parser/` | Output parsers for specific model formats (e.g., MiniMax M2). |
| `logging_utils/` | Colored and newline-aware log formatters. |
| `triton_utils/` | Triton kernel utilities. |
| `vllm_flash_attn/` | Vendored FlashAttention interface and PyNVML bindings. |
| `third_party/` | Vendored third-party code (FlashMLA). |
| `py.typed` | PEP 561 marker for type-checking support. |

## Architecture Overview

A typical request flows through these layers:

```
Client (HTTP / Python API)
  --> entrypoints/ (OpenAI server, LLM class, CLI)
    --> engine/ or v1/engine/ (request queuing, scheduling)
      --> v1/core/ (KV cache management, scheduler)
        --> v1/executor/ (process orchestration, distributed launch)
          --> v1/worker/ (model runner, input batching)
            --> model_executor/ (model forward pass, layers, kernels)
              --> distributed/ (tensor/pipeline parallelism, collectives)
  --> outputs back up the stack
```

The V1 engine (`v1/`) is the primary active path. The legacy `engine/` module
remains for backward compatibility but delegates to V1 internally.
