# vLLM Test Suite

This directory contains the full test suite for vLLM. Tests are organized by
functional area and rely on [pytest](https://docs.pytest.org/) as the test
runner.

## Directory Structure

### Top-Level Test Files

Standalone test modules that live directly under `tests/`:

| File | Purpose |
|------|---------|
| `conftest.py` | Root-level pytest fixtures (LLM helpers, image/video/audio assets, model comparison utilities) |
| `ci_envs.py` | CI-specific environment variables (`VLLM_CI_NO_SKIP`, `VLLM_CI_DTYPE`, etc.) |
| `utils.py` | Shared test utilities (server management, remote OpenAI/Anthropic client helpers, quantization helpers) |
| `test_config.py` | Configuration parsing and validation |
| `test_regression.py` | Regression tests for previously reported bugs |
| `test_sequence.py` | Sequence and sequence-group logic |
| `test_logprobs.py` | Log-probability computation |
| `test_logger.py` | Logging infrastructure |
| `test_inputs.py` | Input processing |
| `test_envs.py` | Environment variable handling |
| `test_seed_behavior.py` | Random seed determinism |

### Core Subsystems

#### `kernels/`

GPU kernel tests, organized into:

- `attention/` -- Attention backends: FlashAttention, FlashInfer, Triton decode/prefill, MLA decode, cascade attention, and more.
- `core/` -- Fundamental kernels: activations, rotary embeddings, layer norms, positional encodings.
- `moe/` -- Mixture-of-Experts kernels: fused MoE, DeepGeMM, CUTLASS, Triton, routing, expert-parallel variants.
- `quantization/` -- Quantized GEMM kernels: FP8, INT8, AWQ, GPTQ, GGUF, Marlin, NVFP4, MXFP4, CUTLASS scaled-MM.
- `mamba/` -- Mamba (state-space model) kernels.
- `helion/` -- Helion kernel tests.

#### `models/`

Model correctness tests comparing vLLM output against HuggingFace reference:

- `language/generation/` -- Text generation models (Gemma, Granite, Grok, Mistral, hybrid architectures, etc.).
- `language/pooling/` -- Embedding and pooling language models.
- `multimodal/generation/` -- Vision-language and audio-language models (Qwen2-VL, Pixtral, Ultravox, Whisper, etc.).
- `multimodal/pooling/` -- Multimodal embedding models.
- `multimodal/processing/` -- Multimodal input processing.
- `quantization/` -- Quantized model correctness (AWQ, BitsAndBytes, FP8, GGUF, GPTQ-Marlin, ModelOpt, NVFP4, MXFP4/8).
- `fixtures/` -- Shared test data for model tests.

#### `entrypoints/`

API and server entrypoint tests:

- `openai/` -- OpenAI-compatible API: chat completions, text completions, tool parsing, models endpoint, responses API, speech-to-text, realtime.
- `anthropic/` -- Anthropic Messages API compatibility.
- `llm/` -- Offline `LLM` class: generation, chat, accuracy, structured output, GPU utilization.
- `pooling/` -- Pooling/embedding server endpoints (embed, classify, reward, scoring, token-classify).
- `serve/` -- `vllm serve` CLI tests.
- `sagemaker/` -- SageMaker endpoint compatibility.
- `rpc/` -- RPC-based engine communication.
- `offline_mode/` -- Offline (no-network) operation.
- `weight_transfer/` -- Weight transfer between models.

#### `distributed/`

Multi-GPU and multi-node tests:

- Tensor-parallel and pipeline-parallel correctness.
- Communication operations (NCCL, custom all-reduce, symmetric memory).
- Expert-parallel load balancing (EPLB).
- Multi-process executor tests.

#### `v1/`

Tests specific to the V1 engine architecture:

- `core/` -- V1 scheduler, prefix caching, KV cache management, encoder cache.
- `engine/` -- V1 engine core, async LLM, output processing.
- `worker/` -- GPU model runner, input batch, profiler.
- `attention/` -- V1 attention backend selection and splitting.
- `cudagraph/` -- CUDA graph dispatch and capture modes.
- `distributed/` -- Data-parallel, disaggregated batched decoding (DBO).
- `e2e/` -- End-to-end tests: cascade attention, sliding window, speculative decoding, streaming input, pooling.
- `spec_decode/` -- Speculative decoding (EAGLE, MTP, n-gram, tree attention).
- `kv_connector/` -- KV cache transfer connectors (NIXL integration, unit tests).
- `kv_offload/` -- CPU/GPU KV cache offloading.
- `structured_output/` -- Guided/structured generation.
- `logits_processors/` -- Custom logits processors.
- `sample/` -- Sampling and log-probability tests.
- `metrics/` -- Prometheus metrics and stats collection.
- `streaming_input/` -- Streaming input processing.
- `shutdown/` -- Graceful shutdown and error handling.
- `tracing/` -- OpenTelemetry tracing.
- `executor/` -- V1 executor tests.
- `entrypoints/openai/` -- V1-specific OpenAI endpoint tests.

### Functional Areas

#### `quantization/`

Quantization integration tests: FP8, compressed tensors, CPU offload with
quantization, auto-round, GPTQ dynamic/v2, LM head quantization, MoE
quantization on various hardware.

#### `lora/`

LoRA adapter tests: adding/removing adapters, multi-LoRA serving, LoRA with
tensor parallelism (Llama, ChatGLM, DeepSeek), kernel-level fused MoE + LoRA,
checkpoint formats, HuggingFace LoRA loading.

#### `compile/`

Torch compilation tests:

- `fullgraph/` -- Full-graph compilation validation.
- `fusions_e2e/` -- End-to-end fusion correctness.
- `correctness_e2e/` -- Compiled vs. eager output comparison.
- `passes/` -- Compiler pass tests.
- AOT compilation, dynamic shapes, compile ranges, structured logging.

#### `samplers/`

Sampling strategy tests: beam search, logprobs, `ignore_eos`, bad-words
filtering.

#### `multimodal/`

Multimodal infrastructure tests: image/audio/video processing, embedding
validation, caching, hashing, registry, sparse tensors.

#### `reasoning/`

Reasoning/thinking token parser tests for various model families (DeepSeek-R1,
Granite, Hunyuan, Mistral, QwQ, etc.).

#### `tool_parsers/`

Tool-call parsing tests for model-specific formats (Hermes, Llama, Mistral,
Jamba, Granite, DeepSeek, etc.).

#### `tool_use/`

End-to-end tool-use tests: chat completions with tool calls, parallel tool
calls, tool-choice enforcement.

#### `tokenizers_/`

Tokenizer tests: basic tokenization, detokenization, HuggingFace and Mistral
tokenizer backends, tokenizer registry.

#### `detokenizer/`

Detokenization behavior: stop strings, stop reasons, minimum token
enforcement, disabling detokenization.

#### `renderers/`

Chat template rendering tests for completion and HuggingFace/Mistral formats.

#### `transformers_utils/`

HuggingFace Transformers utility tests: config parsing, processor handling,
repo utilities.

#### `engine/`

Engine argument parsing and short multimodal context tests.

#### `model_executor/`

Model executor internals: weight utilities, custom op enablement, layer-level
tests, model loader variants (RunAI Streamer, Tensorizer).

#### `weight_loading/`

Weight loading smoke tests driven by model lists (`models.txt`,
`models-large.txt`, AMD variants).

#### `tracing/`

OpenTelemetry tracing integration during model loading.

#### `cuda/`

CUDA-specific tests: compatibility paths, context management, platform
detection without CUDA init.

#### `rocm/`

ROCm/AMD-specific tests (AITER kernels, AMD-specific attention selectors).

### Support Directories

| Directory | Purpose |
|-----------|---------|
| `plugins/` | Dummy plugin packages used by plugin tests (dummy models, platforms, stat loggers, IO processors, LoRA resolvers). |
| `plugins_tests/` | Tests that exercise the plugin system using the packages in `plugins/`. |
| `prompts/` | Sample prompt text files used by fixtures. |
| `system_messages/` | System message templates for tests. |
| `tools/` | Configuration validator tests. |
| `utils_/` | Tests for `vllm.utils` submodules (async, caching, hashing, networking, memory, serialization, etc.). |
| `vllm_test_utils/` | Installable helper package (`blame`, `monitor`) that does not import vLLM internals. |
| `standalone_tests/` | Scripts meant to run outside the normal pytest harness (lazy import checks, compile-only checks, nightly dependency checks). |
| `evals/` | Evaluation benchmarks (GSM8K, GPQA via GPT-OSS). |
| `benchmarks/` | Benchmark CLI tests (latency, throughput, startup, serve). |
| `basic_correctness/` | Smoke tests for basic inference correctness, CPU offload, and cumem. |
| `config/` | Configuration generation and model-architecture config tests. |

## Running Tests

### Prerequisites

Set up the development environment as described in `AGENTS.md`:

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test.in
```

### Running a Specific Test File

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running a Specific Test Function

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py::test_function_name -v
```

### Running a Directory of Tests

```bash
.venv/bin/python -m pytest tests/kernels/core/ -v
```

### Common Options

```bash
# Show full output and don't capture stdout
.venv/bin/python -m pytest tests/test_config.py -v -s

# Run tests matching a keyword expression
.venv/bin/python -m pytest tests/models/ -k "test_registry" -v

# Run with multiple workers (requires pytest-xdist)
.venv/bin/python -m pytest tests/utils_/ -v -n 4
```

### Running Linters

```bash
pre-commit run --all-files
pre-commit run ruff-check --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

## CI Environment Variables

The `ci_envs.py` module exposes variables that control test behavior in CI:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_CI_NO_SKIP` | `0` | When set to `1`, run all models in a family instead of just one representative. |
| `VLLM_CI_DTYPE` | `None` | Override the dtype used by vLLM during tests. |
| `VLLM_CI_HEAD_DTYPE` | `None` | Override the head dtype used by vLLM during tests. |
| `VLLM_CI_HF_DTYPE` | `None` | Override the dtype used by HuggingFace reference models. |
| `VLLM_CI_ENFORCE_EAGER` | `None` | Control whether tests use `enforce_eager` mode. |

## Key Fixtures

The root `conftest.py` provides fixtures used across the test suite:

- **`sample_json_schema`** -- A sample JSON schema for structured output tests.
- **`HfRunner`** -- Wrapper for running HuggingFace models as a reference baseline.
- **`VllmRunner`** -- Wrapper for running vLLM models for comparison against HfRunner.
- **Image/video/audio asset helpers** -- `ImageTestAssets`, `VideoTestAssets`, `AudioTestAssets` for multimodal tests.

## Writing Tests

- Place new tests in the subdirectory matching the vLLM module under test.
- Use existing fixtures from `conftest.py` and `utils.py` where possible.
- For model correctness tests, follow the pattern in `tests/models/` of comparing vLLM output against a HuggingFace reference.
- GPU-dependent tests will be skipped automatically when no GPU is available.
- Use `ci_envs` variables to control behavior that should differ between local and CI runs.
