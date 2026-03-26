# vLLM Examples

This directory contains example scripts, configurations, and templates that demonstrate how to use vLLM for inference, serving, and related workflows. The examples range from minimal getting-started scripts to advanced multi-node disaggregated serving setups.

## Directory Structure

```
examples/
  basic/                    # Minimal starter examples
  offline_inference/        # Batch/offline inference with the LLM Python API
  online_serving/           # Running and querying the vLLM API server
  pooling/                  # Embedding, reranking, classification, and other pooling tasks
  rl/                       # Reinforcement learning integration (RLHF)
  others/                   # Miscellaneous utilities (LMCache, tensorization, logging)
  *.jinja                   # Chat and tool-call prompt templates
```

## Quick Start

If you are new to vLLM, start with the basic examples.

**Offline inference** (no server required):

```bash
python examples/basic/offline_inference/basic.py
```

**Online serving** (OpenAI-compatible API):

```bash
# Terminal 1 -- start the server
vllm serve meta-llama/Llama-2-7b-chat-hf

# Terminal 2 -- send a request
python examples/basic/online_serving/openai_chat_completion_client.py
```

## Subdirectories

### basic/

Minimal examples for first-time users.

- `offline_inference/` -- Use the `LLM` Python class to generate, chat, embed, classify, score, and compute rewards. Start with `basic.py` for the simplest possible usage; the other scripts accept command-line arguments for experimenting with models, quantization, and CPU offload. See the [subdirectory README](basic/offline_inference/README.md) for details.
- `online_serving/` -- OpenAI-compatible chat completion and text completion clients that connect to a running `vllm serve` instance.

### offline_inference/

Advanced offline (batch) inference examples using the `LLM` class and engine APIs.

| Area | Scripts |
| ---- | ------- |
| Multimodal | `vision_language.py`, `vision_language_multi_image.py`, `audio_language.py`, `encoder_decoder_multimodal.py` |
| Structured output | `structured_outputs.py` |
| Speculative decoding | `spec_decode.py`, `mlpspeculator.py` |
| Data parallelism | `data_parallel.py`, `torchrun_example.py`, `torchrun_dp_example.py` |
| Prefix caching | `prefix_caching.py`, `prefix_caching_flexkv.py`, `automatic_prefix_caching.py` |
| LoRA | `multilora_inference.py`, `lora_with_quantization_inference.py` |
| Tool calling | `chat_with_tools.py` |
| Batch processing | `batch_llm_inference.py`, `openai_batch/` (OpenAI batch file format) |
| Streaming | `async_llm_streaming.py` |
| Context extension | `context_extension.py`, `qwen_1m.py` |
| Disaggregated prefill | `disaggregated_prefill.py`, `disaggregated-prefill-v1/` |
| KV cache | `kv_load_failure_recovery/`, `llm_engine_reset_kv.py` |
| Profiling and metrics | `simple_profiling.py`, `metrics.py`, `reproducibility.py` |
| Model loading | `load_sharded_state.py`, `save_sharded_state.py`, `skip_loading_weights_in_engine_init.py` |
| Logits processors | `logits_processor/` (custom engine-level and request-level processors) |
| Prompt embeddings | `prompt_embed_inference.py` |
| Engine API | `llm_engine_example.py`, `run_one_batch.py`, `pause_resume.py`, `extract_hidden_states.py` |
| Model-specific | `mistral-small.py`, `qwen2_5_omni/`, `qwen3_omni/`, `routed_experts_e2e.py` |

### online_serving/

Examples for running the vLLM OpenAI-compatible API server and building applications on top of it.

**API clients:**

- `api_client.py` -- Generic API client.
- `openai_chat_completion_client_for_multimodal.py` -- Multimodal chat completions.
- `openai_chat_completion_client_with_tools*.py` -- Tool/function calling (multiple variants).
- `openai_chat_completion_with_reasoning*.py` -- Reasoning models (DeepSeek R1, etc.) with optional streaming.
- `openai_chat_completion_tool_calls_with_reasoning.py` -- Combining tools and reasoning.
- `openai_responses_client*.py` -- OpenAI Responses API, including MCP tools.
- `openai_realtime_client.py`, `openai_realtime_microphone_client.py` -- Realtime WebSocket API for audio.
- `openai_transcription_client.py`, `openai_translation_client.py` -- Whisper-based speech endpoints.
- `prompt_embed_inference_with_openai_client.py` -- Prompt embeddings via the API.
- `token_generation_client.py` -- Token-level generation.
- `batched_chat_completions.py` -- Batched requests.

**Web UIs:**

- `gradio_openai_chatbot_webserver.py` -- Gradio chatbot frontend.
- `gradio_webserver.py` -- Gradio interface (direct).
- `streamlit_openai_chatbot_webserver.py` -- Streamlit chatbot frontend.

**RAG:**

- `retrieval_augmented_generation_with_langchain.py` -- RAG with LangChain and Milvus.
- `retrieval_augmented_generation_with_llamaindex.py` -- RAG with LlamaIndex.

**Deployment and scaling:**

- `ray_serve_deepseek.py` -- Deploy with Ray Serve.
- `multi_instance_data_parallel.py` -- Multi-instance data parallelism.
- `data_parallel_pause_resume.py` -- Pause/resume with data parallelism.
- `multi-node-serving.sh` -- Multi-node serving.
- `run_cluster.sh` -- Cluster launch script.
- `sagemaker-entrypoint.sh` -- AWS SageMaker entrypoint.
- `elastic_ep/` -- Elastic expert parallelism with dynamic scaling.

**Disaggregated serving:**

- `disaggregated_prefill.sh` -- Disaggregated prefill/decode with proxy.
- `disaggregated_serving/` -- XpYd proxy demos, KV event publishing, MooncakeConnector.
- `disaggregated_serving_p2p_nccl_xpyd/` -- P2P NCCL-based disaggregated serving.
- `disaggregated_encoder/` -- Encoder-prefill-decode disaggregation (EPD) for vision-language models.
- `ec_both_encoder/` -- Encoder cache with both encoder modes.

**Structured outputs:**

- `structured_outputs/` -- Standalone structured output demo with streaming and concurrent requests.

**Monitoring and observability:**

- `dashboards/` -- Grafana and Perses monitoring dashboards.
- `prometheus_grafana/` -- Docker Compose setup for Prometheus and Grafana.
- `opentelemetry/` -- OpenTelemetry tracing with Jaeger.
- `kv_events_subscriber.py` -- Subscribe to KV cache events.

### pooling/

Examples for non-generative model tasks: embedding, reranking, classification, and scoring.

- `embed/` -- Embedding requests (online and offline), including vision embeddings, Matryoshka embeddings, Jina Embeddings v3, long text chunked processing, and OpenAI-compatible clients. Contains Jinja templates for VLM embedding models in `template/`.
- `score/` -- Reranking and scoring with models like BGE, Cohere, ColBERT, ColQwen, and Qwen3 Reranker. Includes vision reranking and Jinja templates for reranker prompt formatting in `template/`.
- `classify/` -- Sequence classification (text and vision) via the classification API.
- `token_classify/` -- Token-level classification (NER) offline and online.
- `token_embed/` -- Token-level (multi-vector) embeddings for ColBERT-style retrieval.
- `pooling/` -- Generic pooling API (e.g., reward models).
- `plugin/` -- Custom model plugins (Prithvi geospatial MAE example with custom I/O processor).

### rl/

Reinforcement learning from human feedback (RLHF) integration examples using vLLM as the inference engine with Ray for distributed training.

- `rlhf_nccl.py` -- Weight syncing between a training model and a vLLM inference engine over NCCL.
- `rlhf_ipc.py` -- Weight syncing via shared memory IPC.
- `rlhf_http_nccl.py` -- HTTP-based coordination with NCCL weight transfer.
- `rlhf_http_ipc.py` -- HTTP-based coordination with IPC weight transfer.
- `rlhf_nccl_fsdp_ep.py` -- NCCL with FSDP and expert parallelism.
- `rlhf_async_new_apis.py` -- Async RLHF with batch-invariant generation and native weight syncing APIs.

### others/

- `lmcache/` -- Integration with [LMCache](https://github.com/LMCache/LMCache) for disaggregated prefill, CPU offload, and KV cache sharing. See the [subdirectory README](others/lmcache/README.md).
- `tensorize_vllm_model.py` -- Serialize and deserialize vLLM models using CoreWeave's Tensorizer format.
- `logging_configuration.md` -- Guide to configuring vLLM logging via environment variables.

## Jinja Templates

The `.jinja` files in the root of this directory are [Jinja2](https://jinja.palletsprojects.com/) chat templates used to format conversations for various model families. They are passed to vLLM via the `--chat-template` flag.

**Basic chat templates** (`template_*.jinja`):

Format conversations into model-specific prompt structures (ChatML, Alpaca, Baichuan, ChatGLM, Falcon, InkBot, TeleFLM).

```bash
vllm serve <model> --chat-template examples/template_chatml.jinja
```

**Tool-calling chat templates** (`tool_chat_template_*.jinja`):

Extend chat templates with function/tool-calling support for models that support it. These templates define how tool definitions, tool calls, and tool results are serialized in the prompt. Available for: DeepSeek (R1, V3, V3.1), Gemma 3, GLM-4, Granite, Hermes, Hunyuan, InternLM2, Llama 3.x/4, MiniMax M1, Mistral, Phi-4 Mini, Qwen3 Coder, ToolACE, and xLAM.

```bash
vllm serve <model> \
    --chat-template examples/tool_chat_template_llama4_json.jinja \
    --enable-auto-tool-choice \
    --tool-call-parser llama4
```

## Prerequisites

Most examples require a working vLLM installation. See the project [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) or the development setup in `AGENTS.md`. Some examples have additional dependencies noted in their docstrings or README files (e.g., `gradio`, `cohere`, `langchain`, `ray`).

## Further Reading

- [vLLM Documentation](https://docs.vllm.ai/)
- [Offline Inference API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
- [Sampling Parameters](https://docs.vllm.ai/en/latest/api/inference_params.html)
- [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)
