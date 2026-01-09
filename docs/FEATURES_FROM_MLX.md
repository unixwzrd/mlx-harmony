# Features to Incorporate from MLX-LM

This document identifies valuable features from `mlx-lm` and `mlx-examples` that we should consider incorporating into `mlx-harmony`.

## 🔥 High-Value Features

### 1. Model Management CLI (`mlx_lm.manage`)

**What**: Command-line tool to scan, list, and delete models from Hugging Face cache.

**Implementation**:

- Uses `huggingface_hub.scan_cache_dir()` to scan cache
- Tabulated output showing: repo ID, size, path, last accessed
- Pattern-based filtering
- Confirmation prompts for deletion

**Why we need it**: Users need to discover available models and manage disk space.

**Files to reference**: `mlx-lm/mlx_lm/manage.py`

**Our implementation**: Add `mlx-harmony-manage` or `mlx-harmony-list` command

---

### 2. Prompt Caching System (`LRUPromptCache`)

**What**: Intelligent KV cache management for multi-turn conversations.

**Features**:

- LRU cache with configurable max size
- Cache search (exact, shorter, longer matches)
- Automatic cache trimming
- Cache persistence (save/load to disk)

**Why we need it**: Dramatically speeds up multi-turn conversations by avoiding recomputation of prompt prefixes.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (lines 174-308), `mlx-lm/mlx_lm/cache_prompt.py`

**Our implementation**:

- Add `LRUPromptCache` class to `mlx_harmony.cache`
- Integrate into chat loop
- Add `--prompt-cache-file` option to CLI

---

### 3. `/v1/models` and `/health` Endpoints

**What**: Additional HTTP endpoints for model listing and health checks.

**Implementation**:

- `/v1/models`: Scans HF cache and returns available models
- `/health`: Simple health check returning `{"status": "ok"}`

**Why we need it**: Standard API endpoints for clients to discover available models and check server status.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (lines 1440-1519)

**Our implementation**: Add these endpoints to our FastAPI server

---

### 4. Model Provider Pattern

**What**: Centralized model loading and caching with lazy loading.

**Features**:

- On-demand model loading
- Model caching with key-based lookup
- Support for adapters (LoRA)
- Draft model support
- Automatic cleanup

**Why we need it**: Efficient resource management in server mode, support for dynamic model switching.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (lines 389-504)

**Our implementation**: Refactor server to use `ModelProvider` pattern

---

### 5. Speculative Decoding (Draft Model)

**What**: Use a smaller draft model to propose tokens, then verify with main model.

**Features**:

- Configurable draft model path
- `num_draft_tokens` parameter
- Automatic acceptance/rejection
- Fallback to main model

**Why we need it**: Significant speedup for generation (2-3x faster in some cases).

**Files to reference**: `mlx-lm/mlx_lm/server.py` (uses `draft_model` in `stream_generate`)

**Our implementation**: Add draft model support to `TokenGenerator.generate()`

---

### 6. Logprobs Support

**What**: Return log probabilities for generated tokens.

**Features**:

- Top-k logprobs per token
- Configurable number of top tokens
- Per-token probability scores

**Why we need it**: Useful for debugging, confidence scoring, and advanced use cases.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (lines 808-815)

**Our implementation**: Add `logprobs` parameter to `TokenGenerator.generate()`

---

## 🎯 Medium-Value Features

### 7. Batch Generation

**What**: Process multiple prompts concurrently in a batch.

**Features**:

- Efficient batching with padding
- Shared KV cache
- Per-request progress tracking
- Request queuing

**Why we need it**: Better throughput for server use cases with multiple concurrent requests.

**Files to reference**: `mlx-lm/mlx_lm/generate.py` (BatchGenerator class, lines 924-1080)

**Our implementation**: Add batch processing to server mode

---

### 8. KV Cache Quantization

**What**: Quantize KV cache to reduce memory usage.

**Features**:

- Configurable bit-width (e.g., 8-bit)
- Group size configuration
- Delayed quantization start

**Why we need it**: Reduce memory usage for long contexts or large models.

**Files to reference**: `mlx-lm/mlx_lm/cache_prompt.py` (kv_bits, kv_group_size options)

**Our implementation**: Add quantization options to `TokenGenerator` and cache system

---

### 9. Text Completions Endpoint (`/v1/completions`)

**What**: Non-chat completion endpoint for simpler use cases.

**Why we need it**: Some clients prefer the simpler completions API over chat completions.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (handle_text_completions method)

**Our implementation**: Add `/v1/completions` endpoint to our server

---

### 10. Role Mapping Customization

**What**: Allow clients to customize role prefixes in prompts.

**Why we need it**: Flexibility for different chat templates and use cases.

**Files to reference**: `mlx-lm/mlx_lm/server.py` (role_mapping in request handling)

**Our implementation**: Add role mapping to our Harmony message conversion

---

## 📚 Nice-to-Have Features

### 11. Evaluation Framework

**What**: Built-in evaluation utilities for benchmarking models.

**Why we need it**: Useful for testing and comparing models.

**Files to reference**: `mlx-lm/mlx_lm/evaluate.py`

---

### 12. Tool Use Examples

**What**: Examples of OpenAI-style tool/function calling.

**Why we need it**: Reference implementation for our GPT-OSS tools.

**Files to reference**: `mlx-lm/mlx_lm/examples/openai_tool_use.py`

---

### 13. Distributed Inference

**What**: Multi-GPU/device inference support.

**Why we need it**: Scale to larger models or higher throughput.

**Files to reference**: `mlx-lm` distributed features (uses `mx.distributed`)

---

## 🎨 Implementation Priority

1. **Prompt Caching** - Highest impact for user experience
2. **Model Management CLI** - Essential for usability
3. **Server Endpoints** (`/v1/models`, `/health`) - Standard API expectations
4. **Model Provider Pattern** - Foundation for efficient server mode
5. **Speculative Decoding** - Significant performance win
6. **Logprobs Support** - Advanced feature for power users
7. **Batch Generation** - Performance optimization
8. **KV Cache Quantization** - Memory optimization
9. **Text Completions Endpoint** - API completeness
10. **Role Mapping** - Flexibility

---

## 📝 Notes

- Most features can be adapted from `mlx-lm` with Harmony-specific modifications
- Prompt caching is particularly valuable for GPT-OSS models with Harmony format
- Model provider pattern will help with our profile system
- Some features (like distributed inference) are lower priority for our current scope

---

**Last Updated**: 2026-01-06

[← Back to README](../README.md)
