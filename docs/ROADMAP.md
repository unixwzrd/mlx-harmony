# MLX-Harmony Roadmap

**Created**: 2026-01-07
**Updated**: 2026-01-28

**Project Status**: ✅ v0.2.0 - Standalone Implementation with Markdown Rendering

## Overview

This roadmap tracks planned enhancements, improvements, and features for `mlx-harmony`. Items are organized by priority and category.

---

## 🎯 High Priority

### Model Conversion and Import

- [ ] **External Model Conversion to MLX**
  - [ ] Convert Hugging Face safetensors models to MLX and store locally or publish to Hugging Face Hub
  - [ ] Convert GGUF models to MLX and store locally or publish to Hugging Face Hub
  - [ ] Validate converted models against reference outputs (token parity and short prompts)
  - [ ] Document supported architectures and limitations

- [ ] **On-the-fly Conversion**
  - [ ] Load safetensors models and convert to MLX in-memory for immediate use
  - [ ] Load GGUF models and convert to MLX in-memory for immediate use
  - [ ] Cache converted weights to avoid repeated conversions
  - [ ] Expose CLI options to control conversion output and cache locations

### Model Management CLI

Inspired by `mlx-lm`'s `mlx_lm.manage` tool.

- [ ] **Model Scanner** (`mlx-harmony-list` or `mlx-harmony-manage`)
  - [ ] Scan Hugging Face cache for MLX models
  - [ ] List available models with metadata (size, path, last accessed)
  - [ ] Filter/search models by pattern
  - [ ] Integration with profiles.json for easy selection
  - [ ] Display model compatibility (GPT-OSS vs standard)

- [ ] **Model Cleanup**
  - [ ] Delete models from cache
  - [ ] Confirm before deletion
  - [ ] Size reporting and disk space management

**Reference**: `mlx-lm/mlx_lm/manage.py` - Uses `huggingface_hub.scan_cache_dir()`

---

### Prompt Caching System

Inspired by `mlx-lm`'s prompt caching feature for multi-turn conversations.

- [ ] **KV Cache Management**
  - [ ] `LRUPromptCache` implementation for chat loop
  - [ ] Cache prompt prefixes to avoid recomputation
  - [ ] Configurable cache size (max entries)
  - [ ] Cache persistence (save/load from disk)
  - [ ] Cache trimming for long conversations

- [ ] **CLI for Prompt Caching**
  - [ ] `mlx-harmony-cache-prompt` command
  - [ ] Save prompt cache to file for reuse
  - [ ] Load cached prompts in chat/generate
  - [ ] KV cache quantization options

- [ ] **Server Integration**
  - [ ] Automatic prompt cache reuse across requests
  - [ ] Cache sharing for same conversation context
  - [ ] Memory-efficient cache management

**Reference**: `mlx-lm/mlx_lm/server.py` (LRUPromptCache class), `mlx-lm/mlx_lm/cache_prompt.py`

**Benefits**: Faster multi-turn conversations, reduced computation for repeated prompts

---

### Tool Executors Implementation

The tool infrastructure is in place, but executors need real implementations.

- [ ] **Browser Tool** (`src/mlx_harmony/tools/browser.py`)
  - [ ] Implement web navigation with `aiohttp` or `playwright`
  - [ ] Support: navigate, click, type, screenshot, extract content
  - [ ] Error handling and timeouts
  - [ ] Security: URL validation, sandboxed execution
  - [ ] Example usage in docs

- [ ] **Python Tool** (`src/mlx_harmony/tools/python.py`)
  - [ ] Docker-based sandboxed Python execution
  - [ ] Support: execute code, install packages, file I/O (limited)
  - [ ] Resource limits: CPU, memory, execution time
  - [ ] Security: file system restrictions, network isolation
  - [ ] Error handling and stdout/stderr capture

- [ ] **Apply Patch Tool** (`src/mlx_harmony/tools/patch.py`)
  - [ ] Git-based patch application
  - [ ] Support: unified diff format, hunk parsing
  - [ ] Validation: syntax checking, conflict detection
  - [ ] Dry-run mode for preview
  - [ ] Rollback capability

**Dependencies**: `aiohttp>=3.12.0`, `docker>=7.1.0`, `gitpython` (new)

---

### Testing Infrastructure

- [x] **Unit Tests**
  - [x] `tests/test_config.py`: Config loading, placeholder expansion
  - [x] `tests/test_generator.py`: TokenGenerator, format detection
  - [x] `tests/test_tools.py`: Tool parsing, execution stubs
  - [x] `tests/test_chat.py`: Chat save/load, conversation management
  - [x] `tests/test_server.py`: API endpoints, streaming, error handling

- [x] **Integration Tests**
  - [x] Conversation save/load with metadata
  - [x] Profile loading and resolution
  - [x] Server request/response cycles (FastAPI TestClient)
  - [x] Server streaming responses
  - [x] Server error handling
  - [ ] End-to-end chat with tool calls
  - [ ] Error handling edge cases

- [x] **CI/CD Setup**
  - [x] GitHub Actions workflow (`.github/workflows/ci.yml`)
  - [x] Test matrix (Python 3.12, 3.13)
  - [x] Linting (ruff, black)
  - [x] Coverage reporting (Codecov integration)
  - [ ] Type checking (mypy or pyright) - optional

**Current Status**: Core unit tests implemented. Uses small test model (`mlx-community/Qwen1.5-0.5B-Chat-4bit`) for inference tests.  
**Target Coverage**: >80%

---

### Documentation Improvements

- [ ] **API Documentation**
  - [ ] Sphinx or MkDocs setup
  - [ ] Full API reference with examples
  - [ ] Type hints documentation
  - [ ] Architecture diagrams

- [ ] **Usage Guides**
  - [ ] Tool usage tutorial
  - [ ] Prompt config best practices
  - [ ] Profile management guide
  - [ ] Troubleshooting section

- [ ] **Examples Directory**
  - [ ] `examples/chat_with_tools.py`
  - [ ] `examples/custom_tool.py`
  - [ ] `examples/server_client.py`
  - [ ] `examples/profile_setup.py`

---

## 🔧 Medium Priority

### Performance Optimizations

- [ ] **Model Caching** (from mlx-lm)
  - [ ] `ModelProvider` class with lazy loading
  - [ ] Cache loaded models across requests (server)
  - [ ] LRU cache for multiple models
  - [ ] Memory management and cleanup
  - [ ] Configurable cache size limits
  - [ ] Model unloading when not in use

- [ ] **Prompt Token Count Caching**
  - [ ] Cache per-message token counts in memory
  - [ ] Key caches by message ID/UUID and tokenizer hash
  - [ ] Invalidate cache on message edits or model/tokenizer change
  - [ ] Use cached counts to speed up context trimming

- [ ] **Speculative Decoding** (from mlx-lm)
  - [ ] Draft model support for faster generation
  - [ ] Configurable `num_draft_tokens`
  - [ ] Automatic draft model selection
  - [ ] Fallback to main model on rejection

- [ ] **KV Cache Optimization**
  - [ ] Quantized KV cache support
  - [ ] Configurable KV cache size limits
  - [ ] Rotating KV cache for long contexts

- [ ] **Streaming Improvements**
  - [ ] Streaming tool call detection
  - [ ] Progressive message parsing
  - [ ] Reduced latency for first token

- [ ] **Quantization Support**
  - [ ] Auto-quantization options
  - [ ] Quantized model caching
  - [ ] Memory-efficient loading

---

### Batch Generation Support

Inspired by `mlx-lm`'s `BatchGenerator` for concurrent requests.

- [ ] **Batch Processing**
  - [ ] Process multiple prompts concurrently
  - [ ] Efficient batching with padding
  - [ ] Shared KV cache across batch
  - [ ] Progress tracking per request
  - [ ] Request queuing system

**Reference**: `mlx-lm/mlx_lm/generate.py` (BatchGenerator class)

**Benefits**: Better throughput for server use cases

---

### Enhanced Configuration

- [ ] **Profile Enhancements**
  - [ ] Profile inheritance/composition
  - [ ] Environment variable substitution in profiles
  - [ ] Profile validation on load
  - [ ] Profile listing/searching CLI command

- [ ] **Prompt Config Improvements**
  - [ ] Template inheritance
  - [ ] Conditional placeholders (if/else logic)
  - [ ] External file references (`@include`)
  - [ ] Schema validation (JSON Schema)

- [ ] **Sampling Presets**
  - [ ] Predefined sampling profiles (creative, balanced, deterministic)
  - [ ] `--preset` CLI flag
  - [ ] Custom preset definitions in config

---

### Server Enhancements

- [ ] **Additional Endpoints** (from mlx-lm)
  - [ ] `/v1/models` - List available models (scan HF cache)
  - [ ] `/health` - Health check endpoint
  - [ ] `/metrics` - Prometheus metrics (optional)
  - [ ] `/v1/completions` - Non-chat completion endpoint (text completions)

- [ ] **Model Provider Pattern** (from mlx-lm)
  - [ ] `ModelProvider` class for on-demand model loading
  - [ ] Model caching across requests
  - [ ] Support for adapter paths (LoRA)
  - [ ] Dynamic model switching via API

- [ ] **Advanced Generation Features**
  - [ ] Logprobs support (top token probabilities)
  - [ ] Streaming with progress callbacks
  - [ ] Role mapping customization
  - [ ] Draft model support (speculative decoding)

- [ ] **Advanced Features**
  - [ ] Request queuing for concurrent requests
  - [ ] Rate limiting
  - [ ] Authentication/API keys
  - [ ] CORS configuration
  - [ ] WebSocket support for streaming

- [ ] **Tool Support in API**
  - [ ] Tool call execution in server
  - [ ] Tool result streaming
  - [ ] Tool configuration via API

---

## 🎨 Nice to Have

### Developer Experience

- [ ] **CLI Improvements**
  - [ ] Interactive profile/model selection
  - [ ] `mlx-harmony-list` - List available models/profiles
  - [ ] `mlx-harmony-config` - Config validation/editing tool
  - [ ] Progress bars for model loading
  - [ ] Colored output and better formatting

- [ ] **Debugging Tools**
  - [ ] `--debug` mode with verbose logging
  - [ ] `--raw-prompt` flag to dump rendered prompt
  - [ ] `--trace-tokens` for token-level debugging
  - [ ] Harmony format visualization

- [ ] **Type Hints & Linting**
  - [ ] Full type coverage
  - [ ] mypy or pyright strict mode
  - [ ] Pre-commit hooks

---

### Advanced Features

- [ ] **Custom Tools**
  - [ ] Plugin system for custom tools
  - [ ] Tool registration API
  - [ ] Tool schema validation
  - [ ] Example custom tool template

- [ ] **Multi-Model Support**
  - [ ] Model routing/load balancing
  - [ ] Fallback models
  - [ ] Parallel inference across models

- [ ] **Distributed Inference**
  - [ ] Multi-GPU support
  - [ ] Model sharding
  - [ ] Pipeline parallelism

- [ ] **Harmony Extensions**
  - [ ] Channel configuration
  - [ ] Function tools (ToolNamespaceConfig)
  - [ ] Custom special tokens
  - [ ] Advanced reasoning effort controls

---

### Integration & Ecosystem

- [ ] **LangChain Integration**
  - [ ] LangChain LLM wrapper
  - [ ] Tool bindings for LangChain agents
  - [ ] Example notebooks

- [ ] **Gradio/Streamlit UI**
  - [ ] Web UI for chat interface
  - [ ] Tool configuration UI
  - [ ] Model/profiles management UI

- [ ] **VS Code Extension**
  - [ ] Syntax highlighting for prompt configs
  - [ ] IntelliSense for configs
  - [ ] Quick launch commands

---

## 🐛 Bug Fixes & Maintenance

### Known Issues

- [ ] _Track known bugs here as they're discovered_
- [ ] _Add GitHub issues as they're reported_

### Maintenance

- [ ] **Dependency Management**
  - [ ] Regular dependency updates
  - [ ] Security vulnerability scanning
  - [ ] Compatibility testing

- [ ] **Code Quality**
  - [ ] Code review process
  - [ ] Refactoring opportunities
  - [ ] Performance profiling

---

## 📊 Version History

### v0.2.0 (Current) ✅ - 2026-01-09

- [x] Standalone generation and sampling implementation
- [x] Beautiful markdown rendering with rich library
- [x] Fixed MLX API compatibility issues
- [x] Fixed sampling implementation to match mlx-lm exactly
- [x] Added `\help` command for out-of-band commands
- [x] XTC special tokens auto-detection
- [x] Removed prewarm_cache feature

### v0.1.0 ✅ - 2026-01-06

- [x] Core TokenGenerator
- [x] PromptConfig with placeholders
- [x] Profiles system
- [x] Tool infrastructure (stubs)
- [x] CLI tools (chat, generate, server)
- [x] Basic documentation

For detailed changelog, see [CHANGELOG.md](../CHANGELOG.md).

### v0.3.0 (Planned)

- [ ] Browser tool implementation
- [ ] Python tool implementation
- [ ] Apply patch tool
- [ ] Model caching (ModelProvider pattern)
- [ ] Prompt caching system
- [ ] Enhanced server features

See sections above for detailed roadmap items.

---

## 🔄 Review Process

**When to Update This Roadmap:**

- After completing major features
- When new enhancement ideas emerge
- After user feedback/feature requests
- During release planning

**Priority Guidelines:**

- **High**: Core functionality, user blockers, security
- **Medium**: Quality of life, performance, nice UX
- **Nice to Have**: Polish, advanced features, ecosystem

---

## 📝 Notes

- Tool executors should be implemented with security as the top priority
- Testing infrastructure is critical before adding more features
- Documentation should keep pace with feature development
- Consider user feedback when prioritizing items

---

## 🤝 Contributing

Contributions welcome! Please:

1. Check this roadmap for planned work
2. Open an issue to discuss major features
3. Submit PRs with tests and documentation
4. Update this roadmap when completing items

---

[← Back to README](../README.md)
