# Changelog

All notable changes to mlx-harmony will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-06

### Added
- Initial release of mlx-harmony
- `TokenGenerator` class for multi-model MLX inference with automatic Harmony format for GPT-OSS models
- CLI tools: `mlx-harmony-chat`, `mlx-harmony-generate`, `mlx-harmony-server`
- PromptConfig system with JSON configuration for:
  - Harmony prompt fragments (system_model_identity, reasoning_effort, conversation_start_date, knowledge_cutoff, developer_instructions)
  - Dynamic placeholder expansion (`<|DATE|>`, `<|DATETIME|>`, `{custom}`)
  - Sampling defaults (temperature, top_p, min_p, top_k, repetition_penalty, etc.)
- Profile system for bundling model paths with prompt configs
- GPT-OSS tool infrastructure:
  - Tool call parsing from Harmony messages
  - Tool execution framework (browser, python, apply_patch - currently stubbed)
  - Automatic tool call detection and loop handling in chat CLI
- Full sampling hyperparameter control:
  - temperature, top_p, min_p, top_k
  - xtc_probability, xtc_threshold, min_tokens_to_keep
  - repetition_penalty, repetition_context_size
  - logit_bias
- OpenAI-compatible HTTP API server (`/v1/chat/completions`)
- Comprehensive documentation and examples

### Technical Details
- Built on `mlx-lm` (>=0.1.0) and `openai-harmony` (>=0.0.8)
- Automatic GPT-OSS model detection
- Harmony format used automatically for GPT-OSS, native chat templates for other models
- Supports local models, Hugging Face Hub models, and quantized models
- FastAPI-based server with streaming support

### Notes
- Tool executors (browser, python, apply_patch) are currently stubs returning "not_implemented" messages
- The infrastructure for parsing and executing tools is complete and ready for implementation
- Tests are provided but may require environment-specific MLX setup
