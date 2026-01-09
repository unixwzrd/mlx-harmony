# Changelog

All notable changes to mlx-harmony will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2026-01-09 - v0.2.0

### Added

- **Beautiful markdown rendering** for assistant responses using `rich` library (similar to `glow`/`mdless`)
  - Automatic formatting of headers, lists, code blocks (with syntax highlighting), bold/italic text, and blockquotes
  - `--no-markdown` flag to disable markdown rendering for plain text output
- **`\help` command** to list all out-of-band commands in the chat interface
- **Improved error handling** for invalid out-of-band commands (shows list of valid commands)
- **XTC special tokens auto-detection** - automatically detects EOS and newline tokens when XTC is enabled
- **Standalone generation and sampling implementation** - removed dependency on mlx-lm's generation pipeline

### Changed

- **Sampling implementation** - completely rewritten to match mlx-lm's implementation exactly:
  - Fixed `top_k` sampling to use `argpartition` and `put_along_axis` (matching mlx-lm)
  - Fixed `top_p` sampling to use `take_along_axis`/`put_along_axis` pattern instead of `searchsorted`
  - Fixed sampling order: apply filters (top_p, min_p, xtc, top_k) first, then temperature, then categorical sampling
  - Fixed temperature application to use `logprobs * (1/temp)` pattern matching mlx-lm's `categorical_sampling`
- **Generation pipeline** - now uses standalone `stream_generate` implementation with proper stop token handling
- **MLX API compatibility** - fixed all MLX API usage to match available functions:
  - Replaced `mx.zeros_like(..., dtype=...)` with `.astype()` pattern
  - Replaced `mx.scatter()` with `mx.put_along_axis()` pattern
  - All sampling functions now use only MLX functions that exist in the current API

### Fixed

- **Newline preservation** - fixed issue where newlines were being stripped from assistant messages in chat history
- **Analysis text parts error** - fixed `UnboundLocalError` when parsing Harmony messages
- **Code block rendering** - fixed markdown code block recognition by ensuring proper newlines around fenced code blocks
- **Sampling quality** - fixed nonsense/mixed language output by correcting sampling implementation to match mlx-lm

### Removed

- **`prewarm_cache` feature** - removed filesystem cache pre-warming (filesystem cache handles this naturally)

### Technical Details

- All sampling functions now match mlx-lm's implementation exactly
- Standalone generation pipeline with proper KV cache management
- Improved error messages for out-of-band commands
- Better handling of edge cases in Harmony message parsing

## 2026-01-06 - v0.1.0

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

[0.2.0]: https://github.com/unixwzrd/mlx-harmony/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/unixwzrd/mlx-harmony/releases/tag/v0.1.0
