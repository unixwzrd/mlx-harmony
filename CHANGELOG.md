# Changelog

**Created**: 2026-01-11
**Updated**: 2026-01-23

All notable changes to mlx-harmony will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2026-01-22 - Unreleased

### Added

- Generation backend protocol with GPT-OSS and native backends.
- Timing hooks in standalone generation (opt-in per prompt config).
- Tool registry and tool runner modules for pluggable tool execution.
- Minimal Harmony repro harness with stop/final boundary reporting and TPS summary.

### Changed

- Debug logging of raw LLM responses now preserves Harmony markers and stop tokens for full-stream inspection.
- Added loop/repetition guards to prevent runaway analysis/final loops during generation.
- Skipped per-token decode and logsumexp on greedy path to reduce CPU overhead.
- Truncation markers now append `[truncated]` for final/thinking when output is cut short.
- Harmony parsing now runs as a single pass after generation (no per-token StreamableParser hot path).
- Analysis/commentary budget no longer stops generation early; truncation is display-only.
- Harmony parsing appends stop token for parsing when missing to finalize message boundaries.
- Harmony parsing now falls back to raw text display when headers are missing and adds `[truncated]` on max-tokens cutoffs.
- Tool parsing now prepends the assistant header and stop token when needed to handle headerless completions.
- Built-in time placeholders are now frozen at prompt-config load to stabilize prompt prefixes.
- Harmony parsing now allows non-strict parsing when a header is prepended to handle headerless completions.
- Resume prompt now restarts from the beginning to avoid “continued” artifacts after truncation.
- Resume now omits truncated assistant content from recovery prompts to avoid seeding repetition.
- Resume now boosts repetition guards for recovery turns (penalty/context size/loop detection).
- Clarified parsing logs to indicate header-prepend is a local parse step, not a prompt re-submission.
- Resume prompt now caps recovery lists (max 8 items) to reduce repetition.
- Loop detection now supports `off`, `cheap`, and `full` modes via config/CLI, with full n-gram checks gated behind `full`.
- Chat-level loop detection now honors `off/cheap/full` to reduce per-token CPU overhead.
- Cache clearing is disabled automatically when `mlock` is enabled to reduce wired memory oscillation.
- Generation-loop cache clearing is now opt-in via `clear_cache_generation` (defaults off).
- Warning/error logs now route to the debug log file while console stays info-only.
- Harmony prompt conversation start date is now stable per run to improve prompt cache reuse.
- Prompt cache reuse now leverages longest-common-prefix and logs prefill offsets in TSV metrics.
- Added simple resume-on-max-tokens behavior to continue incomplete responses.
- Harmony default repetition context now uses a larger window when unspecified to reduce repeated analysis/final loops.
- Debug raw response logging avoids duplicating Harmony headers when the response already includes them.
- Harmony defaults to a mild repetition penalty when unspecified to reduce looping.
- Resume retry now triggers when loop detection stops generation, with stronger recovery constraints.
- Debug console output now suppresses raw responses when an automatic recovery retry is queued.
- Harmony parse diagnostics now emit full token details only under `--debug` to keep logs consistent.
- Recovery prompts no longer assume list outputs and resume hyperparameters are restored after recovery.
- Recovery prompt now adapts list vs paragraph based on the user request to avoid list-specific assumptions.

## 2026-01-20 - Unreleased

### Added

- Prompt config diagnostics for memory churn: `clear_cache`, `clear_cache_interval`, and `log_memory_stats`.

### Changed

- Generation prefill cache clearing is now configurable to help diagnose wired memory oscillation.
- Repetition penalty processing now uses unique token IDs to reduce per-token overhead.

### Fixed

- Deterministic prompt config JSON trailing comma.
- GPT-OSS cache import path after rollback.

## 2026-01-16 - v0.4.0

Major refactoring of the codebase to improve readability, maintainability, and performance. Extensive performance enhancements, including:

- Prompt token caching with per-message token IDs for faster prompt construction in Harmony mode
- Benchmarking harness using a standard data set of questions, from Alpaca Eval English dataset.
- major performance improvements after refactoring an d reviewing generation code.

### Added

- Prompt token caching with per-message token IDs for faster prompt construction in Harmony mode.
- Schema v2 chat log container with migration, consistency checks, and a migration CLI utility.
- Deterministic generation controls: `seed` and `reseed_each_turn` in prompt config and CLI.
- Dataset profiling scripts now accept extra CLI args for easier profiling sweeps.

### Changed

- Chat logs now store hyperparameters only when they change and track last-used metadata fields.
- Standalone generation loop refactored for clarity and future optimization work.

### Fixed

- Memory stability during profiling runs by defaulting to non-lazy model loading.

## 2026-01-11 - Unreleased

### Added

- Debug logging now captures full prompts and full responses for Harmony runs, with optional token logging for prompt/response (`--debug-tokens in|out|both`).
- New helper modules for chat decomposition and Harmony parsing to keep `chat.py` focused.
- Test guard to skip `@requires_model` tests unless `MLX_HARMONY_RUN_MODEL_TESTS=1` is set.

### Changed

- Prompt config is Pydantic-based, with explicit validation and corrected `example_dialogues` shape.
- Harmony prompt construction is consolidated to reduce duplication.
- Tool parsing now handles list-based message content and aligns with Harmony `Author` requirements.

### Fixed

- Placeholder parsing now supports lower-case `<|assistant|>` and similar variants.
- Server profile loading can be patched in tests with consistent imports.

## 2025-01-28 - v0.3.0

### Fixed

- **Harmony model generation stopping prematurely**: Fixed issue where generation was stopping after analysis channel instead of continuing to final channel. The `<|end|>` token (200007) is now correctly filtered out from stop tokens, as it's only a message separator. Only `<|return|>` (200002) and `<|call|>` (200012) now stop generation, allowing models to generate both analysis and final channels properly.
- **Duplicate output for Harmony models**: Removed streaming display during token generation for Harmony models to prevent duplicate output. All output is now displayed after parsing completes, ensuring proper ordering (thinking first, then response) and consistent formatting with name prefix.
- **Output ordering**: Fixed display order so thinking/analysis content displays before the final assistant response.

### Changed

- **Enhanced thinking/analysis display**: Added Rich markdown rendering support for thinking messages, matching the assistant response formatting. Thinking messages now support headers, lists, code blocks, and other markdown features, and respect the `--no-markdown` flag.
- **Improved stop token handling**: Harmony models now correctly distinguish between message separators (`<|end|>`) and generation stoppers (`<|return|>`, `<|call|>`).

### Technical

- Updated linting: Fixed all type annotations in `chat.py` to use modern Python 3.12 syntax (`dict` instead of `Dict`, `list` instead of `List`, `X | None` instead of `Optional[X]`).
- Code cleanup: Removed unused imports (`StreamState`) and fixed whitespace issues.

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
