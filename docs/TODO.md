# TODO Checklist

**Created**: 2026-01-09
**Updated**: 2026-01-25

Quick reference checklist for **active short-term work items**. For longer-term planning and detailed feature roadmaps, see [ROADMAP.md](./ROADMAP.md).

## 🚀 Current Sprint / Active Work

<!-- Update this section with items you're actively working on -->

- [ ] Testing and validation of mlock implementation
- [~] Integration test run in progress (full chat flow with model)
- [ ] Baseline performance notes for CLI/server output paths
- [ ] Document any breaking changes or compatibility assumptions
- [ ] Investigate response richness regression (markdown/plaintext output and response length)

---

## 🎯 Short-Term Priorities

_These are items actively planned for upcoming releases. For comprehensive long-term planning, see [ROADMAP.md](./ROADMAP.md)._

### Testing

- [ ] End-to-end integration tests (full chat flow with model)
- [ ] Testing and validation of mlock implementation
- [ ] Prompt rendering tests for Harmony vs non-Harmony path (stubbed, no model download)
- [ ] Edge-case tests for empty prompts and mixed roles

### Core Features

- [x] Core TokenGenerator implementation
- [x] PromptConfig with dynamic placeholders (including time placeholders: TIME, TIMEZ, TIMEA, TIMEU)
- [x] Profiles system
- [x] Tool infrastructure (parsing + stubs)
- [x] CLI tools (chat, generate, server)
- [x] Conversation logging with timestamps and hyperparameters
- [x] Conversation resume functionality (load/save JSON)
- [x] Debug mode with file output (`--debug-file`)
- [x] Assistant greeting support
- [x] Memory management infrastructure (mlock)
- [x] Fixed MLX API compatibility issues (zeros_like, scatter, searchsorted)
- [x] Fixed sampling implementation to match mlx-lm exactly
- [x] Fixed newline preservation in chat history
- [x] Added \help command for out-of-band commands
- [x] Added error handling for invalid \ commands
- [x] Removed prewarm_cache feature

### Documentation

- [x] Basic documentation and examples
- [x] Comprehensive documentation with navigation links
- [x] Prompt config reference documentation
- [x] Memory management guide
- [x] Beautiful markdown rendering for assistant responses (rich library integration, similar to glow/mdless)
- [ ] API documentation setup (Sphinx/MkDocs)
- [ ] Tool usage tutorial (when tool executors are implemented)
- [ ] Update [README.md](../README.md) and [FEATURES_FROM_MLX.md](./FEATURES_FROM_MLX.md) if defaults/behavior change
- [ ] Add architecture overview to [ROADMAP.md](./ROADMAP.md) if module boundaries change
- [ ] Keep [tests/README.md](../tests/README.md) updated with new markers/requirements

---

**Note**: Tool executors (browser, python, apply_patch), model caching, prompt caching, and other major features are tracked in [ROADMAP.md](./ROADMAP.md) under High/Medium priority sections.

---

## 📋 Quick Add

_Add quick TODOs here as they come up during development:_

- [ ] Server request/response logging (optional debug mode)
- [ ] Add unit tests to validate Harmony vs non-Harmony decode paths (stubbed, no model downloads)
- [ ] Handle stop token sequences that are multi-token in [generate_standalone.py](../src/mlx_harmony/generate_standalone.py)
- [ ] Add per-turn IDs + parent/child links for chat logs
- [ ] Add max context window handling from model config (with override)
- [ ] Add seed support for deterministic chat/profiling runs
- [ ] Add CLI option to manually retry a truncated answer

---

## 📋 Quick Add

_Add quick TODOs here as they come up during development:_

- [ ] Server request/response logging (optional debug mode)

---

**Note**: This file focuses on **short-term active work items**. For comprehensive long-term planning, feature requests, and detailed roadmaps, see [ROADMAP.md](./ROADMAP.md).

[← Back to README](../README.md)
