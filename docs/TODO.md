# TODO Checklist

**Last Updated**: 2026-01-09

Quick reference checklist for **active short-term work items**. For longer-term planning and detailed feature roadmaps, see [ROADMAP.md](./ROADMAP.md).

## 🚀 Current Sprint / Active Work

<!-- Update this section with items you're actively working on -->

- [ ] Testing and validation of mlock implementation
- [ ] End-to-end integration tests (full chat flow with model)

---

## ✅ Recently Completed

- [x] Core TokenGenerator implementation
- [x] PromptConfig with dynamic placeholders (including time placeholders: TIME, TIMEZ, TIMEA, TIMEU)
- [x] Profiles system
- [x] Tool infrastructure (parsing + stubs)
- [x] CLI tools (chat, generate, server)
- [x] Basic documentation and examples
- [x] Initial GitHub commit
- [x] Conversation logging with timestamps and hyperparameters
- [x] Conversation resume functionality (load/save JSON)
- [x] Debug mode with file output (`--debug-file`)
- [x] Assistant greeting support
- [x] Memory management infrastructure (mlock)
- [x] Comprehensive documentation with navigation links
- [x] Prompt config reference documentation
- [x] Memory management guide
- [x] Beautiful markdown rendering for assistant responses (rich library integration, similar to glow/mdless)
- [x] Fixed MLX API compatibility issues (zeros_like, scatter, searchsorted)
- [x] Fixed sampling implementation to match mlx-lm exactly
- [x] Fixed newline preservation in chat history
- [x] Added \help command for out-of-band commands
- [x] Added error handling for invalid \ commands
- [x] Removed prewarm_cache feature

---

## 🎯 Short-Term Priorities

_These are items actively planned for upcoming releases. For comprehensive long-term planning, see [ROADMAP.md](./ROADMAP.md)._

### Testing

- [ ] End-to-end integration tests (full chat flow with model)
- [ ] Testing and validation of mlock implementation

### Documentation

- [ ] API documentation setup (Sphinx/MkDocs)
- [ ] Tool usage tutorial (when tool executors are implemented)

---

**Note**: Tool executors (browser, python, apply_patch), model caching, prompt caching, and other major features are tracked in [ROADMAP.md](./ROADMAP.md) under High/Medium priority sections.

---

## 📋 Quick Add

_Add quick TODOs here as they come up during development:_

- [ ] Server request/response logging (optional debug mode)

---

## 📋 Quick Add

_Add quick TODOs here as they come up during development:_

- [ ] Server request/response logging (optional debug mode)

---

**Note**: This file focuses on **short-term active work items**. For comprehensive long-term planning, feature requests, and detailed roadmaps, see [ROADMAP.md](./ROADMAP.md).

[← Back to README](../README.md)
