# TODO Checklist

Quick reference checklist for active work items. For detailed roadmap, see [ROADMAP.md](./ROADMAP.md).

## 🚀 Current Sprint / Active Work

<!-- Update this section with items you're actively working on -->

- [ ] Memory management (mlock) refinements - ensuring buffers stay wired
- [ ] Testing and validation of mlock implementation

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
- [x] Memory management infrastructure (mlock, pre-warming)
- [x] Comprehensive documentation with navigation links
- [x] Prompt config reference documentation
- [x] Memory management guide

---

## 🎯 Next Up (Priority Order)

### Tool Executors

- [ ] Browser tool implementation
- [ ] Python tool implementation  
- [ ] Apply patch tool implementation

### Testing

- [x] Unit tests for config module
- [x] Unit tests for generator
- [x] Unit tests for tools
- [x] Integration tests for chat loop (conversation save/load)
- [x] Server API tests (FastAPI endpoints, streaming, error handling)
- [x] Server integration tests with real model
- [ ] End-to-end integration tests (full chat flow with model)
- [x] CI/CD setup
  - [x] GitHub Actions workflow (ci.yml)
  - [x] Linting (black, ruff)
  - [x] Test matrix (Python 3.12, 3.13)
  - [x] Coverage reporting

### Documentation

- [ ] API documentation setup
- [ ] Tool usage tutorial
- [x] Examples directory
  - [x] Basic chat example
  - [x] One-shot generation example
  - [x] Prompt config example
  - [x] Profile usage example
  - [x] Conversation resume example
  - [x] Custom placeholders example
  - [x] Sampling parameters example
  - [x] Server client example
  - [x] Tools infrastructure example
- [x] Troubleshooting guide

---

## 📋 Quick Add

_Add quick TODOs here as they come up during development:_

- [ ] Server request/response logging (optional debug mode)

---

**Last Updated**: 2026-01-07

[← Back to README](../README.md)
