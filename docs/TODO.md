# TODO Checklist

**Created**: 2026-01-09
**Updated**: 2026-01-18

Quick reference checklist for **active short-term work items**. For longer-term planning and detailed feature roadmaps, see [ROADMAP.md](./ROADMAP.md).

**Priority Legend**:
- **P0** = blocking / urgent
- **P1** = next up
- **P2** = nice-to-have

## 🚀 Current Sprint / Active Work

<!-- Update this section with items you're actively working on -->

### P0 (Blocking)

**Voice I/O**

- [ ] STT token normalization: single canonical implementation wired into `gen.step`
- [ ] Tune STT silence detection to avoid cutoffs/bleed between turns
- [ ] TTS prompt buffer reset/flush between turns to avoid residual carryover
- [ ] Tie TTS chunk sizing to prompt config limits

### P1 (Next Up)

**Generation / Pipeline**

- [ ] Verify `generate_standalone` stop-token/finish-reason streaming semantics
- [ ] Finalize standalone loader (`mlock`/`no_fs_cache`) and document behavior
- [ ] Restore or intentionally drop Harmony tool-role/example handling in prompt build
- [ ] Restore deterministic timing/perf metrics (prompt start time, per-turn latency, throughput)

**Testing**

- [ ] End-to-end integration tests (full chat flow with model)
- [ ] Testing and validation of mlock implementation

### P2 (Nice-to-have)

**Performance / Stability**

- [ ] Baseline performance notes for CLI/server output paths
- [ ] Document any breaking changes or compatibility assumptions

**Moshi Integration (prioritized)**

- [ ] Moshi integration pass (config/CLI flag/wiring/tests/docs) — see [Moshi Integration Checklist](#moshi-integration-checklist)

**Testing**

- [ ] Prompt rendering tests for Harmony vs non-Harmony path (stubbed, no model download)
- [ ] Edge-case tests for empty prompts and mixed roles

**Documentation**

- [ ] API documentation setup (Sphinx/MkDocs)
- [ ] Tool usage tutorial (when tool executors are implemented)
- [ ] Update [README.md](../README.md) and [FEATURES_FROM_MLX.md](./FEATURES_FROM_MLX.md) if defaults/behavior change
- [ ] Add architecture overview to [ROADMAP.md](./ROADMAP.md) if module boundaries change
- [ ] Keep [tests/README.md](../tests/README.md) updated with new markers/requirements

---

### Completed (Core & CLI)

- [x] Core TokenGenerator implementation
- [x] CLI tools (chat, generate, server)
- [x] Tool infrastructure (parsing + stubs)
- [x] Profiles system
- [x] Assistant greeting support

### Completed (Conversation & Logging)

- [x] Conversation logging with timestamps and hyperparameters
- [x] Conversation resume functionality (load/save JSON)
- [x] Debug mode with file output (`--debug-file`)
- [x] Fixed newline preservation in chat history

### Completed (Prompt & UX)

- [x] SessionConfig (formerly PromptConfig) with dynamic placeholders (TIME, TIMEZ, TIMEA, TIMEU)
- [x] Added \help command for out-of-band commands
- [x] Added error handling for invalid \ commands
- [x] Beautiful markdown rendering for assistant responses (rich library integration)
- [x] Removed prewarm_cache feature

## 🎯 Short-Term Priorities

_These are items actively planned for upcoming releases. For comprehensive long-term planning, see [ROADMAP.md](./ROADMAP.md)._

### Completed (Testing)

- [x] Fixed sampling implementation to match mlx-lm exactly
- [x] Fixed MLX API compatibility issues (zeros_like, scatter, searchsorted)

### Completed (Documentation)

- [x] Prompt config reference documentation
- [x] Memory management guide
- [x] Comprehensive documentation with navigation links

---

**Note**: Tool executors (browser, python, apply_patch), model caching, prompt caching, and other major features are tracked in [ROADMAP.md](./ROADMAP.md) under High/Medium priority sections.

---

## 📋 Long-Term / Backlog (P3)

_Lower-priority items to revisit after the current sprint._

- [ ] Server request/response logging (optional debug mode)
- [ ] Add unit tests to validate Harmony vs non-Harmony decode paths (stubbed, no model downloads)
- [ ] Handle stop token sequences that are multi-token in [generate_standalone.py](../src/mlx_harmony/generate_standalone.py)
- [ ] Add per-turn IDs + parent/child links for chat logs
- [ ] Add max context window handling from model config (with override)
- [ ] Add seed support for deterministic chat/profiling runs
- [ ] Reconcile `voice_moshi` fixes from tmp (merge working/broken deltas cleanly)
- [ ] Restore Moshi STT/TTS improvements (thresholds, warmup, VAD/silence handling)
- [ ] Improve TTS chunking/sentence splitting to reduce choppiness
- [ ] Review hot-mic/start-stop UX and make it smoother (avoid CPU spikes)
- [ ] Verify `TokenGenerator` token-id extraction via re-encoding with current tokenizer

---

## Moshi Integration Checklist

_Short-term checklist for integrating Moshi STT/TTS behind a CLI flag._

- [ ] User action items (prereqs I can’t do for you)
  - [ ] Install `moshi-mlx` (and any required system deps like `sounddevice`/`rustymimi`).
  - [ ] Download local MLX STT model weights: `models/STT/stt-2.6b-en-mlx`.
  - [ ] Download local MLX TTS model weights: `models/TTS/moshiko-mlx-q8`.
  - [ ] (Optional) Download voice embeddings repo for TTS (`DEFAULT_DSM_TTS_VOICE_REPO`) if not bundled.
  - [ ] Confirm mic permissions on macOS for terminal/CLI.
  - [ ] Confirm audio output device works for playback (or provide output wav path).
- [ ] Confirm Moshi dependencies installed (`moshi-mlx` and MLX models).
- [ ] Read and reference [../moshi/README.md](../moshi/README.md) and [../moshi-dsm/README.md](../moshi-dsm/README.md).
- [ ] Add `--moshi` CLI flag in [src/mlx_harmony/cli_args.py](../src/mlx_harmony/cli_args.py).
- [ ] Add Moshi config file support (see [Moshi Config](./MOSHI_CONFIG.md)).
- [ ] Auto-load `configs/moshi.json` when `--moshi` is set and no config path is supplied.
- [ ] Add STT listen duration and optional VAD settings (CLI + config).
- [ ] Add TTS chunking controls for sentence/length-based streaming.
- [ ] Add Moshi smoke-test flag for quick validation.
- [ ] Add voice status output and timing logs for STT/TTS.
- [ ] Explore parallel STT + TTS processing with non-speech filtering (avoid barge-in on coughs/uhs/sneezes).
- [ ] Add Moshi config/unit tests (skip if deps missing).
- [ ] Define Moshi adapter interfaces (STT/TTS) in a new voice module.
- [ ] Implement MLX STT adapter (mic → partial/final text).
- [ ] Implement MLX TTS adapter (text chunks → audio stream).
- [ ] Wire `chat.py` to use Moshi adapters when `--moshi` is enabled.
- [ ] Add minimal error handling when `moshi_mlx` is missing (fail fast with guidance).
- [ ] Add basic smoke test or stub test for the Moshi flag path.
- [ ] Document CLI usage and required models in [docs/ROADMAP.md](./ROADMAP.md) or README.

**Note**: This file focuses on **short-term active work items**. For comprehensive long-term planning, feature requests, and detailed roadmaps, see [ROADMAP.md](./ROADMAP.md).

[← Back to README](../README.md)
