# Refactor Checklist

**Created**: 2026-01-22
**Updated**: 2026-02-09

## Purpose

Track the refactor + performance plan from [Codex_Instructions-06](../tmp/Codex_Instructions-06.md) with actionable steps and status.

## Document Ownership

- This file tracks refactor and convergence tasks only (module boundaries, shared paths, cleanup).
- Active short-cycle execution is tracked in [NEXT_SPRINT_CHECKLIST.md](./NEXT_SPRINT_CHECKLIST.md).
- Product and long-horizon planning is tracked in [ROADMAP.md](./ROADMAP.md).
- New/untriaged ideas go to [TODO.md](./TODO.md) first, then get promoted.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Scope Note

- This checklist now tracks remaining refactor/convergence architecture work.
- Benchmark acceptance and API feature rollout are tracked in [NEXT_SPRINT_CHECKLIST.md](./NEXT_SPRINT_CHECKLIST.md).
- Historical completed refactor batches remain in [CHANGELOG.md](../CHANGELOG.md).

## Remaining Refactor Work Items

### Controller + Adapter Architecture

- [x] Split controller stack into focused modules (`chat_turn`, `chat_retry`, `chat_attempt`, `chat_adapters`, `chat_types`).
- [x] Remove obsolete `chat_controller.py` facade after call sites moved to direct modules.
- [x] Remove obsolete `chat_driver.py` loop module after shared loop moved to `chat_frontend.py`.
- [x] Define `ModelAdapter` protocol and shared dataclasses for controller types.
- [x] Implement `HarmonyAdapter` (parsing/display uses `parse_harmony_response`).
- [x] Implement `NativeAdapter` (streaming decode + final channel).
- [x] Move model-specific parsing/display out of `chat.py`.
- [x] `chat.py` delegates to the turn runner (goal <250 LOC).

### Backend/Frontend Demarcation Finalization

- [ ] Introduce a single backend service boundary used by both CLI-local and server HTTP paths.
- [ ] Ensure no duplicated prompt/parse/retry orchestration in transport adapters.
- [ ] Ensure server endpoint handlers remain thin wrappers around shared backend methods.
- [ ] Ensure client remains a thin transport adapter plus shared frontend loop.

### Module Boundary Cleanup

- [x] Add `mlx_harmony/repetition_tokens.py` with `TokenRepetitionConfig` + `TokenRepetitionDetector`.
- [x] Integrate detector into `generate_standalone.stream_generate` (primary hot loop).
- [x] On detection: set `finish_reason="stop"` and `stop_reason="loop_detected"`.
- [x] Ensure `TokenGenerator.generate` captures `last_stop_reason` in all stop paths.
- [ ] Rename/rehome overloaded `chat_*` modules where names do not match responsibility.
- [ ] Keep module/class/function Google-style docstrings current during each refactor batch.

## Guardrails Per Batch

- [ ] Update [SOURCE_FILE_MAP.md](./SOURCE_FILE_MAP.md) for module moves/renames.
- [ ] Update [CHANGELOG.md](../CHANGELOG.md) for each merged refactor batch.
- [ ] Run `python -m compileall src`.
- [ ] Run `python -m pytest -q --maxfail=3 --disable-warnings || true`.
