# Refactor Checklist

**Created**: 2026-01-22
**Updated**: 2026-01-25

## Purpose

Track the refactor + performance plan from [Codex_Instructions-06](../tmp/Codex_Instructions-06.md) with actionable steps and status.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Priority Work Items (Codex_Instructions-06)

### 1) KV Windowing (RotatingKVCache)

- [x] Add `max_kv_size` (or `kv_window_tokens`) to `PromptConfig`.
- [x] Thread `max_kv_size` into `make_prompt_cache(...)` from `TokenGenerator.generate`.
- [x] Default to rotating cache when `max_kv_size` is set.
- [ ] Acceptance: long‑run TPS stops degrading past the window.
- [ ] Acceptance: wired memory plateaus (no large oscillations).

### 2) Prompt Size Control (steady state)

- [x] Tighten prompt truncation to stay below max context in long runs.
- [ ] Prefer dropping oldest turns earlier (performance mode).
- [x] Optional perf mode settings: smaller `max_tokens`, smaller retained history, smaller KV window.
- [~] Define perf mode settings in config/CLI (opt-in) and document intended use.
- [ ] Acceptance: prompt_tokens stays below `max_context_tokens` with a safety margin in long runs.

### 3) Hot Loop Cleanup (`generate_standalone.stream_generate`)

- [x] Remove Python `generated_tokens` list (use `generated_token_count` only).
- [x] Run repetition detection every N tokens (e.g., every 8).
- [x] Avoid per‑token `mx.concatenate` for repetition window; update every N tokens or rolling buffer.

### 4) Cache Clearing Discipline

- [ ] Ensure `clear_cache_generation` stays false in normal runs.
- [ ] Avoid periodic `mx.clear_cache()` unless debugging memory churn.

### 5) Instrumentation (make next profile decisive)

- [x] Add `kv_len` (effective context length) to timing stats.
- [x] Add `max_kv_size` and `repetition_window` columns to timing stats.
- [x] Add `loop_detection_mode` column to timing stats.

### 1) Controller + Adapter Architecture

- [x] Split controller stack into focused modules (`chat_turn`, `chat_retry`, `chat_attempt`, `chat_adapters`, `chat_types`).
- [x] Keep `chat_controller.py` as a lightweight facade (re-exports only).
- [x] Define `ModelAdapter` protocol and shared dataclasses for controller types.
- [x] Implement `HarmonyAdapter` (parsing/display uses `parse_harmony_response`).
- [x] Implement `NativeAdapter` (streaming decode + final channel).
- [x] Move model-specific parsing/display out of `chat.py`.
- [x] `chat.py` delegates to the turn runner (goal <250 LOC).

### 2) Token-First Repetition Detection (new module)

- [x] Add `mlx_harmony/repetition_tokens.py` with `TokenRepetitionConfig` + `TokenRepetitionDetector`.
- [x] Integrate detector into `generate_standalone.stream_generate` (primary hot loop).
- [x] On detection: set `finish_reason="stop"` and `stop_reason="loop_detected"`.
- [x] Ensure `TokenGenerator.generate` captures `last_stop_reason` in all stop paths.

### 3) Model-Agnostic Retry Policy (controller-owned)

- [x] Retry on `finish_reason=="length"`, `stop_reason=="loop_detected"`, or post-parse repetition.
- [x] Retry attempts capped (existing `max_resume_attempts`).
- [x] Temporary hyperparameter bumps during retry (penalty/context/loop_detection).
- [x] Restore user-configured hyperparameters after retry.
- [x] Retry guidance is transient and not persisted in conversation history.

### 4) Stream Generation: Model-Agnostic

- [x] Remove decoding/printing logic from `chat_generation.stream_generation`.
- [x] Return tokens only; adapter handles decoding/display.
- [x] Remove Harmony-specific logic from `chat_generation`.

### 5) Prompt + Artifact Logging (exact rendered prompt)

- [x] For each attempt, write artifacts under `logs/`:
  - [x] `prompt.full.<turn>.<attempt>.txt`
  - [x] `prompt.tokens.<turn>.<attempt>.json`
  - [x] `completion.tokens.<turn>.<attempt>.json`
  - [x] `completion.raw.<turn>.<attempt>.txt`
  - [x] `completion.cleaned.<turn>.<attempt>.txt`
  - [x] `parse.channels.<turn>.<attempt>.json`
  - [x] `retry.decision.<turn>.<attempt>.json`
- [x] Ensure `prepare_prompt()` writes the exact rendered prompt string.

### 6) Stop Behavior + Parsing Consistency

- [x] Ensure stop tokens are from `encoding.stop_tokens_for_assistant_actions()`.
- [x] Ensure no early stop based on channel heuristics.
- [x] Ensure parsing completes without stop token injection (or explicitly append for parsing only).
- [x] Use permissive Harmony parsing retry (`strict=False`) before raw-text fallback.
- [x] Decode Harmony debug text with `decode_utf8` where available.

### 7) Loop-Detection Flags

- [x] Keep `off/cheap/full` modes; `cheap` default.
- [x] Ensure hot-loop checks are bounded and `check_every` is honored.

### 8) Post-Parse Repetition Check (optional fallback)

- [x] Add light text repetition checks per channel (analysis/commentary/final).
- [x] Trigger retry decision when repeated phrases detected.

## Deliverables

- [ ] Update `CHANGELOG.md` for each batch.
- [ ] Benchmark note (prefill time, TPS, CPU%, GPU%).
- [ ] Unit tests:
  - [ ] Stop token behavior
  - [ ] Retry after max_tokens
  - [ ] Tool call detection after post-parse

## Cross‑Cutting Tasks

- [ ] Keep benchmark harness repeatable (same prompt, tokens, temperature).
- [ ] After each change: run tests/imports + one benchmark.
- [ ] Report deltas: TPS, CPU%, GPU%, wired memory oscillation, top 20 cProfile.
- [ ] Confirm “no numpy in hot path”.
