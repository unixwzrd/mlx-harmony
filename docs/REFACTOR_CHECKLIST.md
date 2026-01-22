# Refactor Checklist

**Created**: 2026-01-22
**Updated**: 2026-01-22

## Purpose

Track the refactor + performance plan from [tmp/Codex_Instructions-01.md](../tmp/Codex_Instructions-01.md) with actionable steps and a place for status/notes.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Priority Work Items (Codex_Instructions-01)

### 1) Remove StreamableParser from per-token hot path

- [ ] Stop calling `StreamableParser.process(token)` inside `chat_generation.stream_generation`
- [ ] Remove parser seeding from prompt tail tokens
- [ ] Parse completion once after generation (single-pass parsing)
- [ ] Confirm tool call detection still works post-parse
- [ ] Notes:

### 2) Make stop behavior consistent and channel-safe

- [ ] Keep Harmony stop tokens from `encoding.stop_tokens_for_assistant_actions()`
- [ ] Avoid early stops based on StreamableParser channel state
- [ ] Ensure parsing still finalizes without stop token injection
- [ ] Notes:

### 3) Separate model budget from display budget

- [ ] Remove analysis-budget early stop in `chat_generation.stream_generation`
- [ ] Keep `truncate_thinking`/`truncate_response` only for display/save
- [ ] Notes:

### 4) Loop detection policy (fast path vs guard mode)

- [ ] Add config/CLI flag to disable loop detection in production/benchmark
- [ ] Default to cheap detector only (e.g., repeated single token)
- [ ] Move heavier n-gram detection behind the flag
- [ ] Notes:

### 5) Cache clearing + wired memory stability

- [ ] Default `clear_cache` to False when `mlock=true`
- [ ] Avoid `mx.clear_cache()` inside per-token loop by default
- [ ] If needed, clear between runs or on memory pressure only
- [ ] Notes:

### 6) Prompt-cache reuse as conversation grows

- [ ] Ensure prompt prefix stays stable (avoid dynamic fields early)
- [ ] Preserve longest-common-prefix during truncation
- [ ] Track `prefill_start_offset` usage across turns
- [ ] Notes:

### 7) Resume generation after max_tokens

- [ ] Simple resume: append partial assistant + continuation instruction + regen
- [ ] Fast resume: add resumable state to `stream_generate`/`resume_generate`
- [ ] No duplicated channel headers on resume
- [ ] Notes:

### 8) Fix repetition across Harmony channels

- [ ] Confirm no duplicated channel markers in decode/clean steps
- [ ] Tune repetition penalty defaults/window for Harmony if needed
- [ ] Notes:

## Deliverables

- [ ] Benchmark note (before/after): prefill time, TPS, CPU%, GPU%
- [ ] Unit tests for:
  - [ ] Stop token behavior (assistant action boundary)
  - [ ] Resume after max_tokens completes final channel
  - [ ] Tool call detection still works post-parse

## Cross‑Cutting Tasks

- [ ] Keep benchmark harness repeatable (same prompt, tokens, temperature)
- [ ] After each change: run tests/imports + one benchmark
- [ ] Report deltas: TPS, CPU%, GPU%, wired memory oscillation, top 20 cProfile
- [ ] Confirm “no numpy in hot path”
