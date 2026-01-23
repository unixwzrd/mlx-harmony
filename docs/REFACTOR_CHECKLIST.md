# Refactor Checklist

**Created**: 2026-01-22
**Updated**: 2026-01-23

## Purpose

Track the refactor + performance plan from [tmp/Codex_Instructions-01.md](../tmp/Codex_Instructions-01.md) with actionable steps and a place for status/notes.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Priority Work Items (Codex_Instructions-01)

### 1) Remove StreamableParser from per-token hot path

- [x] Stop calling `StreamableParser.process(token)` inside `chat_generation.stream_generation`
- [x] Remove parser seeding from prompt tail tokens
- [x] Parse completion once after generation (single-pass parsing)
- [x] Confirm tool call detection still works post-parse
- [ ] Notes: Tool parsing now reuses header prepend + stop token when parsing completion tokens; needs run validation.

### 2) Make stop behavior consistent and channel-safe

- [ ] Keep Harmony stop tokens from `encoding.stop_tokens_for_assistant_actions()`
- [x] Avoid early stops based on StreamableParser channel state
- [~] Ensure parsing still finalizes without stop token injection
- [~] Add safe fallback when completion lacks Harmony headers (raw text)
- [ ] Notes: Parsing now appends stop token for parse-only when missing; raw-text fallback added for malformed outputs; header-prepended parses run non-strict; needs validation in runs.

### 3) Separate model budget from display budget

- [x] Remove analysis-budget early stop in `chat_generation.stream_generation`
- [x] Keep `truncate_thinking`/`truncate_response` only for display/save
- [ ] Notes:

### 4) Loop detection policy (fast path vs guard mode)

- [x] Add config/CLI flag to disable loop detection in production/benchmark
- [x] Default to cheap detector only (e.g., repeated single token)
- [x] Move heavier n-gram detection behind the flag
- [x] Gate chat-level loop detection to `off/cheap/full` to reduce per-token CPU
- [ ] Notes: Validate with benchmark run (cheap vs full/off) and log loop_detected rates.

### 5) Cache clearing + wired memory stability

- [x] Default `clear_cache` to False when `mlock=true`
- [x] Avoid `mx.clear_cache()` inside per-token loop by default
- [ ] If needed, clear between runs or on memory pressure only
- [ ] Notes:

### 6) Prompt-cache reuse as conversation grows

- [x] Ensure prompt prefix stays stable (avoid dynamic fields early)
- [x] Preserve longest-common-prefix during truncation
- [x] Track `prefill_start_offset` usage across turns
- [ ] Notes: Built-in time placeholders now frozen at prompt config load; LCP reused via prompt cache; prefill_start_offset logged in metrics.

### 7) Resume generation after max_tokens

- [x] Simple resume: append partial assistant + continuation instruction + regen
- [ ] Fast resume: add resumable state to `stream_generate`/`resume_generate`
- [ ] No duplicated channel headers on resume
- [x] Notes: Resume now avoids injecting partial assistant content into the prompt; recovery uses a placeholder assistant turn.
- [x] Notes: Resume now applies stronger repetition guards, caps recovery lists, and can retry on loop-detected stalls.
- [x] Notes: Recovery prompts no longer assume list output and resume hyperparameters restore after recovery.
- [x] Notes: Recovery prompt now adapts list vs paragraph based on the user request.

### 8) Fix repetition across Harmony channels

- [x] Confirm no duplicated channel markers in decode/clean steps
- [~] Tune repetition penalty defaults/window for Harmony if needed
- [~] Add truncation marker when max_tokens ends mid-channel
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
