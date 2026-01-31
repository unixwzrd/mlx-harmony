# Next Sprint Checklist

**Created**: 2026-01-28
**Updated**: 2026-01-30

## Purpose

Track the next-sprint work items across major areas (engineering, performance, tooling, and docs).

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Benchmarking & Acceptance

- [ ] Run KV windowing acceptance checks across multiple `max_kv_size` configs.
- [ ] Confirm long-run TPS stops degrading past the window.
- [ ] Verify wired memory plateaus (no large oscillations).
- [ ] Confirm `prompt_tokens` stays below `max_context_tokens` with safety margin in long runs.
- [ ] Record benchmark notes (prefill time, TPS, CPU%, GPU%).

## Server & API Integration

- [x] Add server CLI flags and document startup basics.
- [ ] Verify `server.py` is wired to the controller stack and adapters.
- [ ] Confirm prompt config handling matches CLI behavior.
- [ ] Document API usage in the User Guide (stub until finalized).

## Tool Calling Integration

- [ ] Validate tool parsing and tool runner wiring end-to-end.
- [ ] Add minimal tool executor stubs (no real browser/python yet).
- [ ] Add a sample tool call integration test.

## OpenAI API Compatibility

- [ ] Add placeholder endpoints with `501` responses for missing OpenAI routes:
  - [ ] `/v1/completions`
  - [ ] `/v1/embeddings`
  - [ ] `/v1/audio/*` (transcriptions, translations, speech)
  - [ ] `/v1/images/*` (generations, edits, variations)
  - [ ] `/v1/moderations`
  - [ ] `/v1/files` + `/v1/batches`
  - [ ] `/v1/responses`
- [ ] Expand `/v1/chat/completions` parameter coverage:
  - [ ] `stop` (string or list)
  - [ ] `n`
  - [ ] `presence_penalty` / `frequency_penalty`
  - [ ] `logprobs`
  - [ ] `response_format` (json_schema stub)
  - [ ] `tool_choice` / `tools` (schema stub)
  - [ ] `seed` and `system_fingerprint`
- [ ] Standardize error shapes + HTTP status codes to match OpenAI responses.
- [ ] Align streaming chunk format (`chat.completion.chunk`) fields with OpenAI.

## Model Management

- [ ] Define minimal model listing / selection workflow for local models.
- [ ] Decide on a model metadata source (profiles, config, or cache scan).

## Documentation

- [ ] Add User Guide stub (usage + tuning, including perf-mode).
- [ ] Add Developer Guide stub (architecture + extension points).
- [ ] Consolidate references to [NOTES.md](./NOTES.md) to avoid duplication.

## Performance Investigations

- [ ] Investigate unbuffered model-weight loading (IO impact + MLX compatibility).
- [ ] Review `generate_standalone.py` hot loop for micro-optimizations after latest profiling.
