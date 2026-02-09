# Next Sprint Checklist

**Created**: 2026-01-28
**Updated**: 2026-02-09

## Purpose

Track the next-sprint work items across major areas (engineering, performance, tooling, and docs).

## Document Ownership

- This is the source of truth for active 1-2 day execution work.
- Refactor-only tasks live in [REFACTOR_CHECKLIST.md](./REFACTOR_CHECKLIST.md).
- Long-horizon plans live in [ROADMAP.md](./ROADMAP.md).
- Untriaged/parking-lot items live in [TODO.md](./TODO.md).

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Current Sprint (1–2 Days)

### CLI/Server Convergence (Priority)

- [x] Define a backend adapter contract (`GenerationClient`) for shared front-end usage.
- [x] Implement Local adapter (`LocalGenerationClient`) wrapping existing local generation path.
- [x] Implement HTTP adapter (`ServerGenerationClient`) for API-backed generation.
- [x] Switch server STDIO client to the shared front-end/driver loop.
- [x] Extract command/OOB processing into `chat_commands.py`.
- [x] Consolidate interactive and prompt-list front-end handling into one shared input-processing path.
- [x] Remove obsolete `chat_controller.py` and `chat_driver.py` modules after call sites moved.
- [ ] Complete adapter parity:
  - [x] Route server non-stream chat completions through the shared `run_chat_turn` pipeline.
  - [ ] Move remaining generation/logging behavior behind adapter usage (no duplicated turn pipeline).
  - [ ] Ensure local and server paths use the same prompt/parse/retry flow at the driver boundary.
- [ ] Complete output parity (CLI vs server-backed client):
  - [x] Runtime artifact parity (debug/timing/profile output content and structure).
  - [ ] User-visible rendering parity (same presented assistant output and formatting).
  - [ ] Match channel handling (`analysis` + `final`) and fallback behavior.
  - [ ] Match Harmony token cleanup/render behavior.
  - [ ] Match markdown rendering behavior.
- [x] Complete logging/artifact parity contract:
  - [x] Define required artifacts per component (`prompt.*`, `completion.*`, `parse.*`, `retry.*`, debug logs, profile files).
  - [x] Enforce same filenames/layout under `runs/.../logs/{cli,server}` and `runs/.../metrics/{cli,server}`.
  - [x] Ensure both paths generate the same report/plot/profile outputs.
- [ ] Shared core-path parity:
  - [ ] Use same parameter-setting path for CLI and server-backed client.
  - [ ] Use same model-load/inference path behind adapter boundary where possible.
  - [ ] Keep only front-end transport differences (local function calls vs HTTP).
- [ ] Startup/model lifecycle parity:
  - [ ] Add explicit server preload/warmup behavior at startup.
  - [x] Expose model load state in health metadata.

### Benchmarking & Acceptance (ON-GOING)

- [x] Stabilize benchmark artifact collection paths (absolute path handling in harness/preserve scripts).
- [x] Split benchmark execution into dedicated profilers (`profile_cli.sh`, `profile_server.sh`) with shared post-processing.
- [ ] Run KV windowing acceptance checks across multiple `max_kv_size` configs.
- [ ] Confirm long-run TPS stops degrading past the window.
- [ ] Verify wired memory plateaus (no large oscillations).
- [ ] Confirm `prompt_tokens` stays below `max_context_tokens` with safety margin in long runs.
- [ ] Record benchmark notes (prefill time, TPS, CPU%, GPU%).

## Next Sprint (Soon, but Not This 1–2 Day Block)

### Endpoint-to-Function Mapping (Plan)

- [ ] Define the shared backend functions and map them to server endpoints:
  - [ ] `POST /v1/chat/completions` → `chat_backend.generate` (shared prompt/parse/retry path)
  - [ ] `GET /v1/models` → `model_registry.list_models` (from profiles/config/models_dir).
  - [ ] `GET /v1/health` → `server_status.health` (model load status + metadata).
  - [ ] `POST /v1/config` (internal) → `config_registry.set_runtime_params` (safe runtime params only).
- [ ] Ensure CLI calls the same backend functions directly (no HTTP), with identical inputs.

## Later (Requires More Design / Depends on Prior Work)

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
- [ ] Add server model-management parity endpoints:
  - [ ] `/v1/models` listing from configured model source.
  - [ ] clear policy for runtime-selectable vs startup-only model/config changes.

## Documentation (Ongoing, update as modules change)

- [ ] Add User Guide stub (usage + tuning, including perf-mode).
- [ ] Add Developer Guide stub (architecture + extension points).
- [ ] Consolidate references to [NOTES.md](./NOTES.md) to avoid duplication.
- [ ] Generate API reference via `pydoc-markdown` into `docs/API_REFERENCE.md`.
- [ ] Document CLI vs server-backed client architecture:
  - [ ] shared front-end loop
  - [ ] adapter boundary (local/python vs HTTP)
  - [ ] artifact/log parity contract

## Performance Investigations (Ongoing)

- [ ] Investigate unbuffered model-weight loading (IO impact + MLX compatibility).
- [ ] Review `generate_standalone.py` hot loop for micro-optimizations after latest profiling.
- [ ] Add full call-tree Graphviz output option (node/edge thresholds set to `0`).
- [ ] Update dependency graphs after convergence/module reduction and record diffs.

## Tests (Refactor Guardrails)

- [x] Unit tests for command module (`chat_commands.py`).
- [x] Unit tests for multiline input modes (`\` continuation and `\\ ... \\` block).
- [ ] Add adapter parity tests (local vs server client contract behavior).
- [ ] Add server-backed client output-equivalence tests against CLI baseline.
