# Refactor Checklist

**Created**: 2026-01-21
**Updated**: 2026-01-21

## Purpose

Track the refactor + performance plan from [tmp/MLX-Refactor-and-Performance-00.md](../tmp/MLX-Refactor-and-Performance-00.md) with actionable steps and a place for status/notes.

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Phase 1: Tighten Boundaries (No Behavior Changes)

### 1) Introduce RunContext

- [x] Create `src/mlx_harmony/runtime/context.py` with `RunContext` dataclass
- [x] Move session-level fields into `RunContext`
- [x] Update `chat/main.py` to build and pass `RunContext`
- [~] Replace long parameter lists with context usage
- [ ] Notes:

### 2) Split chat/main.py into bootstrap vs loop

- [x] Create `src/mlx_harmony/chat_bootstrap.py`
- [x] Move CLI/config/path/generator init into bootstrap
- [~] Keep `chat.py` as thin entrypoint
- [ ] Verify CLI behavior unchanged
- [ ] Notes:

### 3) Separate conversation storage vs rendering

- [x] Ensure `conversation/*` does not import `generation/*`
- [x] Ensure render helpers stay out of `generation/*`
- [x] Fix any cross-imports by inverting dependencies
- [ ] Notes: Conversation directory is currently empty; no cross-imports found.

## Phase 2: Stabilize Generation API (Swap Models Later)

### 4) Add minimal backend protocol

- [x] Create `src/mlx_harmony/generation/backend.py`
- [~] Define `Protocol` (encode/decode/stream_generate)
- [x] Implement GPT‑OSS backend in `generation/backends/gpt_oss_backend.py`
- [x] Implement native backend in `generation/backends/native_backend.py`
- [x] Refactor `TokenGenerator` to use backend
- [ ] Notes:
  - Protocol currently covers `prepare_prompt` + `decode`; stream generation stays in `TokenGenerator` for now.

### 5) Make tool-calling pluggable

- [x] Create `src/mlx_harmony/tools/registry.py`
- [x] Create `src/mlx_harmony/tools/runner.py`
- [x] Move tool execution into `tools/runner.py`
- [x] Update chat loop to call tool runner
- [ ] Notes:

## Phase 3: Prep for Performance Work (No Tuning Yet)

### 6) Isolate sampling + caching helpers

- [x] Add `build_logits_processors(...)`
- [x] Add `apply_logits_processors(...)`
- [x] Update generation loop to use helpers
- [ ] Notes:

### 7) Add optional timing hooks

- [x] Create `src/mlx_harmony/runtime/metrics.py` (Timer + counters)
- [x] Wire into generation loop behind config flag
- [ ] Notes:

## Performance Checklist (From tmp/MLX-Refactor-and-Performance-01.md)

**Source**: [tmp/MLX-Refactor-and-Performance-01.md](../tmp/MLX-Refactor-and-Performance-01.md)

### Phase 1: Lock down interfaces

- [~] Define `ModelBackend` interface: forward pass, cache init/update, special tokens, limits
- [x] Define `Tokenizer` interface: encode/decode/streaming decode; no numpy in hot path
- [x] Confirm `PromptRenderer` interface remains backend-agnostic
- [x] Define `Sampler` interface: sampling + logits processors separated from I/O/rendering
- [~] Deliverable: minimal protocol/classes + adapters for GPT-OSS backend without behavior change

### Phase 2: Hot-path audit

- [x] Hoist repeated attribute lookups into locals inside token loop
- [x] Pre-bind repeated dict lookups (avoid per-token `.get`) (no per-token `.get` in hot loop)
- [x] Avoid per-token string concatenation / formatting (no per-token string building in hot loop)
- [x] Eliminate numpy conversions in hot path (only at boundaries) (no numpy usage found)
- [x] Add “hot loop checklist” comment atop generation loop
- [x] Deliverable: apply audit pass, confirm no behavior change
  - Notes: use `decode_token` when detokenizer is unavailable to avoid full-sequence decode per token.

### Phase 3: Reduce call count per token

- [x] Inline trivial processors or fuse into a single step function
- [x] Collapse chains of tiny functions into one local step function
- [x] Precompute constants outside the loop; keep arrays in MLX
- [x] Deliverable: fewer Python frames per token step

### Phase 4: Allocation & memory-churn control

- [x] Preallocate recurring buffers (token buffers, masks, scratch arrays)
- [x] Avoid per-token list/dict allocation
- [x] Avoid building large intermediate strings per token
- [x] Prefer in-place cache updates where safe
- [ ] Deliverable: reduced wired-memory oscillation

### Phase 5: GPU utilization improvement

- [~] Batch work into fewer MLX calls
- [x] Reduce synchronization points
- [x] Avoid repeated materialization in hot loop (`.item()`, shape queries, conversions)
- [ ] Deliverable: increased GPU utilization, reduced CPU utilization
  - Notes: greedy-path logsumexp skip reverted after zero-token runs; needs different approach.

### Tooling & measurement requirements

- [ ] Keep benchmark harness repeatable (same prompt, tokens, temperature)
- [ ] After each change: run tests/imports + one benchmark
- [ ] Report deltas: TPS, CPU%, GPU%, wired memory oscillation, top 20 cProfile
- [ ] Confirm “no numpy in hot path” (grep or profile evidence)

## Cross‑Cutting Tasks

- [ ] Update imports after each move
- [ ] Run smoke import test after each batch
- [ ] Run `python -m pytest -q --maxfail=3 --disable-warnings || true`
- [ ] Update docs if module boundaries change

## Observations / Open Questions

- [ ] Notes:
