# Performance Plan

**Created**: 2026-01-21
**Updated**: 2026-01-21

**Location**: `docs/PERFORMANCE_PLAN.md` (links are relative to this file)

## Goals

- Improve tokens per second (TPS) consistency across longer contexts.
- Reduce KV prefill overhead and avoid repeated full-prefill work.
- Stabilize memory usage (avoid oscillating allocation/wiring).
- Keep changes incremental and testable (10-iteration deterministic runs).

## Findings From Latest Profiles

- Hotspots:
  - `sampling.py:286(processor)` dominates runtime.
  - `generate_standalone.py:_prefill_kv_cache` is the next largest cost.
- TPS falls as `prompt_tokens` grows, matching the cost growth of repetition penalty and KV prefill.
- Memory “flap” correlates with frequent cache clearing and full prefills.

## Recommendations (Priority Order)

### 1) Cap Repetition Penalty Window

- **Why**: The repetition penalty cost grows with token history size; it becomes O(n) per token.
- **Change**: Add a hard cap (e.g., 768) to the effective repetition window even if `repetition_context_size` is larger.
- **Current**: Deterministic config uses `repetition_context_size=256`, so this cap should be a no-op unless that value grows.
- **Config**: Keep `repetition_context_size` as user-controlled, but clamp it to the cap.
- **File**: [src/mlx_harmony/sampling.py](../src/mlx_harmony/sampling.py)

### 2) Reduce or Disable KV Cache Clearing

- **Why**: Clearing cache each turn forces a full prefill and drives both time and memory oscillation.
- **Change**: Default `clear_cache=false` or increase `clear_cache_interval` to allow reuse.
- **Config**: [configs/prompt-config.deterministic.json](../configs/prompt-config.deterministic.json)

### 3) Confirm Prefill Reuse

- **Why**: Prefill should only process new tokens, not the full prompt every turn.
- **Change**: Log `prefill_start_offset` and computed `prefill_tokens` in `_prefill_kv_cache`.
- **File**: [src/mlx_harmony/generate_standalone.py](../src/mlx_harmony/generate_standalone.py)

### 4) Tune Prefill Chunk Size

- **Why**: Large `step_size` can cause big, expensive chunks and GPU command buffer spikes.
- **Change**: Try `step_size=512` or `1024` instead of `2048`, measure impact.
- **File**: [src/mlx_harmony/generate_standalone.py](../src/mlx_harmony/generate_standalone.py)

## Test Protocol

- Use deterministic test config and run **10 iterations**.
- Capture:
  - `stats/timings-debug.csv`
  - `stats/profile.stats.txt`
  - `stats/profile.dot`
- Compare:
  - TPS trend vs. `prompt_tokens`
  - Total time in `processor` and `_prefill_kv_cache`
  - Memory stability (no oscillation/wiring thrash)

## Notes

- Apply changes one at a time to keep regression risk low.
- If a change degrades tests or import stability, revert that batch.
