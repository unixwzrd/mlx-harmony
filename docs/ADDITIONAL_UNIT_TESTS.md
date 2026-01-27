# Additional Unit Tests

**Created**: 2026-01-25
**Updated**: 2026-01-25

Small, additive checklist for unit-test ideas. Keep this lean to avoid slowing current refactor work.

## Prompt Truncation

- [ ] `max_context_tokens_margin` reduces effective context size without dropping system/developer turns.
- [ ] Margin >= max_context_tokens logs warning and is ignored.

## Performance Mode Overrides

- [ ] `performance_mode` enables `perf_max_tokens` override in `TokenGenerator.generate`.
- [ ] `performance_mode` enables `perf_max_kv_size` override in `TokenGenerator.generate`.
- [ ] `performance_mode` enables `perf_max_context_tokens` override in `resolve_max_context_tokens`.

## KV Windowing

- [ ] `make_prompt_cache` returns `RotatingKVCache` when `max_kv_size` is set.
- [ ] `TokenGenerator` cache reuse respects `max_kv_size` changes.

## Retry Behavior

- [ ] `length` retries are skipped when `_looks_complete_response` returns true.
- [ ] `length` retries fire for truncated outputs (no terminal punctuation / ellipsis).

## Token-Based Repetition Detection

- [ ] `_detect_token_repetition` flags repeated token blocks.
- [ ] Structured markdown does not trigger `_detect_token_repetition`.
