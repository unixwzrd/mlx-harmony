# Notes

**Created**: 2026-01-28
**Updated**: 2026-01-28

## Memory Locking (mlock) and Wired Limits

- The 10% margin is used to set the wired memory limit when `mlock=true`, based on the estimated model size.
- The limit can be capped by Metal’s `max_recommended_working_set_size`, so the full margin may not apply if it exceeds that cap.
- When parameters are wired, they are pinned in memory to avoid paging; the weights are still read during inference, but not modified.
- Inference still performs vector operations on activations and temporary buffers, which are *not* the same as model weights and can increase memory pressure.
- If margin feels too high for a given device, we can consider making it configurable or adapt it based on headroom.

## Open Questions for Documentation

- Should the margin be configurable in `PromptConfig`?
- How should we message OOM behavior when `mlock` is enabled vs disabled?
- Should we expose a diagnostic that reports wired limit, estimated size, and actual wired usage?
