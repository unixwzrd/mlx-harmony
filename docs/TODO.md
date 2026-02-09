# TODO Checklist

**Created**: 2026-01-09
**Updated**: 2026-02-09

## Purpose

Hold untriaged work and quick-capture items only. This file is an inbox, not the active sprint plan.

## Document Ownership

- Active execution tasks: [NEXT_SPRINT_CHECKLIST.md](./NEXT_SPRINT_CHECKLIST.md)
- Refactor/convergence tasks: [REFACTOR_CHECKLIST.md](./REFACTOR_CHECKLIST.md)
- Long-horizon planning: [ROADMAP.md](./ROADMAP.md)

## Inbox (Untriaged)

- [ ] Add benchmark summary artifact (`summary.json`) per run for easy historical diffing.
- [ ] Add a single command/script to regenerate broad/unfiltered graph outputs from existing `*.stats`.
- [ ] Add docs section for profile graph generation (`compact`, `broad`, `unfiltered`) in [scripts/README.md](../scripts/README.md).
- [ ] Evaluate pruning obsolete benchmark wrapper scripts after workflow stabilizes.
- [ ] Add architecture boundary diagram (frontend, adapter, backend-core) in docs.
- [ ] Add module naming cleanup pass plan (reduce overloaded `chat_*` naming where scope differs).

## Promotion Rule

- Move an item to [NEXT_SPRINT_CHECKLIST.md](./NEXT_SPRINT_CHECKLIST.md) when it is committed for the next 1-2 day cycle.
- Move an item to [REFACTOR_CHECKLIST.md](./REFACTOR_CHECKLIST.md) when it is primarily structural/refactor work.
- Move an item to [ROADMAP.md](./ROADMAP.md) when it is not intended for immediate execution.

[← Back to README](../README.md)
