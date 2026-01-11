# MLX Harmony Refactoring Plan

**Created**: 2026-01-11
**Updated**: 2026-01-11

## Goals

- [ ] Reduce duplication and clarify module boundaries while preserving current behavior.
- [ ] Align implementation with project standards (type hints, Pydantic for config/validation, logging module usage).
- [ ] Improve test coverage for critical paths, edge cases, and CLI/server flows.
- [ ] Keep changes incremental and reversible with small, reviewable batches.
- [ ] Prioritize GPT-OSS correctness first while keeping extension points for non-GPT-OSS models.

## Decisions From Review

- [x] Use MLX-native tokenization and inference for all models.
- [x] Use Harmony prompt construction and decoding for GPT-OSS only, via `openai_harmony`.
- [x] Keep non-GPT-OSS prompt construction in the appropriate model-specific path.
- [x] Plan tool-calling as a separate module after refactor.
- [x] Use Python's logging facilities with structured JSON chat logs for retrieval and record keeping.

## Review Findings

- [x] **Architecture is CLI-centric and monolithic**: [chat.py](../src/mlx_harmony/chat.py) mixes CLI parsing, rendering, conversation persistence, tool handling, and generation orchestration.
- [x] **Repeated Harmony prompt construction**: [generator.py](../src/mlx_harmony/generator.py) duplicates message-to-Harmony conversion in `_harmony_messages_to_prompt` and `_harmony_messages_to_token_ids`.
- [x] **Config and tool schemas are dataclass-based**: [config.py](../src/mlx_harmony/config.py) and [tools/__init__.py](../src/mlx_harmony/tools/__init__.py) use dataclasses; project standards call for Pydantic.
- [ ] **Inline imports and print-based logging**: [loader.py](../src/mlx_harmony/loader.py), [load_optimized.py](../src/mlx_harmony/load_optimized.py), [server.py](../src/mlx_harmony/server.py), [generate.py](../src/mlx_harmony/generate.py) import in function scope and rely on `print` instead of a logging module.
- [ ] **Server singleton is not concurrency-safe**: [server.py](../src/mlx_harmony/server.py) caches a global generator across requests without locking, which can be unsafe under concurrent load.
- [ ] **Sampling and cache modules lack full type hints**: [sampling.py](../src/mlx_harmony/sampling.py) and [cache.py](../src/mlx_harmony/cache.py) are missing type hints on many functions/attributes despite project standards.
- [ ] **Stop token handling is single-token only**: [generate_standalone.py](../src/mlx_harmony/generate_standalone.py) notes the lack of multi-token stop sequences (TODO), which can cause incorrect termination behavior for some prompts.
- [ ] **Error handling is inconsistent**: Some code swallows exceptions or emits warnings without structured logging, leading to hard-to-debug failures.

## Refactor Plan (Incremental Phases)

### Status Notes

- [x] Added a logging module and migrated [loader.py](../src/mlx_harmony/loader.py) and [load_optimized.py](../src/mlx_harmony/load_optimized.py) to use it.
- [ ] Remaining modules still use `print` and/or inline imports: [server.py](../src/mlx_harmony/server.py), [generate.py](../src/mlx_harmony/generate.py).

### Phase 0: Baseline and Safety

- [ ] Capture baseline behavior and performance notes (current CLI and server output paths).
- [ ] Add missing tests that do not require model downloads (see Test Plan below).
- [ ] Document any breaking changes or compatibility assumptions.

### Phase 1: Configuration and Validation

- [x] Replace `PromptConfig` dataclass with a Pydantic model in [config.py](../src/mlx_harmony/config.py).
- [x] Add explicit validation for numeric ranges (sampling parameters, truncation limits).
- [ ] Update any call sites and tests (e.g., [tests/test_config.py](../tests/test_config.py), [tests/test_generator.py](../tests/test_generator.py)).
- [x] Introduce Pydantic models for tool definitions and tool calls in [tools/__init__.py](../src/mlx_harmony/tools/__init__.py).

### Phase 2: Logging and Error Handling

- [ ] Introduce a project logging module (if not already present) and replace `print` calls with logger usage.
- [ ] Standardize error handling to raise clear exceptions for invalid inputs and critical failures.
- [ ] Ensure warnings are actionable and include remediation guidance.

### Phase 3: Generator and Prompt Flow Consolidation

- [x] Extract shared Harmony message-building logic into a single helper in [generator.py](../src/mlx_harmony/generator.py).
- [ ] Consolidate default resolution for sampling parameters (reduce duplicate “config vs CLI” logic).
- [ ] Ensure Harmony and non-Harmony token decoding paths are explicit and tested.

### Phase 4: Chat CLI Decomposition

- [x] Split [chat.py](../src/mlx_harmony/chat.py) into focused helpers or modules.
- [ ] Prompt rendering
- [ ] Conversation persistence
- [ ] CLI parsing
- [ ] Tool execution loop
- [ ] Keep the CLI surface stable; refactor only internal structure.
- [ ] Add unit tests for parsing, directory normalization, and hyperparameter updates.

### Phase 5: Server Hardening

- [ ] Make generator caching concurrency-safe (lock or per-request generator).
- [ ] Allow profile path configuration rather than hard-coding `configs/profiles.example.json` in [server.py](../src/mlx_harmony/server.py).
- [ ] Expand server tests to cover streaming and non-streaming responses.

### Phase 6: Performance and Memory

- [ ] Review `mlock` and cache prewarm flows for consistent behavior across [loader.py](../src/mlx_harmony/loader.py) and [load_optimized.py](../src/mlx_harmony/load_optimized.py).
- [ ] Confirm memory lock paths are safe when model size can’t be estimated.
- [ ] Avoid repeated cache clearing if it harms throughput.

## Test Plan

### Current Tests

- [ ] Confirm tests pass without a model: `pytest -m "not slow and not requires_model"` (see [tests/README.md](../tests/README.md)).
- [ ] Confirm config and tools tests still pass after Pydantic migration: [tests/test_config.py](../tests/test_config.py), [tests/test_tools.py](../tests/test_tools.py).

Note: `python -m pytest -q --maxfail=3 --disable-warnings` currently aborts during collection with an import error in [generate_standalone.py](../src/mlx_harmony/generate_standalone.py) (Abort trap: 6). Track and resolve this before relying on automated test runs.

### New/Expanded Tests

- [ ] **Prompt config validation**: boundary checks for sampling fields and truncation limits.
- [ ] **Prompt rendering**: Harmony vs non-Harmony path in [generator.py](../src/mlx_harmony/generator.py).
- [ ] **Chat history round-trip**: ensure timestamps and hyperparameters are preserved in [chat.py](../src/mlx_harmony/chat.py).
- [ ] **Server behavior**: test streaming vs non-streaming responses in [server.py](../src/mlx_harmony/server.py).

### Edge Cases to Validate

- [ ] Empty prompt, empty messages, and mixed roles.
- [ ] Multi-turn tool call sequences with no final response.
- [ ] Stop token sequences that are multi-token (verify correct stopping behavior).
- [ ] Prompt config placeholders in system/developer messages with mixed casing.
- [ ] Max context window handling: detect from model config when available and allow explicit override to avoid context overflow.

## Documentation Updates

- [ ] Update [README.md](../README.md) and [docs/FEATURES_FROM_MLX.md](../docs/FEATURES_FROM_MLX.md) if behavior changes or new defaults are introduced.
- [ ] Add a brief architecture overview to [docs/ROADMAP.md](../docs/ROADMAP.md) if module boundaries change.
- [ ] Keep [tests/README.md](../tests/README.md) updated with any new markers or test requirements.

## Open Questions

- [ ] Should Harmony decoding always use `HarmonyEncoding` rather than the native tokenizer for GPT-OSS responses?
- [ ] Do we want tool execution to remain stubbed, or should it move into a dedicated implementation module?
- [ ] What is the intended logging interface (standard library logging vs a project wrapper)?
