# Source File Map

**Created**: 2026-01-21
**Updated**: 2026-01-23

This document lists the Python modules under [src/mlx_harmony](../src/mlx_harmony) and describes their current roles.

## Entry Points (Root Wrappers)

- [chat.py](../src/mlx_harmony/chat.py): CLI chat entrypoint wrapper.
- [generate.py](../src/mlx_harmony/generate.py): One-shot generation entrypoint.
- [server.py](../src/mlx_harmony/server.py): FastAPI server wrapper.
- [convert_dialogue.py](../src/mlx_harmony/convert_dialogue.py): Dialogue conversion entrypoint.

## Chat Core (Root Modules)

- [chat_bootstrap.py](../src/mlx_harmony/chat_bootstrap.py): Chat startup/bootstrap wiring.
- [chat_controller.py](../src/mlx_harmony/chat_controller.py): Facade re-export for chat turn + retry/adapter helpers.
- [chat_cli.py](../src/mlx_harmony/chat_cli.py): CLI argument parsing for chat.
- [chat_generation.py](../src/mlx_harmony/chat_generation.py): Streaming generation orchestration (no per-token Harmony parsing).
- [chat_harmony.py](../src/mlx_harmony/chat_harmony.py): Harmony parsing and message extraction.
- [chat_history.py](../src/mlx_harmony/chat_history.py): Debug logging and chat metadata helpers.
- [chat_io.py](../src/mlx_harmony/chat_io.py): Read user input and load/save chat sessions.
- [chat_migration.py](../src/mlx_harmony/chat_migration.py): Chat schema migration logic.
- [chat_prompt.py](../src/mlx_harmony/chat_prompt.py): Prompt building and truncation helpers.
- [chat_render.py](../src/mlx_harmony/chat_render.py): Rendering of assistant/thinking output.
- [chat_utils.py](../src/mlx_harmony/chat_utils.py): Command parsing and hyperparameter resolution.
- [chat_adapters.py](../src/mlx_harmony/chat_adapters.py): Model adapter protocol + Harmony/native adapters.
- [chat_attempt.py](../src/mlx_harmony/chat_attempt.py): Single-attempt generation + artifacts.
- [chat_retry.py](../src/mlx_harmony/chat_retry.py): Retry policy + recovery prompt handling.
- [chat_turn.py](../src/mlx_harmony/chat_turn.py): Turn orchestration (prompt → generate → parse → retry).
- [chat_types.py](../src/mlx_harmony/chat_types.py): Shared chat dataclasses for controller stack.

## Generation

- [generator.py](../src/mlx_harmony/generator.py): TokenGenerator implementation.
- [generate_standalone.py](../src/mlx_harmony/generate_standalone.py): Core generation loop.
- [sampling.py](../src/mlx_harmony/sampling.py): Logits processors and sampling utilities.
- [repetition_tokens.py](../src/mlx_harmony/repetition_tokens.py): Token-level repetition detection utilities.
- [prompt_cache.py](../src/mlx_harmony/prompt_cache.py): Prompt token caching helpers.
- [cache.py](../src/mlx_harmony/cache.py): KV cache and rotating cache implementation.
- [generation/backend.py](../src/mlx_harmony/generation/backend.py): Backend protocol definitions.
- [generation/backends/__init__.py](../src/mlx_harmony/generation/backends/__init__.py): Backend exports.
- [generation/backends/gpt_oss_backend.py](../src/mlx_harmony/generation/backends/gpt_oss_backend.py): GPT‑OSS prompt backend.
- [generation/backends/native_backend.py](../src/mlx_harmony/generation/backends/native_backend.py): Native prompt backend.

## Prompt Rendering

- [prompts/base.py](../src/mlx_harmony/prompts/base.py): Prompt renderer base protocol.
- [prompts/harmony.py](../src/mlx_harmony/prompts/harmony.py): Harmony prompt rendering.
- [prompts/native.py](../src/mlx_harmony/prompts/native.py): Native prompt rendering.

## Runtime

- [loader.py](../src/mlx_harmony/loader.py): Model loading helpers.
- [load_optimized.py](../src/mlx_harmony/load_optimized.py): Optimized loading helpers.
- [tokenizer_native.py](../src/mlx_harmony/tokenizer_native.py): Native tokenizer implementation.
- [runtime/context.py](../src/mlx_harmony/runtime/context.py): RunContext dataclass.
- [runtime/metrics.py](../src/mlx_harmony/runtime/metrics.py): Timing metrics helpers.
- [runtime/sampler.py](../src/mlx_harmony/runtime/sampler.py): Sampling protocol definitions.
- [runtime/tokenizer.py](../src/mlx_harmony/runtime/tokenizer.py): Tokenizer protocol definitions.
- [hyperparameters.py](../src/mlx_harmony/hyperparameters.py): Hyperparameter precedence helpers.

## Config

- [config.py](../src/mlx_harmony/config.py): Prompt config schema + loading helpers.

## Tools

- [tools/__init__.py](../src/mlx_harmony/tools/__init__.py): Tool exports and parsing helpers.
- [tools/registry.py](../src/mlx_harmony/tools/registry.py): Tool registry selection.
- [tools/runner.py](../src/mlx_harmony/tools/runner.py): Tool execution runner.
- [tools/types.py](../src/mlx_harmony/tools/types.py): Tool schema/types.

## Models

- [models/__init__.py](../src/mlx_harmony/models/__init__.py): Model exports.
- [models/base.py](../src/mlx_harmony/models/base.py): Base args and attention helpers.
- [models/gpt_oss.py](../src/mlx_harmony/models/gpt_oss.py): GPT‑OSS architecture.
- [models/rope_utils.py](../src/mlx_harmony/models/rope_utils.py): RoPE helpers.
- [models/switch_layers.py](../src/mlx_harmony/models/switch_layers.py): Switch layer utilities.

## Utilities

- [logging.py](../src/mlx_harmony/logging.py): Logging helpers.
- [__init__.py](../src/mlx_harmony/__init__.py): Package exports.

[Back to README](../README.md)
