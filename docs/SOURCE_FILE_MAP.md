# Source File Map

**Created**: 2026-01-21
**Updated**: 2026-02-09

This document lists the Python modules under [src/mlx_harmony](../src/mlx_harmony) and describes their current roles.

## Entry Points (Root Wrappers)

- [chat.py](../src/mlx_harmony/chat.py): CLI chat entrypoint wrapper.
- [generate.py](../src/mlx_harmony/generate.py): One-shot generation entrypoint.
- [server.py](../src/mlx_harmony/server.py): FastAPI server wrapper.
- [convert_dialogue.py](../src/mlx_harmony/convert_dialogue.py): Dialogue conversion entrypoint.

## Chat Core (Root Modules)

- [chat_bootstrap.py](../src/mlx_harmony/chat_bootstrap.py): Chat startup/bootstrap wiring.
- [chat_frontend.py](../src/mlx_harmony/chat_frontend.py): Shared CLI front-end loop (input, commands, rendering).
- [chat_backend.py](../src/mlx_harmony/chat_backend.py): Front-end backend adapters (local vs HTTP).
- [chat_commands.py](../src/mlx_harmony/chat_commands.py): Out-of-band command parsing and help/models rendering.
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
- [generation/client.py](../src/mlx_harmony/generation/client.py): GenerationClient interface + Local/Server adapters.

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
- [api_client.py](../src/mlx_harmony/api_client.py): HTTP client for `/v1/chat/completions` + request logging.
- [__init__.py](../src/mlx_harmony/__init__.py): Package exports.

## Documentation Notes

- [NOTES.md](./NOTES.md): Working notes for documentation topics to be consolidated into guides.
- [NEXT_SPRINT_CHECKLIST.md](./NEXT_SPRINT_CHECKLIST.md): Cross-area checklist for upcoming sprint work.
- [SERVER_GUIDE.md](./SERVER_GUIDE.md): Server startup and API overview.

## Scripts

- [bench_run.sh](../scripts/bench_run.sh): Benchmark harness (dataset run + vm_stat + merge).
- [filter-vm_stat.py](../scripts/filter-vm_stat.py): vm_stat → TSV/JSON conversion helper.
- [merge_timing_metrics.py](../scripts/merge_timing_metrics.py): Merge timing and vm_stat TSVs.
- [profile_module.py](../scripts/profile_module.py): Generic cProfile wrapper for module/script execution.
- [profile_cli.sh](../scripts/profile_cli.sh): CLI profiling runner (dataset-driven via STDIN).
- [profile_server.sh](../scripts/profile_server.sh): Server + client profiling runner for HTTP workflow.
- [clean_logs.sh](../scripts/clean_logs.sh): Cleans staged log artifacts before component runs.
- [preserve_logs.sh](../scripts/preserve_logs.sh): Moves staged logs into run bundle component directories.
- [process_stats.sh](../scripts/process_stats.sh): Shared stats/plot/profile-artifact processor for component runs.
- [clean_run_artifacts.sh](../scripts/clean_run_artifacts.sh): Manual cleanup utility for shared log artifacts.
- [dataset_run_common.sh](../scripts/dataset_run_common.sh): Shared helpers for dataset runs (metrics, vm_stat, plots).
- [run_dataset_harness.sh](../scripts/run_dataset_harness.sh): Unified dataset runner for CLI/server workflows.
- [module_dep_graph.py](../scripts/module_dep_graph.py): Generate module dependency graphs (DOT/TSV).
- [TPSvsWiredMemory.py](../scripts/TPSvsWiredMemory.py): Plot tokens-per-second versus wired memory from merged TSV.

## Consolidation Notes

- [api_client.py](../src/mlx_harmony/api_client.py) and [generation/client.py](../src/mlx_harmony/generation/client.py) overlap in HTTP adapter responsibilities.
- [server.py](../src/mlx_harmony/server.py) uses a simplified generation path compared to the CLI stack; keep converging to shared adapters.

[Back to README](../README.md)
