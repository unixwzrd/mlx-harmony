# Source File Map

**Created**: 2026-01-15
**Updated**: 2026-01-16

This document lists the modules under [src/mlx_harmony](../src/mlx_harmony) and what each one does. It reflects the new subpackage layout introduced to keep the project organized.

## Grouping Overview

### Entry Points (Root Wrappers)

These live at the package root to preserve CLI entrypoints while the implementation lives under `cli/`.

- [chat.py](../src/mlx_harmony/chat.py): CLI chat entrypoint wrapper.
- [generate.py](../src/mlx_harmony/generate.py): One-shot generation CLI wrapper.
- [server.py](../src/mlx_harmony/server.py): FastAPI server wrapper.
- [convert_dialogue.py](../src/mlx_harmony/convert_dialogue.py): Dialogue conversion CLI wrapper.
- [chat_migration.py](../src/mlx_harmony/chat_migration.py): Chat migration CLI wrapper.
- [check_mic.py](../src/mlx_harmony/check_mic.py): Mic checker wrapper (used by `hotmic`).

### CLI (`cli/`)

- [cli/chat.py](../src/mlx_harmony/cli/chat.py): Main interactive chat loop.
- [cli/generate.py](../src/mlx_harmony/cli/generate.py): One-shot generation CLI.
- [cli/server.py](../src/mlx_harmony/cli/server.py): FastAPI /v1/chat/completions server.
- [cli/cli_args.py](../src/mlx_harmony/cli/cli_args.py): CLI argument parser setup.
- [cli/chat_commands.py](../src/mlx_harmony/cli/chat_commands.py): Slash commands, hyperparameter overrides.
- [cli/convert_dialogue.py](../src/mlx_harmony/cli/convert_dialogue.py): Dialogue text to JSON conversion.
- [cli/chat_migration.py](../src/mlx_harmony/cli/chat_migration.py): Chat schema migration CLI logic.

### Conversation (`conversation/`)

- [conversation/conversation_history.py](../src/mlx_harmony/conversation/conversation_history.py): IDs, timestamps, log paths, chat history helpers.
- [conversation/conversation_io.py](../src/mlx_harmony/conversation/conversation_io.py): Read user input and load/save chats.
- [conversation/conversation_migration.py](../src/mlx_harmony/conversation/conversation_migration.py): Schema migration and validation.

### Generation (`generation/`)

- [generation/generator.py](../src/mlx_harmony/generation/generator.py): TokenGenerator implementation.
- [generation/generate_standalone.py](../src/mlx_harmony/generation/generate_standalone.py): Core generation loop.
- [generation/generation_stream.py](../src/mlx_harmony/generation/generation_stream.py): Streaming token decode for chat.
- [generation/sampling.py](../src/mlx_harmony/generation/sampling.py): Logits processors and sampling utilities.
- [generation/cache.py](../src/mlx_harmony/generation/cache.py): KV cache + rotating cache.
- [generation/prompt_cache.py](../src/mlx_harmony/generation/prompt_cache.py): Prompt token caching for truncation speedups.

### Harmony (`harmony/`)

- [harmony/harmony_parser.py](../src/mlx_harmony/harmony/harmony_parser.py): Harmony message parsing and routing.
- [harmony/prompt_builder.py](../src/mlx_harmony/harmony/prompt_builder.py): Prompt rendering, truncation, debug output.
- [harmony/tool_calls.py](../src/mlx_harmony/harmony/tool_calls.py): Tool call extraction and execution wiring.

### Voice (`voice/`)

- [voice/voice_moshi.py](../src/mlx_harmony/voice/voice_moshi.py): Moshi STT/TTS integration.
- [voice/check_mic.py](../src/mlx_harmony/voice/check_mic.py): Mic permission test (used by `hotmic`).

### Runtime (`runtime/`)

- [runtime/loader.py](../src/mlx_harmony/runtime/loader.py): Model loading wrappers and metadata helpers.
- [runtime/load_optimized.py](../src/mlx_harmony/runtime/load_optimized.py): Optimized load paths and tokenizer hookup.
- [runtime/tokenizer_native.py](../src/mlx_harmony/runtime/tokenizer_native.py): Native tokenizer + chat template handling.
- [runtime/hyperparameters.py](../src/mlx_harmony/runtime/hyperparameters.py): Resolve param precedence (CLI/config/default).

### Core Utilities

- [config.py](../src/mlx_harmony/config.py): PromptConfig model + config loading.
- [logging.py](../src/mlx_harmony/logging.py): Project logging helpers.
- [render_output.py](../src/mlx_harmony/render_output.py): Rich output rendering.
- [__init__.py](../src/mlx_harmony/__init__.py): Package exports (TokenGenerator).

### Tools

- [tools/__init__.py](../src/mlx_harmony/tools/__init__.py): Tool configs, parsing, and stub executors.

### Model Architecture (Standalone, No mlx-lm)

- [models/__init__.py](../src/mlx_harmony/models/__init__.py): Model exports.
- [models/base.py](../src/mlx_harmony/models/base.py): Base args and attention helpers.
- [models/gpt_oss.py](../src/mlx_harmony/models/gpt_oss.py): GPT-OSS architecture implementation.
- [models/rope_utils.py](../src/mlx_harmony/models/rope_utils.py): RoPE helpers.
- [models/switch_layers.py](../src/mlx_harmony/models/switch_layers.py): Layer switch utilities.

## File Index and Naming Updates

| File | Purpose | Former name |
| --- | --- | --- |
| [cli/chat.py](../src/mlx_harmony/cli/chat.py) | Chat CLI entrypoint and main loop wiring. | — |
| [cli/chat_commands.py](../src/mlx_harmony/cli/chat_commands.py) | Out-of-band chat commands and hyperparameter updates. | chat_utils.py |
| [cli/cli_args.py](../src/mlx_harmony/cli/cli_args.py) | Chat CLI argument parser definition. | chat_cli.py |
| [cli/chat_migration.py](../src/mlx_harmony/cli/chat_migration.py) | Chat migration CLI logic. | chat_migration.py |
| [conversation/conversation_history.py](../src/mlx_harmony/conversation/conversation_history.py) | IDs, timestamps, path resolution, debug logging. | chat_history.py |
| [conversation/conversation_io.py](../src/mlx_harmony/conversation/conversation_io.py) | Read user input and load/save chat sessions. | chat_io.py |
| [conversation/conversation_migration.py](../src/mlx_harmony/conversation/conversation_migration.py) | Chat schema migration and validation. | chat_migration.py |
| [generation/generation_stream.py](../src/mlx_harmony/generation/generation_stream.py) | Streamed token decoding for chat. | chat_generation.py |
| [harmony/harmony_parser.py](../src/mlx_harmony/harmony/harmony_parser.py) | Harmony parsing and final/analysis routing. | chat_harmony.py |
| [harmony/prompt_builder.py](../src/mlx_harmony/harmony/prompt_builder.py) | Build prompts, debug logs, and truncation. | chat_prompt.py |
| [harmony/tool_calls.py](../src/mlx_harmony/harmony/tool_calls.py) | Tool call extraction + execution wiring. | chat_tools.py |
| [voice/voice_moshi.py](../src/mlx_harmony/voice/voice_moshi.py) | Moshi STT/TTS integration. | — |
| [voice/check_mic.py](../src/mlx_harmony/voice/check_mic.py) | Mic permission checker. | — |

[Back to README](../README.md)
