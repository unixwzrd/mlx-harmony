# Source File Map

**Created**: 2026-01-15
**Updated**: 2026-01-19

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

- [cli/chat_commands.py](../src/mlx_harmony/cli/chat_commands.py): Slash commands, hyperparameter overrides.
- [cli/cli_args.py](../src/mlx_harmony/cli/cli_args.py): CLI argument parser setup.
- [cli/convert_dialogue.py](../src/mlx_harmony/cli/convert_dialogue.py): Dialogue text to JSON conversion.
- [cli/chat_migration.py](../src/mlx_harmony/cli/chat_migration.py): Chat schema migration CLI logic.
- [cli/generate.py](../src/mlx_harmony/cli/generate.py): One-shot generation CLI.
- [cli/server.py](../src/mlx_harmony/cli/server.py): FastAPI /v1/chat/completions server.

### Chat (`chat/`)

- [chat/main.py](../src/mlx_harmony/chat/main.py): Main interactive chat loop (CLI orchestration).
- [chat/input.py](../src/mlx_harmony/chat/input.py): CLI input + command handling.
- [chat/prompt.py](../src/mlx_harmony/chat/prompt.py): Prompt assembly helpers (truncate + render).
- [chat/session.py](../src/mlx_harmony/chat/session.py): Chat session initialization and startup flow.
- [chat/turn.py](../src/mlx_harmony/chat/turn.py): Generation loop + tool calls + TTS handling.
- [chat/voice.py](../src/mlx_harmony/chat/voice.py): Moshi init + voice input helpers.

### Conversation (`conversation/`)

- [conversation/conversation_history.py](../src/mlx_harmony/conversation/conversation_history.py): Re-exports conversation helpers for compatibility.
- [conversation/ids.py](../src/mlx_harmony/conversation/ids.py): Message/chat IDs and timestamp helpers.
- [conversation/paths.py](../src/mlx_harmony/conversation/paths.py): Chat/log path normalization and resolution.
- [conversation/metadata.py](../src/mlx_harmony/conversation/metadata.py): Chat metadata merge and session restore helpers.
- [conversation/debug.py](../src/mlx_harmony/conversation/debug.py): Debug log writers for prompts, tokens, and metrics.
- [conversation/resume.py](../src/mlx_harmony/conversation/resume.py): Resume display helpers for prior chats.
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

### Speech (`speech/`)

- [speech/tts_stream.py](../src/mlx_harmony/speech/tts_stream.py): TTS streaming controller (decoupled from CLI).
- [speech/moshi/config.py](../src/mlx_harmony/speech/moshi/config.py): Moshi config parsing + validation.
- [speech/moshi/loader.py](../src/mlx_harmony/speech/moshi/loader.py): Moshi speech facade (re-exports STT/TTS helpers).
- [speech/moshi/shared.py](../src/mlx_harmony/speech/moshi/shared.py): Shared Moshi config + import helpers.
- [speech/moshi/stt_runtime.py](../src/mlx_harmony/speech/moshi/stt_runtime.py): Moshi STT runtime loop + model init.
- [speech/moshi/tts_runtime.py](../src/mlx_harmony/speech/moshi/tts_runtime.py): Moshi TTS runtime + chunking.
- [speech/moshi/stt.py](../src/mlx_harmony/speech/moshi/stt.py): Moshi STT adapter.
- [speech/moshi/tts.py](../src/mlx_harmony/speech/moshi/tts.py): Moshi TTS adapter.

### Voice (`voice/`)

- [voice/voice_moshi.py](../src/mlx_harmony/voice/voice_moshi.py): Temporary compatibility shim for Moshi speech (pending refactor).
- [tools/hotmic.py](../src/mlx_harmony/tools/hotmic.py): Mic permission test (used by `hotmic`).

### Runtime (`runtime/`)

- [runtime/loader.py](../src/mlx_harmony/runtime/loader.py): Model loading wrappers and metadata helpers.
- [runtime/load_optimized.py](../src/mlx_harmony/runtime/load_optimized.py): Optimized load paths and tokenizer hookup.
- [runtime/tokenizer_bpe.py](../src/mlx_harmony/runtime/tokenizer_bpe.py): ByteLevel BPE tokenizer implementation.
- [runtime/tokenizer_loader.py](../src/mlx_harmony/runtime/tokenizer_loader.py): Load tokenizer.json into ByteLevel BPE tokenizer.
- [runtime/tokenizer_streaming.py](../src/mlx_harmony/runtime/tokenizer_streaming.py): Streaming detokenizer helpers.
- [runtime/hyperparameters.py](../src/mlx_harmony/runtime/hyperparameters.py): Resolve param precedence (CLI/config/default).

### Config (`config/`)

- [config/session_schema.py](../src/mlx_harmony/config/session_schema.py): SessionConfig schema (prompt + session settings).
- [config/moshi_schema.py](../src/mlx_harmony/config/moshi_schema.py): MoshiConfig schema.
- [config/schema.py](../src/mlx_harmony/config/schema.py): Schema re-exports (SessionConfig, MoshiConfig).
- [config/loader.py](../src/mlx_harmony/config/loader.py): JSON loading + placeholder helpers.

### Core Utilities
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
| [chat/main.py](../src/mlx_harmony/chat/main.py) | Chat CLI entrypoint and main loop wiring. | cli/chat.py |
| [chat/input.py](../src/mlx_harmony/chat/input.py) | Input handling + command application. | cli/chat_input.py |
| [chat/prompt.py](../src/mlx_harmony/chat/prompt.py) | Prompt prep helpers. | cli/chat_prompt.py |
| [chat/session.py](../src/mlx_harmony/chat/session.py) | Chat session initialization and startup flow. | — |
| [chat/turn.py](../src/mlx_harmony/chat/turn.py) | Generation loop, tool calls, TTS flow. | cli/chat_turn.py |
| [chat/voice.py](../src/mlx_harmony/chat/voice.py) | Moshi init + voice input helpers. | cli/chat_voice.py |
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
| [speech/moshi/loader.py](../src/mlx_harmony/speech/moshi/loader.py) | Moshi speech facade (re-exports STT/TTS helpers). | voice/voice_moshi.py |
| [speech/moshi/shared.py](../src/mlx_harmony/speech/moshi/shared.py) | Shared Moshi config + import helpers. | — |
| [speech/moshi/stt_runtime.py](../src/mlx_harmony/speech/moshi/stt_runtime.py) | Moshi STT runtime loop + model init. | voice/voice_moshi.py |
| [speech/moshi/tts_runtime.py](../src/mlx_harmony/speech/moshi/tts_runtime.py) | Moshi TTS runtime + chunking. | voice/voice_moshi.py |
| [speech/moshi/stt.py](../src/mlx_harmony/speech/moshi/stt.py) | Moshi STT adapter. | voice/voice_moshi.py |
| [speech/moshi/tts.py](../src/mlx_harmony/speech/moshi/tts.py) | Moshi TTS adapter. | voice/voice_moshi.py |
| [config/session_schema.py](../src/mlx_harmony/config/session_schema.py) | SessionConfig schema (prompt + session settings). | config.py |
| [config/moshi_schema.py](../src/mlx_harmony/config/moshi_schema.py) | MoshiConfig schema. | config.py |
| [config/schema.py](../src/mlx_harmony/config/schema.py) | Schema re-exports (SessionConfig, MoshiConfig). | config.py |
| [config/loader.py](../src/mlx_harmony/config/loader.py) | Config loading + placeholder utilities. | config.py |
| [speech/moshi/config.py](../src/mlx_harmony/speech/moshi/config.py) | Moshi config parsing + validation. | cli/moshi_config.py |
| [speech/tts_stream.py](../src/mlx_harmony/speech/tts_stream.py) | TTS streaming controller (decoupled). | cli/tts_stream.py |
| [voice/voice_moshi.py](../src/mlx_harmony/voice/voice_moshi.py) | Compatibility shim (temporary). | — |
| [tools/hotmic.py](../src/mlx_harmony/tools/hotmic.py) | Mic permission checker. | — |

[Back to README](../README.md)
