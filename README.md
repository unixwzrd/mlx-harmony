# MLX Harmony

**Created**: 2026-01-12
**Updated**: 2026-01-16

![MLX Harmony banner Image](docs/images/MLX_Harmony_Banner.png)

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md) [![Release](https://img.shields.io/github/v/tag/unixwzrd/mlx-harmony?label=release)](https://github.com/unixwzrd/mlx-harmony/releases) [![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)](https://github.com/ml-explore/mlx) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/) [![Harmony](https://img.shields.io/badge/OpenAI--Harmony-GPT--OSS-blue)](https://github.com/openai/openai-harmony) [![Rich](https://img.shields.io/badge/Rich-13.0%2B-FFE066)](https://github.com/Textualize/rich) [![Uvicorn](https://img.shields.io/badge/Uvicorn-0.23%2B-5E4F5F)](https://www.uvicorn.org/)

Run GPT-OSS and other MLX-compatible models through a single, lightweight interface.

`mlx-harmony` combines the best of [`mlx-lm`](https://github.com/ml-explore/mlx-lm) (model architectures) and [`openai-harmony`](https://github.com/openai/openai-harmony) (Harmony formatting) with a standalone implementation that gives you:

- Multi-model **MLX inference** (Llama, Mistral, Qwen, GPT-OSS, …)
- Automatic **Harmony formatting** and tools when using **GPT-OSS** models
- Simple **CLI entrypoints** for chat, one-shot generation, and an **OpenAI-style HTTP API**
- A JSON **prompt config** system with placeholders, examples, and memory optimizations

It’s designed for Apple Silicon first, but will follow MLX wherever it goes.

---

## Features

- **Single generator for many models**  
  Works with any `mlx-lm` model: local paths or Hugging Face repos, including quantized MXFP4/Q8 models.

- **GPT-OSS aware**  
  Automatically detects GPT-OSS models (`openai/gpt-oss-*`) and:
  - Uses Harmony formatting
  - Supports GPT-OSS tools (browser, python, apply_patch) at the chat layer

- **Friendly CLIs**
  - `mlx-harmony-chat` – interactive chat with beautiful markdown rendering (like `glow`/`mdless`)
  - `mlx-harmony-generate` – single-shot text generation
  - `mlx-harmony-server` – OpenAI-style `/v1/chat/completions` endpoint
  - `mlx-harmony-migrate-chat` – migrate chat logs to the latest schema

- **Prompt configs and profiles**
  - JSON configs for system text, developer instructions, placeholders, examples, and sampler settings
  - Profiles that bundle a model path + prompt config for one-flag switching

- **Conversation logging & resume**
  - Save chat sessions as JSON (with timestamps + hyperparameters per turn)
  - Resume with the same or a different model

- **Beautiful markdown rendering**
  - Assistant responses are automatically formatted with rich markdown rendering (similar to `glow`/`mdless`)
  - Headers, lists, code blocks, and bold text are beautifully styled in the terminal
  - Use `--no-markdown` flag to disable if you prefer plain text
- **Performance hooks**
  - Wired memory (mlock) support for model weights (MLX Metal wired memory API; requires macOS 15.0+)
  - Token counting and timing (tokens/second display)
- **Configurable directories**
  - Separate directories for logs, chats, and profiling stats (configurable in prompt config)
- **Voice mode (Moshi)**
  - Optional STT/TTS voice loop with local MLX models
  - Configurable via JSON + CLI overrides
  - Voice references from local voice repo (`tts_voice_path`)

See the docs in [docs/](docs/) for deeper details:

- [`FEATURES_FROM_MLX.md`](docs/FEATURES_FROM_MLX.md) – useful ideas to port from `mlx-lm`
- [`MEMORY_MANAGEMENT.md`](docs/MEMORY_MANAGEMENT.md) – wired memory (mlock) and multi-model notes
- [`PROMPT_CONFIG_REFERENCE.md`](docs/PROMPT_CONFIG_REFERENCE.md) – every field in the JSON config
- [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) – common issues and solutions
- [`ROADMAP.md`](docs/ROADMAP.md) – long-term planning and feature roadmap
- [`TODO.md`](docs/TODO.md) – short-term active work items
- [`MOSHI_CONFIG.md`](docs/MOSHI_CONFIG.md) – Moshi voice configuration

See [`examples/`](examples/) for practical usage examples:

- Basic chat, generation, prompt configs
- Profiles, conversation resume, custom placeholders
- Sampling parameters, server client, tools

---

## Installation

```bash
pip install mlx-harmony
```

This will pull in compatible versions of:

- `mlx-lm` (for model architectures and tokenizer utilities)
- `openai-harmony` (for Harmony format support)
- `rich` (for beautiful markdown rendering in terminal)
- `fastapi`, `uvicorn` (for the HTTP server)
- Plus a few support libs listed in `pyproject.toml`.

**Native tokenization:** `mlx-harmony` uses pure Python native tokenizer loading (no Rust, no sentencepiece, no jinja2, no mlx-lm tokenizers) to avoid PyTorch dependency. Tokenizers are loaded directly from `tokenizer.json` files using pure Python ByteLevel BPE implementation. This provides faster startup and eliminates all external tokenizer dependencies. The implementation is based on GPT-2 style ByteLevel BPE and is compatible with GPT-OSS models.

You'll need a working MLX setup (typically Python 3.12+ on Apple Silicon).

Optional Moshi voice support:

```bash
pip install "mlx-harmony[moshi]"
```

⸻

## Quick start

Chat with any MLX-LM model:

```bash
mlx-harmony-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Chat with a GPT-OSS model (Harmony formatting is automatic):

```bash
mlx-harmony-chat --model openai/gpt-oss-20b
```

Voice mode with Moshi (auto-loads `configs/moshi.json` if present):

```bash
mlx-harmony-chat --model openai/gpt-oss-20b --moshi
```

Run an OpenAI-style HTTP server:

```bash
mlx-harmony-server  # defaults to 0.0.0.0:8000
```

One-shot generation:

```bash
mlx-harmony-generate \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --prompt "Explain MXFP4 in simple terms."
```

---

## Moshi Voice Mode

Moshi voice mode uses local MLX models for STT and TTS, configured in `configs/moshi.json`. If that file exists, it is auto-loaded when you pass `--moshi`.

See the full guide at [MOSHI_CONFIG.md](docs/MOSHI_CONFIG.md).

### Recommended Model Layout

Keep Moshi assets under the project `models/` directory:

```text
models/
  stt-2.6b-en-mlx/
    config.json
    model.safetensors
    tokenizer_en_audio_4000.model
    mimi-pytorch-*.safetensors
  TTS/
    tts-1.6b-en_fr/
      config.json
      model.safetensors
      tokenizer_spm_32k_3.model
      mimi-*.safetensors
    moshi-tts-voices/
      ears/
        p001/
          freeform_speech_01.wav.1e68beda@240.safetensors
```

### moshi.json Relationship

- `stt_model_path` points to the STT model directory.
- `tts_model_path` points to the TTS model directory.
- `tts_voice_path` points to a voice embedding `.safetensors` file inside the voices repo.

### Quick Mic Check

After installing, run:

```bash
hotmic
```

This prints a live mic level meter and helps confirm microphone permissions before running `--moshi`.

### Model Sources

If you want the exact model and voice repo URLs in the docs, share the links you prefer and we’ll add them here.

---

## Profiling (Non-Interactive)

For repeatable profiling without typing delays, use the stdin harness and deterministic config:

```bash
scripts/profile_chat_stdin.sh \
  models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx
```

Deterministic config: [configs/prompt-config.deterministic.json](configs/prompt-config.deterministic.json)

Override defaults with environment variables:

```bash
PROMPT_CONFIG=configs/prompt-config.deterministic.json \
PROMPT_FILE=scripts/profile_chat_stdin.txt \
PROFILE_OUTPUT=profile.stats \
GRAPH_OUTPUT=profile.svg \
DEBUG_FILE=debug.log \
CHAT_FILE=profiling-chat.json \
scripts/profile_chat_stdin.sh models/your-model
```

See [scripts/README.md](scripts/README.md) for profiling details and output formats.

---

## Prompt configs

Prompt configs let you define:

- System / developer instructions
- Harmony-specific fields (`reasoning_effort`, `knowledge_cutoff`, etc.)
- Example dialogues (few-shot)
- Placeholders (`{assistant}`, `<|DATE|>`, `<|TIME|>`, etc.)
- Sampler parameters
- Context window management (`max_context_tokens`)
- Memory hints (`mlock`)
- Deterministic seeding (`seed`, `reseed_each_turn`)

`max_context_tokens` can also be overridden with `--max-context-tokens`. If unset,
the chat CLI will auto-detect `model_max_length` from the model’s
`tokenizer_config.json` when available.

Example ([configs/prompt-config.example.json](configs/prompt-config.example.json)):

```json
{
  "system_model_identity": "You are {assistant} on <|DATE|> at <|TIMEZ|> (local time, <|TIMEU|> UTC).",
  "reasoning_effort": "Medium",
  "conversation_start_date": "<|DATE|>",
  "knowledge_cutoff": "2025-01",
  "developer_instructions": "Address the user as {user} and be concise.",
  "assistant_greeting": "Hello {user}, I'm {assistant}. How can I help you today?",
  "example_dialogues": [
    [
      {"role": "user", "content": "Hello, how can you help me?"},
      {"role": "assistant", "content": "Hello {user}! I'm {assistant} and I'm here to help..."}
    ]
  ],
  "placeholders": {
    "assistant": "Dave",
    "user": "Morgan"
  },
  "max_tokens": 1024,
  "max_context_tokens": 4096,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.0,
  "min_tokens_to_keep": 1,
  "xtc_probability": 0.0,
  "xtc_threshold": 0.0,
  "xtc_special_tokens": null,
  "repetition_penalty": 1.0,
  "repetition_context_size": 20,
  "mlock": false,
  "seed": -1,
  "reseed_each_turn": false,
  "truncate_thinking": 1000,
  "truncate_response": 1000,
  "logs_dir": "logs",
  "chats_dir": "logs"
}
```

`max_tokens` is the maximum number of **new** tokens to generate for a response.

More detail (including all built-in date/time placeholders) is in [`PROMPT_CONFIG_REFERENCE.md`](docs/PROMPT_CONFIG_REFERENCE.md)

---

## Profiles

Profiles bundle “model + prompt config”:

```json
{
  "gpt-oss-20b": {
    "model": "~/models/gpt-oss-20b",
    "prompt_config": "configs/prompt-config.example.json"
  },
  "gpt-oss-120b": {
    "model": "~/models/gpt-oss-120b",
    "prompt_config": "configs/prompt-config.example.json"
  }
}
```

CLI:

```bash
mlx-harmony-chat \
  --profile gpt-oss-20b \
  --profiles-file configs/profiles.example.json
```

Profiles file example: [configs/profiles.example.json](configs/profiles.example.json)

---

## GPT-OSS tools

When using GPT-OSS models you can surface tool hooks:

```bash
mlx-harmony-chat \
  --model openai/gpt-oss-20b \
  --browser \
  --python \
  --apply-patch
```

Flags:

- `--browser` – browser / web navigation tool
- `--python` – Python execution tool
- `--apply-patch` – apply-patch tool for code edits

Right now, the loop does:
	1.	Detect tool calls in Harmony messages
	2.	Parse the tool + arguments
	3.	Call a stub executor (returns “not implemented”)
	4.	Feed the tool result back into the conversation

That gives you a clean place to plug real executors later (sandboxed Python, a Playwright browser, git patch application, etc.).

---

## Conversation logging & resume

You can save and resume conversations using a single chat name:

```bash
# Start a new chat or continue existing one
mlx-harmony-chat \
  --model openai/gpt-oss-20b \
  --chat mychat
```

Chat files are saved to `chats_dir/<name>.json` (default: `logs/<name>.json`). If the chat file exists, it loads and continues; if not, a new chat is created. New messages are automatically appended after each exchange.

**Dynamic hyperparameter changes during chat:**

You can change hyperparameters on the fly:

```bash
>> \set temperature=0.7
[INFO] Set temperature = 0.7

>> \set max_tokens=2048
[INFO] Set max_tokens = 2048
```

Valid parameters: `temperature`, `top_p`, `min_p`, `top_k`, `max_tokens`, `min_tokens_to_keep`, `repetition_penalty`, `repetition_context_size`, `xtc_probability`, `xtc_threshold`, `seed`

**Out-of-band commands:**

During chat, you can use special commands (prefixed with `\`):

- `\help` - Show list of all out-of-band commands
- `\list` or `\show` - Display current hyperparameters
- `\set <param>=<value>` - Change a hyperparameter (see above for valid parameters)
- `q` or `Control-D` - Quit the chat

If you enter an invalid `\` command, you'll see an error message with the list of valid commands.

The conversation JSON includes:

- Full message history (with timestamps)
- Model + prompt config metadata (schema versioned)
- Hyperparameters per turn (only when they change)
- Tool messages
- Generation stats (tokens/second)

Hyperparameters from the saved file are restored unless you override them via CLI or change them during chat.

If you need to migrate older logs, use:

```bash
mlx-harmony-migrate-chat logs/my-chat.json --in-place
```

---

## Memory & performance

Wired memory (mlock) keeps model weights in physical RAM, preventing swapping and improving inference performance:

- `mlock` – keep model weights wired in memory using MLX's Metal wired-memory APIs (requires macOS 15.0+).

You can set this in the prompt config:

```json
{
  "mlock": false
}
```

CLI override:

```bash
mlx-harmony-chat --model models/my-model --mlock
```

Mlock support requires macOS 15.0+ with Metal backend. For details on how it works, limitations, and best practices (especially when loading multiple models), see [`MEMORY_MANAGEMENT.md`](docs/MEMORY_MANAGEMENT.md). That guide covers:

- How wired memory works on macOS
- How to size the wired limit vs model size
- What happens with multiple large models loaded at once

---

## Python API

For direct programmatic use:

```python
from mlx_harmony import TokenGenerator

generator = TokenGenerator("openai/gpt-oss-20b")
messages = [{"role": "user", "content": "Hello!"}]

tokens = list(generator.generate(messages=messages, max_tokens=64))
text = generator.tokenizer.decode([int(t) for t in tokens])
print(text)
```

You still get:

- Automatic GPT-OSS detection and Harmony formatting (for GPT-OSS models)
- Native chat templates for other MLX-LM models
- Access to all sampling parameters exposed by mlx-lm

---

## Testing

The project includes a comprehensive test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only fast tests (skip model downloads)
pytest -m "not slow and not requires_model"

# Run tests with coverage
pytest --cov=mlx_harmony --cov-report=html
```

Tests use a small model (`mlx-community/Qwen1.5-0.5B-Chat-4bit`, ~300MB) for inference tests, which is automatically downloaded and cached by HuggingFace on first use.

See [`tests/README.md`](tests/README.md) for detailed test documentation.

## CI/CD

Continuous integration is configured with GitHub Actions:

- **Linting**: Runs `black` and `ruff` on Ubuntu
- **Fast Tests**: Runs unit tests on Ubuntu (Python 3.12, 3.13)
- **Full Tests**: Runs complete test suite on macOS-14 (Apple Silicon required for MLX)
- **Coverage**: Reports code coverage on pushes to `main`

See [`.github/workflows/README.md`](.github/workflows/README.md) for workflow details.

---

## Roadmap

The longer-term plan is tracked in:

- [`ROADMAP.md`](docs/ROADMAP.md)
- [`FEATURES_FROM_MLX.md`](docs/FEATURES_FROM_MLX.md)
- [`TODO.md`](docs/TODO.md)

Highlights on the horizon:

- Real implementations for GPT-OSS tools (browser, python, apply_patch)
- Prompt cache and KV-cache integration from `mlx-lm`
- `/v1/models` and `/health` endpoints in the server
- Speculative decoding and other performance features from upstream `mlx-lm`

Pull requests and issue reports are very welcome. If you're planning to tackle something big, check the roadmap first so we don't step on each other.
