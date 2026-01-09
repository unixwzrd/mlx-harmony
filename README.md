# MLX Harmony

Run GPT-OSS and other MLX-LM models through a single, lightweight interface.

`mlx-harmony` is a small wrapper around [`mlx-lm`](https://github.com/ml-explore/mlx-lm) and [`openai-harmony`](https://github.com/openai/openai-harmony) that gives you:

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
  - `mlx-harmony-chat` – interactive chat (with optional tools)
  - `mlx-harmony-generate` – single-shot text generation
  - `mlx-harmony-server` – OpenAI-style `/v1/chat/completions` endpoint

- **Prompt configs and profiles**
  - JSON configs for system text, developer instructions, placeholders, examples, and sampler settings
  - Profiles that bundle a model path + prompt config for one-flag switching

- **Conversation logging & resume**
  - Save chat sessions as JSON (with timestamps + hyperparameters per turn)
  - Resume with the same or a different model

- **Performance hooks**
  - Optional filesystem pre-warming
  - Wired memory (mlock) support for model weights (MLX Metal wired memory API; requires macOS 15.0+)
  - Token counting and timing (tokens/second display)
- **Configurable directories**
  - Separate directories for logs, chats, and profiling stats (configurable in prompt config)

See the docs in `docs/` for deeper details:

- [`IMPLEMENTATION_SUMMARY.md`](docs/IMPLEMENTATION_SUMMARY.md) – current state and architecture
- [`NEW_PROJECT_DESIGN.md`](docs/NEW_PROJECT_DESIGN.md) – original design doc
- [`FEATURES_FROM_MLX.md`](docs/FEATURES_FROM_MLX.md) – useful ideas to port from `mlx-lm`
- [`MEMORY_MANAGEMENT.md`](docs/MEMORY_MANAGEMENT.md) – wired memory, cache pre-warm, multi-model notes
- [`PROMPT_CONFIG_REFERENCE.md`](docs/PROMPT_CONFIG_REFERENCE.md) – every field in the JSON config
- [`TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) – common issues and solutions
- [`ROADMAP.md`](docs/ROADMAP.md), [`TODO.md`](docs/TODO.md) – what's next

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

- `mlx-lm`
- `openai-harmony`
- `fastapi`, `uvicorn` (for the HTTP server)
- Plus a few support libs listed in `pyproject.toml`.

You’ll need a working MLX setup (typically Python 3.12+ on Apple Silicon).

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

## Prompt configs

Prompt configs let you define:

- System / developer instructions
- Harmony-specific fields (`reasoning_effort`, `knowledge_cutoff`, etc.)
- Example dialogues (few-shot)
- Placeholders (`{assistant}`, `<|DATE|>`, etc.)
- Sampler parameters
- Memory hints (`prewarm_cache`, `mlock`)

Example (`configs/prompt-config.example.json`):

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
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.0,
  "min_tokens_to_keep": 1,
  "max_tokens": 1024,
  "repetition_penalty": 1.0,
  "repetition_context_size": 20,
  "prewarm_cache": true,
  "mlock": false,
  "truncate_thinking": 1000,
  "truncate_response": 1000,
  "logs_dir": "logs",
  "chats_dir": "logs"
}
```

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

Valid parameters: `temperature`, `top_p`, `min_p`, `top_k`, `max_tokens`, `repetition_penalty`, `repetition_context_size`

The conversation JSON includes:

- Full message history (with timestamps)
- Model + prompt config metadata
- Hyperparameters per turn (including changes made during chat)
- Tool messages
- Generation stats (tokens/second)

Hyperparameters from the saved file are restored unless you override them via CLI or change them during chat.

---

## Memory & performance

Two knobs exist for memory behavior:

- `prewarm_cache` (default: true) – read model weight files into the OS cache before load to speed up subsequent loads.
- `mlock` – keep model weights wired in memory using MLX's Metal wired-memory APIs (requires macOS 15.0+).

You can set these in the prompt config:

```json
{
  "prewarm_cache": true,
  "mlock": false
}
```

CLI overrides:

```bash
mlx-harmony-chat --model models/my-model --mlock
mlx-harmony-chat --model models/my-model --no-prewarm-cache
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
