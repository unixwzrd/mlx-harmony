# MLX Harmony

`mlx-harmony` is a lightweight wrapper around `mlx-lm` and `openai-harmony`
that gives you:

- **Multi‑model MLX inference** using `mlx-lm` (Llama, Mistral, Qwen, GPT‑OSS, …)
- **Harmony formatting** automatically enabled for GPT‑OSS models
- Simple **chat**, **generate**, and **HTTP API server** entrypoints

### Installation

```bash
pip install mlx-harmony
```

You will also need compatible versions of `mlx-lm` and `openai-harmony`
installed; they are declared as dependencies in `pyproject.toml`.

### Quick start

Chat with any MLX‑LM model:

```bash
mlx-harmony-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Chat with a GPT‑OSS model (Harmony format is used automatically):

```bash
mlx-harmony-chat --model openai/gpt-oss-20b
```

Run the HTTP API server (OpenAI‑style `/v1/chat/completions`):

```bash
mlx-harmony-server  # defaults to port 8000
```

Generate text once from the CLI:

```bash
mlx-harmony-generate \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --prompt "Explain MXFP4 in simple terms."
```

### Prompt configs (Harmony + placeholders)

You can supply a JSON prompt config to set Harmony fragments, placeholders, default sampling parameters, and model loading optimizations. See **[PROMPT_CONFIG_REFERENCE.md](./docs/PROMPT_CONFIG_REFERENCE.md)** for complete documentation of all parameters.

`configs/prompt-config.example.json`:

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
  "mlock": false
}
```

**Key Parameters:**

- **`max_tokens`**: Maximum tokens to generate (default: 1024 for Harmony models, 512 otherwise). Increase for longer responses.
- **`temperature`**: Sampling temperature (0.0-2.0, default: 0.8). Higher = more creative.
- **`top_p`**: Nucleus sampling (0.0-1.0, default: 0.9). Keeps tokens with cumulative probability ≤ top_p.
- **`repetition_penalty`**: Penalty for repetition (>1.0 reduces repetition, default: 1.0).

See **[PROMPT_CONFIG_REFERENCE.md](./docs/PROMPT_CONFIG_REFERENCE.md)** for complete documentation of all parameters.

**Example Dialogues (Few-shot Examples)**: The `example_dialogues` field allows you to include sample conversations that are part of the prompt but not sent every time like system/developer instructions. This is useful for:

- Role-playing assistants
- Customer service bots
- Demonstrating desired conversation style

**Importing Dialogue Text Format**: You can convert simple dialogue text format to JSON for `example_dialogues`:

```bash
# Convert dialogue text file to JSON format
mlx-harmony-convert-dialogue input.txt output.json --as-example-dialogues
```

Input format:

```text
assistant: Hello, how may I help you today?
user: I'd like to know something about fruit.
assistant: What would you like to know about fruit?
user: What is the difference between a fruit and a vegetable?
```

Use it:

```bash
mlx-harmony-chat --model ~/models/gpt-oss-20b --prompt-config configs/prompt-config.example.json
```

### Profiles (optional)

You can define profiles to bundle a model path and prompt config:

`configs/profiles.example.json`:

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
mlx-harmony-chat --profile gpt-oss-20b --profiles-file configs/profiles.example.json
```

### GPT-OSS Tools (GPT-OSS models only)

For GPT-OSS models, you can enable tools that the model can call during conversation:

```bash
mlx-harmony-chat \
  --model openai/gpt-oss-20b \
  --browser \
  --python \
  --apply-patch
```

Available tools:

- `--browser`: Browser tool for web navigation and interaction
- `--python`: Python tool for executing Python code in a sandbox
- `--apply-patch`: Apply patch tool for code modifications

**Note**: Tool execution is currently stubbed. The infrastructure for detecting and parsing tool calls is in place, but actual tool executors need to be implemented. When a tool call is detected, the chat loop will:

1. Parse the tool call from the model's response
2. Execute the tool (currently returns a "not implemented" message)
3. Feed the result back into the conversation
4. Continue generation with the tool result

### Debug Mode

Enable debug mode to see raw prompts and responses:

```bash
mlx-harmony-chat \
  --model openai/gpt-oss-20b \
  --debug
```

This outputs the raw text prompts sent to the LLM and the raw responses received, useful for debugging prompt formatting and model behavior.

### Conversation Logging & Resuming

Save conversations to resume later:

```bash
# Save conversation to a JSON file (auto-saves after each exchange)
mlx-harmony-chat \
  --model openai/gpt-oss-20b \
  --save-conversation conversations/my-chat.json
```

Resume a previous conversation:

```bash
# Load and continue from a saved conversation
mlx-harmony-chat --load-conversation conversations/my-chat.json
```

The conversation JSON format includes metadata (model, prompt config, tools, hyperparameters) and the full message history with timestamps. Each message (turn) includes a timestamp when it was created. See `configs/conversation.example.json` for the schema.

**Features:**

- Auto-saves after each exchange (turn)
- Timestamps on every message
- Preserves model/prompt config information
- Saves and restores hyperparameters (temperature, top_p, etc.)
- **Per-turn hyperparameter tracking**: Each assistant and tool message stores the hyperparameters used for that generation, allowing you to see how hyperparameters changed during the conversation
- Supports resuming with same or different model
- Hyperparameters from saved conversation are restored unless overridden via CLI
- Tool messages are preserved in the format

### Python API

```python
from mlx_harmony import TokenGenerator

generator = TokenGenerator("openai/gpt-oss-20b")
messages = [{"role": "user", "content": "Hello!"}]

tokens = list(generator.generate(messages=messages, max_tokens=64))
text = generator.tokenizer.decode([int(t) for t in tokens])
print(text)
```

---

## Performance Profiling

Profile the chat interface to identify performance bottlenecks:

```bash
# Profile the actual running chat (recommended - shows real-world usage)
python scripts/profile_chat.py \
  --model models/your-model \
  --prompt-config configs/your-config.json \
  --graph profile.svg

# Or profile just startup/initialization
python scripts/profile_startup.py \
  --model models/your-model \
  --prompt-config configs/your-config.json \
  --graph profile.svg

# View the call graph
open profile.svg  # macOS
# or
xdg-open profile.svg  # Linux
```

See **[scripts/README.md](./scripts/README.md)** for detailed profiling instructions.

---

## Model Loading Optimizations

### Filesystem Cache Pre-warming

By default, `mlx-harmony` pre-warms the filesystem cache before loading models, which can significantly speed up model loading (especially on subsequent loads):

```bash
mlx-harmony-chat --model models/my-model  # Pre-warming enabled by default

# Disable pre-warming
mlx-harmony-chat --model models/my-model --no-prewarm-cache
```

You can also control this in your prompt config JSON:

```json
{
  "prewarm_cache": true
}
```

### Memory Locking (mlock)

On macOS with Metal backend, you can lock model weights in memory to prevent swapping:

```bash
mlx-harmony-chat --model models/my-model --mlock
```

You can also enable this in your prompt config JSON:

```json
{
  "mlock": true
}
```

**Note**: Memory locking (mlock) requires:

- macOS with Metal backend
- Model size must fit within 90% of Metal's recommended working set size
- Uses MLX's `set_wired_limit()` under the hood (mlock equivalent)

For detailed information about memory management, including wired memory, pre-warming, and considerations for loading multiple models, see **[Memory Management Guide](docs/MEMORY_MANAGEMENT.md)**.

---

## Roadmap & Contributing

- **[ROADMAP.md](./docs/ROADMAP.md)**: Detailed roadmap with planned enhancements and features
- **[FEATURES_FROM_MLX.md](./docs/FEATURES_FROM_MLX.md)**: Features identified from mlx-lm/mlx-examples to incorporate
- **[TODO.md](./docs/TODO.md)**: Active checklist for current work items
- **[CHANGELOG.md](./CHANGELOG.md)**: Version history and release notes

Contributions welcome! Please check the roadmap before starting work on major features.
