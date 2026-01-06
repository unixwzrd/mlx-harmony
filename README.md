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

You can supply a JSON prompt config to set Harmony fragments, placeholders, and default sampling:

`configs/prompt-config.example.json`:

```json
{
  "system_model_identity": "You are {assistant} on <|DATE|> at <|DATETIME|>.",
  "reasoning_effort": "Medium",
  "conversation_start_date": "<|DATE|>",
  "knowledge_cutoff": "2025-01",
  "developer_instructions": "Address the user as {user} and be concise.",
  "placeholders": {
    "assistant": "Dave",
    "user": "Morgan"
  },
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.0,
  "min_tokens_to_keep": 1,
  "repetition_penalty": 1.0,
  "repetition_context_size": 20
}
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

### Python API

```python
from mlx_harmony import TokenGenerator

generator = TokenGenerator("openai/gpt-oss-20b")
messages = [{"role": "user", "content": "Hello!"}]

tokens = list(generator.generate(messages=messages, max_tokens=64))
text = generator.tokenizer.decode([int(t) for t in tokens])
print(text)
```
