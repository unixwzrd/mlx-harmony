# New Project Design: MLX-Harmony (Multi-Model + GPT-OSS Features)

## Concept

Create a **new project** that combines:

- **MLX-LM's multi-model support** (any model MLX-LM supports)
- **OpenHarmony-MLX's GPT-OSS features** (Harmony format, tools) when using GPT-OSS models
- **MLX backend only** (no torch, triton, vllm)
- **Maximal functionality** from both libraries

## Project Name: `mlx-harmony` or `harmony-mlx`

A unified application that:

- Works with **any MLX-LM supported model** (Llama, Mistral, Qwen, GPT-OSS, etc.)
- Automatically enables **Harmony format + tools** when using GPT-OSS models
- Uses **standard chat templates** for other models
- Provides **chat, server, CLI** functionality
- **MLX backend only** (simplified architecture)

## Architecture

```
┌─────────────────────────────────────┐
│   mlx-harmony (New Project)         │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  MLX-LM Integration         │   │
│   │  - Any model support        │   │
│   │  - Model loading            │   │
│   │  - Inference engine         │   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  Harmony Integration        │   │
│   │  - GPT-OSS format           │   │
│   │  - Auto-detect GPT-OSS      │   │
│   │  - Fallback to chat template│   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  GPT-OSS Tools              │   │
│   │  - Browser tool             │   │
│   │  - Python tool              │   │
│   │  - Apply patch tool         │   │
│   │  (Only for GPT-OSS models)  │   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  CLI & Server               │   │
│   │  - Chat interface           │   │
│   │  - Generation CLI           │   │
│   │  - HTTP API server          │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
         ▲              ▲
         │              │
    ┌────┴────┐    ┌────┴────┐
    │ mlx-lm  │    │ harmony │
    └─────────┘    └─────────┘
```

## Project Structure

```
mlx-harmony/
├── pyproject.toml
├── README.md
├── src/
│   └── mlx_harmony/
│       ├── __init__.py
│       ├── generator.py          # TokenGenerator (works with any model)
│       ├── chat.py               # Chat CLI with tools support
│       ├── server.py              # HTTP API server
│       ├── generate.py            # Generation CLI
│       ├── tools/                 # GPT-OSS tools (browser, python, apply_patch)
│       │   ├── __init__.py
│       │   ├── browser.py
│       │   ├── python.py
│       │   └── apply_patch.py
│       └── utils.py               # Model detection, format selection
└── tests/
    ├── test_generator.py
    ├── test_chat.py
    ├── test_server.py
    └── test_tools.py
```

## Core Features

### 1. Multi-Model Support

Works with **any MLX-LM supported model**:

- GPT-OSS models (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`)
- Llama models (`mlx-community/Llama-3.2-3B-Instruct-4bit`)
- Mistral models (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`)
- Qwen models (`mlx-community/Qwen2.5-7B-Instruct-4bit`)
- And 100+ more models

### 2. Auto-Detection

Automatically detects model type and enables appropriate features:

- **GPT-OSS models**: Harmony format + tools enabled
- **Other models**: Standard chat template, no tools

### 3. GPT-OSS Specific Features

When using GPT-OSS models:

- ✅ Harmony format rendering/parsing
- ✅ Browser tool
- ✅ Python tool
- ✅ Apply patch tool
- ✅ Reasoning effort control
- ✅ Full chain-of-thought support

### 4. Standard Model Support

When using other models:

- ✅ Native chat templates
- ✅ Standard generation
- ✅ All MLX-LM features

## Implementation

### Core Generator

```python
# src/mlx_harmony/generator.py
from typing import List, Dict, Optional, Iterator, Union
from mlx_lm import load, stream_generate
from openai_harmony import (
    load_harmony_encoding, HarmonyEncodingName,
    Conversation, Message, Role, SystemContent, TextContent
)

class TokenGenerator:
    """
    Multi-model token generator using MLX-LM + Harmony.
    
    Works with any MLX-LM supported model.
    Automatically uses Harmony format for GPT-OSS models.
    """
    
    def __init__(
        self,
        model_path: str,
        use_harmony: Optional[bool] = None,  # None = auto-detect
        lazy: bool = False,
    ):
        """
        Initialize generator for any MLX-LM model.
        
        Args:
            model_path: Path to model checkpoint or Hugging Face repo
            use_harmony: Whether to use Harmony format
                        None = auto-detect (True for GPT-OSS, False for others)
            lazy: Lazy load model weights
        """
        # Load model using MLX-LM
        self.model, self.tokenizer = load(model_path, lazy=lazy)
        self.model_path = model_path
        
        # Auto-detect if GPT-OSS model
        self.is_gpt_oss = self._is_gpt_oss_model(model_path)
        
        # Determine Harmony usage
        if use_harmony is None:
            use_harmony = self.is_gpt_oss  # Auto-enable for GPT-OSS
        self.use_harmony = use_harmony and self.is_gpt_oss  # Only if GPT-OSS
        
        # Load Harmony encoding if needed
        if self.use_harmony:
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        else:
            self.encoding = None
    
    def _is_gpt_oss_model(self, model_path: str) -> bool:
        """Detect if model is GPT-OSS."""
        path_lower = model_path.lower()
        return "gpt-oss" in path_lower or "gpt_oss" in path_lower
    
    def generate(
        self,
        prompt_tokens: Optional[List[int]] = None,
        messages: Optional[List[Dict]] = None,
        prompt: Optional[str] = None,
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
        system_message: Optional[str] = None,
    ) -> Iterator[Union[int, Dict]]:
        """
        Generate tokens with automatic format selection.
        
        - GPT-OSS models: Uses Harmony format
        - Other models: Uses native chat template
        """
        # Prepare prompt
        prompt_str = self._prepare_prompt(
            prompt_tokens=prompt_tokens,
            messages=messages,
            prompt=prompt,
            system_message=system_message
        )
        
        # Prepare generation kwargs
        kwargs = {"temperature": temperature}
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens
        
        # Handle stop tokens
        if stop_tokens:
            stop_strings = self._tokens_to_stop_strings(stop_tokens)
            kwargs["stop"] = stop_strings
        
        # Generate using MLX-LM
        for response in stream_generate(
            self.model, self.tokenizer, prompt_str, **kwargs
        ):
            token_id = self._extract_token_id(response)
            
            if return_logprobs:
                result = {"token": token_id}
                result["logprob"] = getattr(response, "logprob", None)
                yield result
            else:
                yield token_id
    
    def _prepare_prompt(
        self,
        prompt_tokens: Optional[List[int]] = None,
        messages: Optional[List[Dict]] = None,
        prompt: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """Prepare prompt with automatic format selection."""
        if messages:
            return self._messages_to_prompt(messages, system_message)
        elif prompt_tokens:
            return self.tokenizer.decode(prompt_tokens)
        elif prompt:
            if self.use_harmony:
                messages = [{"role": "user", "content": prompt}]
                return self._messages_to_prompt(messages, system_message)
            return prompt
        else:
            raise ValueError("Must provide prompt_tokens, messages, or prompt")
    
    def _messages_to_prompt(
        self,
        messages: List[Dict],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages to prompt using appropriate format."""
        if self.use_harmony and self.encoding:
            # Use Harmony format for GPT-OSS
            return self._harmony_messages_to_prompt(messages, system_message)
        else:
            # Use native chat template for other models
            return self._native_messages_to_prompt(messages, system_message)
    
    def _harmony_messages_to_prompt(
        self,
        messages: List[Dict],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages using Harmony format (GPT-OSS)."""
        harmony_messages = []
        
        if system_message:
            harmony_messages.append(
                Message.from_role_and_content(
                    Role.SYSTEM,
                    SystemContent.new().with_model_identity(system_message)
                )
            )
        
        for msg in messages:
            role_str = msg.get("role", "user").upper()
            role = Role(role_str)
            content_text = msg.get("content", "")
            content = TextContent.new().with_text(content_text)
            harmony_messages.append(
                Message.from_role_and_content(role, content)
            )
        
        conversation = Conversation.from_messages(harmony_messages)
        prompt_tokens = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        return self.tokenizer.decode(prompt_tokens)
    
    def _native_messages_to_prompt(
        self,
        messages: List[Dict],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages using native chat template (non-GPT-OSS)."""
        if system_message:
            messages = [{"role": "system", "content": system_message}] + messages
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    def _tokens_to_stop_strings(self, stop_tokens: List[int]) -> List[str]:
        """Convert stop token IDs to stop strings."""
        stop_strings = []
        for token_id in stop_tokens:
            stop_str = self.tokenizer.decode([token_id])
            if stop_str:
                stop_strings.append(stop_str)
        return stop_strings
    
    def _extract_token_id(self, response) -> int:
        """Extract token ID from MLX-LM response."""
        if hasattr(response, "text"):
            all_tokens = self.tokenizer.encode(response.text)
            if all_tokens:
                return all_tokens[-1]
        return int(response) if isinstance(response, (int, str)) else 0
```

### Chat CLI with Tools

```python
# src/mlx_harmony/chat.py
import argparse
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.tools import get_tools_for_model

def main():
    parser = argparse.ArgumentParser(description="Chat with any MLX-LM model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or Hugging Face repo",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser tool (GPT-OSS only)",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        help="Enable Python tool (GPT-OSS only)",
    )
    parser.add_argument(
        "--apply-patch",
        action="store_true",
        help="Enable apply_patch tool (GPT-OSS only)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()
    
    # Initialize generator
    generator = TokenGenerator(args.model)
    
    # Get tools if GPT-OSS model
    tools = []
    if generator.is_gpt_oss:
        tools = get_tools_for_model(
            browser=args.browser,
            python=args.python,
            apply_patch=args.apply_patch,
        )
    
    # Chat loop
    print(f"[INFO] Starting chat with {args.model}")
    if generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected - Harmony format + tools enabled")
    else:
        print("[INFO] Using native chat template")
    
    conversation_history = []
    while True:
        user_input = input("\n>> ")
        if user_input.lower() == "q":
            break
        
        messages = conversation_history + [{"role": "user", "content": user_input}]
        
        # Generate response
        response_tokens = []
        for token in generator.generate(
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ):
            response_tokens.append(token)
            # Decode and print incrementally
            text = generator.tokenizer.decode([token])
            print(text, end="", flush=True)
        print()
        
        # Update conversation history
        conversation_history = messages + [
            {"role": "assistant", "content": generator.tokenizer.decode(response_tokens)}
        ]
```

### HTTP API Server

```python
# src/mlx_harmony/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from mlx_harmony.generator import TokenGenerator

app = FastAPI(title="MLX Harmony API")

# Global generator (can be made per-request if needed)
generator: Optional[TokenGenerator] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    max_tokens: int = 512
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI-compatible)."""
    global generator
    
    # Initialize generator if needed or model changed
    if generator is None or generator.model_path != request.model:
        generator = TokenGenerator(request.model)
    
    # Convert messages
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if request.stream:
        # Streaming response
        def generate_stream():
            for token in generator.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                text = generator.tokenizer.decode([token])
                yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        # Non-streaming response
        tokens = list(generator.generate(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ))
        text = generator.tokenizer.decode(tokens)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Package Configuration

### `pyproject.toml`

```toml
[project]
name = "mlx-harmony"
version = "0.1.0"
description = "Multi-model MLX inference with GPT-OSS Harmony format and tools"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "mlx-lm>=0.1.0",
    "openai-harmony>=0.0.8",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "aiohttp>=3.12.0",  # For browser tool
    "docker>=7.1.0",    # For python tool
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black",
    "ruff",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["mlx_harmony"]

[project.scripts]
mlx-harmony-chat = "mlx_harmony.chat:main"
mlx-harmony-server = "mlx_harmony.server:main"
mlx-harmony-generate = "mlx_harmony.generate:main"
```

## Usage Examples

### GPT-OSS Model (Harmony + Tools)

```bash
# Chat with GPT-OSS (Harmony format + tools)
mlx-harmony-chat --model openai/gpt-oss-20b --browser --python

# Generate text
mlx-harmony-generate --model openai/gpt-oss-20b --prompt "Hello!"

# Start server
mlx-harmony-server --model openai/gpt-oss-20b
```

### Other Models (Native Format)

```bash
# Chat with Llama (native chat template)
mlx-harmony-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Chat with Mistral
mlx-harmony-chat --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Generate text
mlx-harmony-generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "Hello!"
```

### Python API

```python
from mlx_harmony import TokenGenerator

# GPT-OSS model (auto-detects Harmony format)
gpt_oss = TokenGenerator("openai/gpt-oss-20b")
messages = [{"role": "user", "content": "Hello!"}]
for token in gpt_oss.generate(messages=messages, max_tokens=100):
    print(token)

# Llama model (uses native chat template)
llama = TokenGenerator("mlx-community/Llama-3.2-3B-Instruct-4bit")
messages = [{"role": "user", "content": "Hello!"}]
for token in llama.generate(messages=messages, max_tokens=100):
    print(token)
```

## Features Comparison

| Feature | MLX-LM | mlx-harmony (New) |
|---------|--------|-------------------|
| **Model Support** | Any MLX model | Any MLX model |
| **Harmony Format** | ❌ | ✅ (auto for GPT-OSS) |
| **GPT-OSS Tools** | ❌ | ✅ (browser, python, apply_patch) |
| **Chat CLI** | ✅ | ✅ (with tools support) |
| **Server** | ✅ (OpenAI API) | ✅ (OpenAI API) |
| **Generation CLI** | ✅ | ✅ |
| **Auto-Detection** | ❌ | ✅ (GPT-OSS vs others) |
| **Backend** | MLX only | MLX only |

## Benefits

### 1. **Multi-Model Support**

- Works with any MLX-LM model
- Not limited to GPT-OSS

### 2. **Automatic Feature Detection**

- GPT-OSS models: Harmony format + tools
- Other models: Native chat templates

### 3. **Simplified Architecture**

- MLX backend only (no torch, triton, vllm)
- Cleaner codebase

### 4. **Maximal Functionality**

- All MLX-LM features
- All GPT-OSS specific features
- Best of both worlds

### 5. **Easy to Use**

- Simple CLI commands
- Python API
- HTTP server

## Migration from OpenHarmony-MLX

If you want to migrate from OpenHarmony-MLX:

1. **Install new package**:

   ```bash
   pip install mlx-harmony
   ```

2. **Replace commands**:

   ```bash
   # Old
   python -m gpt_oss.chat --backend mlx model/
   
   # New
   mlx-harmony-chat --model model/
   ```

3. **Python API**:

   ```python
   # Old
   from gpt_oss.mlx_gpt_oss import TokenGenerator
   
   # New
   from mlx_harmony import TokenGenerator
   ```

## Summary

**New Project: `mlx-harmony`**

- ✅ **Multi-model**: Works with any MLX-LM model
- ✅ **Auto-detection**: Harmony format for GPT-OSS, native for others
- ✅ **GPT-OSS features**: Tools, Harmony format when using GPT-OSS
- ✅ **MLX only**: Simplified, no other backends
- ✅ **Maximal functionality**: Chat, server, CLI, tools
- ✅ **Easy to use**: Simple commands, Python API

This gives you:

- The flexibility of MLX-LM (any model)
- The power of OpenHarmony-MLX (GPT-OSS features)
- Simplified architecture (MLX only)
- One unified application

---

[← Back to README](../README.md)
