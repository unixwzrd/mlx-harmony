from __future__ import annotations

import json
import os
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_harmony.config import load_profiles, load_prompt_config
from mlx_harmony.generator import TokenGenerator

app = FastAPI(title="MLX Harmony API")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 0.0
    repetition_context_size: int = 20
    profile: Optional[str] = None
    prompt_config: Optional[str] = None
    profiles_file: Optional[str] = None
    stream: bool = False


_generator: Optional[TokenGenerator] = None
_generator_prompt_config_path: Optional[str] = None
_generator_lock = Lock()

DEFAULT_PROFILES_FILE = "configs/profiles.example.json"


def _get_generator(model: str, prompt_config_path: Optional[str]) -> TokenGenerator:
    global _generator
    global _generator_prompt_config_path
    with _generator_lock:
        if (
            _generator is None
            or _generator.model_path != model
            or _generator_prompt_config_path != prompt_config_path
        ):
            prompt_cfg = load_prompt_config(prompt_config_path) if prompt_config_path else None
            _generator = TokenGenerator(model, prompt_config=prompt_cfg)
            _generator_prompt_config_path = prompt_config_path
        return _generator


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Minimal OpenAI-compatible chat completions endpoint.

    Supports:
    - messages
    - temperature
    - max_tokens
    - stream
    """
    # Resolve profile if provided
    model_path = request.model
    prompt_config_path = request.prompt_config
    if request.profile:
        profiles_path = request.profiles_file or os.getenv(
            "MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE
        )
        profiles = load_profiles(profiles_path)
        if request.profile not in profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{request.profile}' not found",
            )
        profile = profiles[request.profile]
        model_path = profile.get("model", model_path)
        prompt_config_path = prompt_config_path or profile.get("prompt_config")

    generator = _get_generator(model_path, prompt_config_path)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:

        def generate_stream():
            for token_id in generator.generate(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                min_p=request.min_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            ):
                text = generator.tokenizer.decode([int(token_id)])
                chunk = {
                    "choices": [
                        {
                            "delta": {"content": text},
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    tokens = list(
        generator.generate(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
        )
    )
    text = generator.tokenizer.decode([int(t) for t in tokens])

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                }
            }
        ]
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
