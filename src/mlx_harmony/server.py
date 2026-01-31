from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from mlx_harmony.chat_history import (
    make_timestamp,
    write_debug_metrics,
    write_debug_prompt,
    write_debug_response,
)
from mlx_harmony.config import load_profiles, load_prompt_config
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.logging import get_logger

app = FastAPI(title="MLX Harmony API")
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
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


ChatMessage.model_rebuild()
ChatRequest.model_rebuild()


_generator: Optional[TokenGenerator] = None
_generator_prompt_config_path: Optional[str] = None
_generator_lock = Lock()
_server_metrics_row_index = 0
_last_prompt_start_time: float | None = None

DEFAULT_PROFILES_FILE = "configs/profiles.example.json"
DEFAULT_SERVER_DEBUG_LOG = "logs/server-debug.log"
DEFAULT_SERVER_LOG_PROMPTS = "1"


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


def _resolve_profile(
    request: ChatRequest,
) -> tuple[str, Optional[str]]:
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
    return model_path, prompt_config_path


def _decode_tokens(generator: TokenGenerator, token_ids: list[int]) -> str:
    if hasattr(generator, "tokenizer") and hasattr(generator.tokenizer, "decode"):
        return generator.tokenizer.decode(token_ids)
    backend = getattr(generator, "backend", None)
    if backend is not None and hasattr(backend, "decode"):
        return backend.decode(token_ids)
    return ""


def _next_server_metrics_row_index() -> int:
    global _server_metrics_row_index
    _server_metrics_row_index += 1
    return _server_metrics_row_index


def _metrics_timestamp() -> dict[str, str | float]:
    timestamp = make_timestamp()
    return {
        "timestamp_iso": timestamp["iso"],
        "timestamp_unix": timestamp["unix"],
    }


def _collect_memory_stats() -> dict[str, float | int | str]:
    if os.getenv("MLX_HARMONY_SERVER_COLLECT_MEMORY", "0") != "1":
        return {}
    try:
        import mlx.core as mx
    except Exception:
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, float | int | str] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats


def _write_server_metrics(
    *,
    generator: TokenGenerator,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_seconds: float,
    prompt_start_delta: float | None,
    prefill_seconds: float | None,
    debug_path: Path,
) -> None:
    timing_stats = getattr(generator, "last_timing_stats", None) or {}
    max_kv_size = None
    if getattr(generator, "prompt_config", None) is not None:
        max_kv_size = getattr(generator.prompt_config, "max_kv_size", None)
    kv_len = prompt_tokens
    if max_kv_size is not None:
        try:
            kv_len = min(prompt_tokens, int(max_kv_size))
        except (TypeError, ValueError):
            kv_len = prompt_tokens
    tokens_per_second = completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    write_debug_metrics(
        debug_path=debug_path,
        metrics={
            "row_index": _next_server_metrics_row_index(),
            **_metrics_timestamp(),
            "prompt_tokens": prompt_tokens,
            "kv_len": kv_len,
            "completion_tokens": completion_tokens,
            "generated_tokens": completion_tokens,
            "prefill_seconds": prefill_seconds or timing_stats.get("prefill"),
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": tokens_per_second,
            "prompt_start_to_prompt_start_seconds": prompt_start_delta,
            "max_context_tokens": getattr(generator.prompt_config, "max_context_tokens", None)
            if getattr(generator, "prompt_config", None) is not None
            else None,
            "max_kv_size": max_kv_size,
            "repetition_window": None,
            "loop_detection_mode": None,
            "prefill_start_offset": getattr(generator, "_last_prefill_start_offset", None),
            **_collect_memory_stats(),
        },
    )


def _should_log_prompt_response() -> bool:
    return os.getenv("MLX_HARMONY_SERVER_LOG_PROMPTS", DEFAULT_SERVER_LOG_PROMPTS) == "1"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    model_path, prompt_config_path = _resolve_profile(request)

    generator = _get_generator(model_path, prompt_config_path)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG))
    if _should_log_prompt_response():
        try:
            raw_prompt = generator.render_prompt(messages)
            write_debug_prompt(
                debug_path=debug_path,
                raw_prompt=raw_prompt,
                show_console=False,
            )
        except Exception as exc:
            logger.warning("Failed to write server debug prompt: %s", exc)

    if request.stream:

        def generate_stream():
            prompt_token_count = len(generator.render_prompt_tokens(messages))
            prompt_start = time.perf_counter()
            global _last_prompt_start_time
            prompt_start_delta = None
            if _last_prompt_start_time is not None:
                prompt_start_delta = prompt_start - _last_prompt_start_time
            _last_prompt_start_time = prompt_start
            generation_start = time.perf_counter()
            completion_tokens = 0
            debug_path = os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG)
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            collected_chunks: list[str] = []
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
                completion_tokens += 1
                text = _decode_tokens(generator, [int(token_id)])
                if _should_log_prompt_response():
                    collected_chunks.append(text)
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            generation_end = time.perf_counter()
            elapsed = generation_end - generation_start
            _write_server_metrics(
                generator=generator,
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed,
                prompt_start_delta=prompt_start_delta,
                prefill_seconds=None,
                debug_path=Path(debug_path),
            )
            if _should_log_prompt_response():
                try:
                    raw_response = "".join(collected_chunks)
                    write_debug_response(
                        debug_path=Path(debug_path),
                        raw_response=raw_response,
                        cleaned_response=raw_response,
                        show_console=False,
                    )
                except Exception as exc:
                    logger.warning("Failed to write server debug response: %s", exc)
            finish_reason = generator.last_finish_reason
            if not isinstance(finish_reason, str) or not finish_reason:
                finish_reason = "stop"
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    prompt_token_count = len(generator.render_prompt_tokens(messages))
    prompt_start = time.perf_counter()
    global _last_prompt_start_time
    prompt_start_delta = None
    if _last_prompt_start_time is not None:
        prompt_start_delta = prompt_start - _last_prompt_start_time
    _last_prompt_start_time = prompt_start
    generation_start = time.perf_counter()
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
    generation_end = time.perf_counter()
    text = _decode_tokens(generator, [int(t) for t in tokens])
    prompt_tokens = prompt_token_count
    completion_tokens = len(tokens)
    total_tokens = prompt_tokens + completion_tokens
    _write_server_metrics(
        generator=generator,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=generation_end - generation_start,
        prompt_start_delta=prompt_start_delta,
        prefill_seconds=None,
        debug_path=debug_path,
    )
    if _should_log_prompt_response():
        try:
            write_debug_response(
                debug_path=debug_path,
                raw_response=text,
                cleaned_response=text,
                show_console=False,
            )
        except Exception as exc:
            logger.warning("Failed to write server debug response: %s", exc)
    finish_reason = generator.last_finish_reason
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "stop"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_path,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    profiles_path = os.getenv("MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE)
    data = []
    if os.path.exists(profiles_path):
        try:
            profiles = load_profiles(profiles_path)
        except Exception as exc:
            logger.warning("Failed to load profiles from %s: %s", profiles_path, exc)
            profiles = {}
        for name, profile in profiles.items():
            model_id = profile.get("model", name)
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            )
    return {"object": "list", "data": data}


@app.get("/v1/health")
async def health_check() -> dict[str, str]:
    return {"object": "health", "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLX Harmony API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("MLX_HARMONY_HOST", "0.0.0.0"),
        help="Host interface to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MLX_HARMONY_PORT", "8000")),
        help="Port to bind the server.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MLX_HARMONY_LOG_LEVEL", "info"),
        help="Uvicorn log level (debug, info, warning, error).",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MLX_HARMONY_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes (development only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MLX_HARMONY_WORKERS", "1")),
        help="Number of worker processes (use 1 for in-process model cache).",
    )
    parser.add_argument(
        "--profiles-file",
        default=os.getenv("MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE),
        help="Profiles file path used for model selection.",
    )
    args: argparse.Namespace = parser.parse_args()

    os.environ["MLX_HARMONY_PROFILES_FILE"] = args.profiles_file
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
