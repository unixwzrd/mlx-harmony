"""FastAPI server endpoints and shared runtime plumbing for MLX Harmony."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from mlx_harmony.backend_api import build_conversation_from_messages, run_backend_chat
from mlx_harmony.chat_history import (
    make_message_id,
    make_timestamp,
    write_debug_info,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.chat_utils import (
    build_hyperparameters_from_request,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
)
from mlx_harmony.config import (
    DEFAULT_CONFIG_DIR,
    PromptConfig,
    load_profiles,
    load_prompt_config,
    resolve_config_path,
)
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.logging import get_logger
from mlx_harmony.runtime.model_init import initialize_generator

app = FastAPI(title="MLX Harmony API")
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    profile: Optional[str] = None
    prompt_config: Optional[str] = None
    profiles_file: Optional[str] = None
    stream: bool = False
    return_analysis: bool = False


ChatMessage.model_rebuild()
ChatRequest.model_rebuild()


_generator: Optional[TokenGenerator] = None
_generator_prompt_config_path: Optional[str] = None
_generator_lock = Lock()
_loaded_model_path: Optional[str] = None
_loaded_at_unix: Optional[int] = None
_last_prompt_start_time: float | None = None
_generation_index: int = 0

DEFAULT_PROFILES_FILE = "configs/profiles.example.json"
DEFAULT_SERVER_DEBUG_LOG = "logs/server-debug.log"
DEFAULT_MODELS_DIR = "models"


def _get_generator(model: str, prompt_config_path: Optional[str]) -> TokenGenerator:
    """Load or reuse a cached generator using the shared CLI init path.

    Args:
        model: Model path to load.
        prompt_config_path: Prompt config file path, if any.

    Returns:
        Loaded TokenGenerator instance.
    """
    global _generator
    global _generator_prompt_config_path
    global _loaded_model_path
    global _loaded_at_unix
    with _generator_lock:
        if (
            _generator is None
            or _generator.model_path != model
            or _generator_prompt_config_path != prompt_config_path
        ):
            prompt_cfg = load_prompt_config(prompt_config_path) if prompt_config_path else None
            if prompt_cfg is not None and not isinstance(prompt_cfg, PromptConfig):
                prompt_cfg = PromptConfig.model_validate(prompt_cfg)
            mlock = bool(getattr(prompt_cfg, "mlock", False)) if prompt_cfg else False
            logger.info(
                "Loading model: %s (prompt_config=%s)",
                model,
                prompt_config_path or "none",
            )
            _generator = initialize_generator(
                model_path=model,
                prompt_config=prompt_cfg,
                prompt_config_path=prompt_config_path,
                lazy=False,
                mlock=mlock,
            )
            _generator_prompt_config_path = prompt_config_path
            _loaded_model_path = model
            _loaded_at_unix = int(time.time())
        return _generator


def _resolve_profile(
    request: ChatRequest,
) -> tuple[str, Optional[str], dict[str, object] | None]:
    """Resolve model/prompt config paths from request or profiles.

    Args:
        request: Parsed chat request payload.

    Returns:
        Tuple of (model_path, prompt_config_path, profile_data).
    """
    model_path = request.model
    prompt_config_path = request.prompt_config
    profile_data: dict[str, object] | None = None
    if request.profile:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        profiles_path = request.profiles_file or os.getenv(
            "MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE
        )
        resolved_profiles_path = resolve_config_path(profiles_path, config_dir)
        if resolved_profiles_path is None or not resolved_profiles_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Profiles file not found: {profiles_path}",
            )
        profiles = load_profiles(str(resolved_profiles_path))
        if request.profile not in profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{request.profile}' not found in {resolved_profiles_path}",
            )
        profile = profiles[request.profile]
        profile_data = profile
        model_path = profile.get("model", model_path)
        prompt_config_path = prompt_config_path or profile.get("prompt_config")
        if prompt_config_path:
            resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
            if resolved_prompt_path is None or not resolved_prompt_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt config not found: {prompt_config_path}",
                )
            prompt_config_path = str(resolved_prompt_path)
    elif prompt_config_path:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
        if resolved_prompt_path is None or not resolved_prompt_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Prompt config not found: {prompt_config_path}",
            )
        prompt_config_path = str(resolved_prompt_path)
    # If request omits model/prompt-config, reuse currently loaded defaults.
    if model_path is None:
        model_path = _loaded_model_path
    if prompt_config_path is None:
        prompt_config_path = _generator_prompt_config_path
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="No model provided and no server default model is loaded.",
        )
    return model_path, prompt_config_path, profile_data


def _resolve_profile_from_args(
    *,
    model: Optional[str],
    profile: Optional[str],
    prompt_config: Optional[str],
    profiles_file: str,
) -> tuple[Optional[str], Optional[str]]:
    model_path = model
    prompt_config_path = prompt_config
    if profile:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_profiles_path = resolve_config_path(profiles_file, config_dir)
        if resolved_profiles_path is None or not resolved_profiles_path.exists():
            raise RuntimeError(f"Profiles file not found: {profiles_file}")
        profiles = load_profiles(str(resolved_profiles_path))
        if profile not in profiles:
            raise RuntimeError(f"Profile '{profile}' not found in {resolved_profiles_path}")
        profile_data = profiles[profile]
        model_path = profile_data.get("model", model_path)
        prompt_config_path = prompt_config_path or profile_data.get("prompt_config")
        if prompt_config_path:
            resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
            if resolved_prompt_path is None or not resolved_prompt_path.exists():
                raise RuntimeError(f"Prompt config not found: {prompt_config_path}")
            prompt_config_path = str(resolved_prompt_path)
    elif prompt_config_path:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
        if resolved_prompt_path is None or not resolved_prompt_path.exists():
            raise RuntimeError(f"Prompt config not found: {prompt_config_path}")
        prompt_config_path = str(resolved_prompt_path)
    return model_path, prompt_config_path


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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    global _last_prompt_start_time
    global _generation_index
    model_path, prompt_config_path, profile_data = _resolve_profile(request)

    generator = _get_generator(model_path, prompt_config_path)
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG))

    if request.stream:

        def generate_stream():
            global _generation_index
            prompt_config = getattr(generator, "prompt_config", None)
            conversation = build_conversation_from_messages(
                messages=raw_messages,
                prompt_config=prompt_config,
                make_message_id=make_message_id,
                make_timestamp=make_timestamp,
            )
            global _last_prompt_start_time
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            stream_hyperparameters = build_hyperparameters_from_request(
                request=request,
                prompt_config=prompt_config,
                is_harmony=bool(
                    getattr(generator, "is_gpt_oss", False)
                    and getattr(generator, "use_harmony", False)
                ),
            )
            max_context_tokens = resolve_max_context_tokens(
                args=SimpleNamespace(max_context_tokens=None),
                loaded_max_context_tokens=None,
                loaded_model_path=None,
                prompt_config=prompt_config,
                profile_data=profile_data,
                model_path=model_path,
            )
            last_user_text = None
            for msg in reversed(conversation):
                if msg.get("role") == "user":
                    last_user_text = str(msg.get("content") or "")
                    break
            backend_result = run_backend_chat(
                generator=generator,
                conversation=conversation,
                hyperparameters=stream_hyperparameters,
                assistant_name=get_assistant_name(prompt_config),
                thinking_limit=get_truncate_limits(prompt_config)[0],
                response_limit=get_truncate_limits(prompt_config)[1],
                render_markdown=False,
                debug_path=debug_path,
                debug_tokens=None,
                enable_artifacts=True,
                max_context_tokens=max_context_tokens,
                max_tool_iterations=10,
                max_resume_attempts=2,
                tools=[],
                last_user_text=last_user_text,
                make_message_id=make_message_id,
                make_timestamp=make_timestamp,
                collect_memory_stats=_collect_memory_stats,
                write_debug_metrics=write_debug_metrics,
                write_debug_response=write_debug_response,
                write_debug_info=write_debug_info,
                write_debug_token_texts=write_debug_token_texts,
                write_debug_tokens=write_debug_tokens,
                last_prompt_start_time=_last_prompt_start_time,
                generation_index=_generation_index,
            )
            _last_prompt_start_time = backend_result.last_prompt_start_time
            _generation_index = backend_result.generation_index
            text = backend_result.assistant_text or ""
            if text:
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
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': backend_result.finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    prompt_config = getattr(generator, "prompt_config", None)
    assistant_name = get_assistant_name(prompt_config)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
    render_markdown = False
    tools: list[object] = []

    conversation = build_conversation_from_messages(
        messages=raw_messages,
        prompt_config=prompt_config,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
    )
    messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in conversation]

    hyperparameters = build_hyperparameters_from_request(
        request=request,
        prompt_config=prompt_config,
        is_harmony=bool(
            getattr(generator, "is_gpt_oss", False) and getattr(generator, "use_harmony", False)
        ),
    )
    max_context_tokens = resolve_max_context_tokens(
        args=SimpleNamespace(max_context_tokens=None),
        loaded_max_context_tokens=None,
        loaded_model_path=None,
        prompt_config=prompt_config,
        profile_data=profile_data,
        model_path=model_path,
    )

    last_user_text = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    backend_result = run_backend_chat(
        generator=generator,
        conversation=conversation,
        hyperparameters=hyperparameters,
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
        debug_path=debug_path,
        debug_tokens=None,
        enable_artifacts=True,
        max_context_tokens=max_context_tokens,
        max_tool_iterations=10,
        max_resume_attempts=2,
        tools=tools,
        last_user_text=last_user_text,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        collect_memory_stats=_collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        last_prompt_start_time=_last_prompt_start_time,
        generation_index=_generation_index,
    )
    _last_prompt_start_time = backend_result.last_prompt_start_time
    _generation_index = backend_result.generation_index

    assistant_text = backend_result.assistant_text
    analysis_text: str | None = None
    if request.return_analysis:
        analysis_text = backend_result.analysis_text

    prompt_tokens = backend_result.prompt_tokens
    completion_tokens = backend_result.completion_tokens
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = backend_result.finish_reason
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
                "content": assistant_text,
                **({"analysis": analysis_text} if analysis_text else {}),
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
    """List available local models using the configured models/profiles.

    Returns:
        OpenAI-compatible list response for local models.
    """
    models_dir = os.getenv("MLX_HARMONY_MODELS_DIR", DEFAULT_MODELS_DIR)
    data = []
    base = Path(models_dir)
    if base.exists():
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                data.append(
                    {
                        "id": str(entry),
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                )
    if not data:
        profiles_path = os.getenv("MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE)
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
async def health_check() -> dict[str, str | bool | int | None]:
    """Report server liveness and model load state.

    Returns:
        Health metadata including model load state and paths.
    """
    model_loaded = _generator is not None
    return {
        "object": "health",
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": _loaded_model_path,
        "prompt_config_path": _generator_prompt_config_path,
        "loaded_at_unix": _loaded_at_unix,
    }


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
    parser.add_argument(
        "--model",
        default=os.getenv("MLX_HARMONY_MODEL_PATH", None),
        help="Model path for preload (optional).",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("MLX_HARMONY_PROFILE", None),
        help="Profile name for preload (optional).",
    )
    parser.add_argument(
        "--prompt-config",
        default=os.getenv("MLX_HARMONY_PROMPT_CONFIG", None),
        help="Prompt config path for preload (optional).",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MLX_HARMONY_PRELOAD", "false").lower() == "true",
        help="Preload model at startup (default: false).",
    )
    args: argparse.Namespace = parser.parse_args()

    os.environ["MLX_HARMONY_PROFILES_FILE"] = args.profiles_file
    if args.preload:
        try:
            model_path, prompt_config_path = _resolve_profile_from_args(
                model=args.model,
                profile=args.profile,
                prompt_config=args.prompt_config,
                profiles_file=args.profiles_file,
            )
        except RuntimeError as exc:
            logger.error("Failed to preload model: %s", exc)
        else:
            if model_path:
                _get_generator(model_path, prompt_config_path)
            else:
                logger.warning(
                    "Preload requested but no model/profile provided; skipping preload."
                )
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
