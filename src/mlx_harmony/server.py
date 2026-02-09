"""FastAPI server endpoints and shared runtime plumbing for MLX Harmony."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mlx_harmony.api_contract import (
    ChatRequest,
    RequestValidationError,
    apply_stop_sequences,
    validate_chat_request_supported,
)
from mlx_harmony.backend_api import run_backend_chat
from mlx_harmony.backend_runtime import (
    BackendInputs,
    BackendState,
    execute_backend_turn,
    prepare_backend_inputs,
)
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
    resolve_api_profile_paths,
    resolve_startup_profile_paths,
)
from mlx_harmony.config import (
    PromptConfig,
    load_profiles,
    load_prompt_config,
)
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.logging import get_logger
from mlx_harmony.runtime.model_init import initialize_generator

app = FastAPI(title="MLX Harmony API")
logger = get_logger(__name__)

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


def _prepare_backend_inputs(
    *,
    request: ChatRequest,
    generator: TokenGenerator,
    model_path: str,
    profile_data: dict[str, object] | None,
) -> dict[str, Any]:
    """Compatibility wrapper that prepares shared backend inputs.

    Args:
        request: Parsed chat request payload.
        generator: Loaded generator instance.
        model_path: Resolved model path for this request.
        profile_data: Optional profile payload used for config resolution.

    Returns:
        Dict form of shared backend inputs used by server request handling.
    """
    inputs = prepare_backend_inputs(
        request=request,
        generator=generator,
        model_path=model_path,
        profile_data=profile_data,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
    )
    return {
        "conversation": inputs.conversation,
        "hyperparameters": inputs.hyperparameters,
        "max_context_tokens": inputs.max_context_tokens,
        "last_user_text": inputs.last_user_text,
        "assistant_name": inputs.assistant_name,
        "thinking_limit": inputs.thinking_limit,
        "response_limit": inputs.response_limit,
        "tools": inputs.tools,
        "render_markdown": inputs.render_markdown,
    }


def _execute_backend_turn(
    *,
    generator: TokenGenerator,
    backend_inputs: dict[str, Any],
    debug_path: Path,
) -> Any:
    """Compatibility wrapper that executes one shared backend turn.

    Args:
        generator: Loaded token generator.
        backend_inputs: Dict produced by `_prepare_backend_inputs`.
        debug_path: Server debug log path.

    Returns:
        Backend result object from the shared chat turn path.
    """
    global _last_prompt_start_time
    global _generation_index
    state = BackendState(
        last_prompt_start_time=_last_prompt_start_time,
        generation_index=_generation_index,
    )
    inputs = BackendInputs(
        conversation=backend_inputs["conversation"],
        hyperparameters=backend_inputs["hyperparameters"],
        max_context_tokens=backend_inputs["max_context_tokens"],
        last_user_text=backend_inputs["last_user_text"],
        assistant_name=backend_inputs["assistant_name"],
        thinking_limit=backend_inputs["thinking_limit"],
        response_limit=backend_inputs["response_limit"],
        tools=backend_inputs["tools"],
        render_markdown=backend_inputs["render_markdown"],
    )
    result, updated_state = execute_backend_turn(
        generator=generator,
        inputs=inputs,
        state=state,
        last_saved_hyperparameters={},
        debug_path=debug_path,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        collect_memory_stats=_collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        run_backend_chat_fn=run_backend_chat,
    )
    _last_prompt_start_time = updated_state.last_prompt_start_time
    _generation_index = updated_state.generation_index
    return result


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
    try:
        return resolve_api_profile_paths(
            model=request.model,
            prompt_config=request.prompt_config,
            profile=request.profile,
            profiles_file=request.profiles_file,
            default_profiles_file=DEFAULT_PROFILES_FILE,
            loaded_model_path=_loaded_model_path,
            loaded_prompt_config_path=_generator_prompt_config_path,
            load_profiles_fn=load_profiles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_profile_from_args(
    *,
    model: Optional[str],
    profile: Optional[str],
    prompt_config: Optional[str],
    profiles_file: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        return resolve_startup_profile_paths(
            model=model,
            profile=profile,
            prompt_config=prompt_config,
            profiles_file=profiles_file,
            load_profiles_fn=load_profiles,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


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
    try:
        validate_chat_request_supported(request)
    except RequestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model_path, prompt_config_path, profile_data = _resolve_profile(request)

    generator = _get_generator(model_path, prompt_config_path)
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG))
    backend_inputs = _prepare_backend_inputs(
        request=request,
        generator=generator,
        model_path=model_path,
        profile_data=profile_data,
    )

    if request.stream:

        def generate_stream():
            global _last_prompt_start_time
            global _generation_index
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            backend_result = _execute_backend_turn(
                generator=generator,
                backend_inputs=backend_inputs,
                debug_path=debug_path,
            )
            text = backend_result.assistant_text or ""
            text, was_truncated = apply_stop_sequences(text, request.stop)
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
            finish_reason = "stop" if was_truncated else backend_result.finish_reason
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    backend_result = _execute_backend_turn(
        generator=generator,
        backend_inputs=backend_inputs,
        debug_path=debug_path,
    )

    assistant_text = backend_result.assistant_text or ""
    assistant_text, was_truncated = apply_stop_sequences(assistant_text, request.stop)
    analysis_text: str | None = None
    if request.return_analysis:
        analysis_text = backend_result.analysis_text

    prompt_tokens = backend_result.prompt_tokens
    completion_tokens = backend_result.completion_tokens
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = "stop" if was_truncated else backend_result.finish_reason
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
