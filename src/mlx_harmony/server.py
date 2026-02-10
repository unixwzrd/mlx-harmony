"""FastAPI server endpoints and shared runtime plumbing for MLX Harmony."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_harmony.api_contract import (
    ChatRequest,
    RequestValidationError,
    apply_stop_sequences,
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_health_response,
    build_models_list_response,
    build_openai_error_response,
    validate_chat_request_supported,
)
from mlx_harmony.api_service import (
    collect_configured_model_ids,
    collect_mlx_memory_stats,
    parse_server_startup_settings,
    preload_server_model_if_requested,
    resolve_request_profile_paths,
    resolve_startup_profile,
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
DEFAULT_SYSTEM_FINGERPRINT = "mlx-harmony-local"
PLACEHOLDER_POST_ENDPOINTS = [
    "/v1/completions",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/images/variations",
    "/v1/moderations",
    "/v1/files",
    "/v1/batches",
    "/v1/responses",
]


def _not_implemented(endpoint: str) -> JSONResponse:
    """Return a standardized 501 response for unimplemented API endpoints.

    Args:
        endpoint: Endpoint path identifier.

    Returns:
        JSON response payload with OpenAI-style error envelope.
    """

    return JSONResponse(
        status_code=501,
        content=build_openai_error_response(
            message=f"Endpoint '{endpoint}' is not implemented yet.",
            error_type="not_implemented_error",
            code="not_implemented",
        ),
    )


def _invalid_request(message: str, *, param: str | None = None) -> JSONResponse:
    """Return a standardized 400 invalid_request_error payload.

    Args:
        message: Human-readable validation or request resolution message.

    Returns:
        JSON response with OpenAI-style error envelope and 400 status.
    """

    return JSONResponse(
        status_code=400,
        content=build_openai_error_response(
            message=message,
            error_type="invalid_request_error",
            param=param,
            code=None,
        ),
    )


def _build_not_implemented_handler(endpoint: str) -> Callable[[], Awaitable[JSONResponse]]:
    """Build one async placeholder endpoint handler for an API route.

    Args:
        endpoint: Endpoint path to include in the error message.

    Returns:
        Async callable that returns standardized `501` payloads.
    """

    async def _handler() -> JSONResponse:
        return _not_implemented(endpoint)

    return _handler


def _register_placeholder_endpoints() -> None:
    """Register all unimplemented OpenAI-compatible POST endpoints."""

    for endpoint in PLACEHOLDER_POST_ENDPOINTS:
        route_name = endpoint.strip("/").replace("/", "_").replace("-", "_") + "_placeholder"
        app.add_api_route(
            endpoint,
            _build_not_implemented_handler(endpoint),
            methods=["POST"],
            name=route_name,
        )


@app.exception_handler(FastAPIRequestValidationError)
async def fastapi_validation_error_handler(
    _request: Any, exc: FastAPIRequestValidationError
) -> JSONResponse:
    """Normalize FastAPI 422 request validation failures into OpenAI 400 errors.

    Args:
        _request: Incoming request object (unused).
        exc: FastAPI request validation exception.

    Returns:
        OpenAI-style invalid request error response with HTTP 400.
    """

    errors = exc.errors()
    if not errors:
        return _invalid_request("Invalid request payload.")
    first = errors[0]
    message = str(first.get("msg") or "Invalid request payload.")
    loc = first.get("loc")
    param: str | None = None
    if isinstance(loc, (list, tuple)) and loc:
        # Keep only user-relevant field path (skip "body" root).
        parts = [str(part) for part in loc if str(part) != "body"]
        if parts:
            param = ".".join(parts)
    return _invalid_request(message, param=param)


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
        return _invalid_request(str(exc))
    try:
        model_path, prompt_config_path, profile_data = resolve_request_profile_paths(
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
        return _invalid_request(str(exc))

    generator = _get_generator(model_path, prompt_config_path)
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG))
    backend_inputs = prepare_backend_inputs(
        request=request,
        generator=generator,
        model_path=model_path,
        profile_data=profile_data,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
    )
    collect_memory = os.getenv("MLX_HARMONY_SERVER_COLLECT_MEMORY", "0") == "1"

    def _run_turn(inputs: BackendInputs) -> Any:
        """Execute one backend turn and update server turn state counters."""
        global _last_prompt_start_time
        global _generation_index
        state = BackendState(
            last_prompt_start_time=_last_prompt_start_time,
            generation_index=_generation_index,
        )
        result, updated_state = execute_backend_turn(
            generator=generator,
            inputs=inputs,
            state=state,
            last_saved_hyperparameters={},
            debug_path=debug_path,
            make_message_id=make_message_id,
            make_timestamp=make_timestamp,
            collect_memory_stats=lambda: collect_mlx_memory_stats(enabled=collect_memory),
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

    if request.stream:

        def generate_stream():
            global _last_prompt_start_time
            global _generation_index
            yield f"data: {json.dumps(build_chat_completion_chunk(response_id=response_id, created=created, model=model_path, system_fingerprint=DEFAULT_SYSTEM_FINGERPRINT, delta={'role': 'assistant', 'content': ''}, finish_reason=None))}\n\n"
            backend_result = _run_turn(backend_inputs)
            text = backend_result.assistant_text or ""
            text, was_truncated = apply_stop_sequences(text, request.stop)
            if text:
                chunk = build_chat_completion_chunk(
                    response_id=response_id,
                    created=created,
                    model=model_path,
                    system_fingerprint=DEFAULT_SYSTEM_FINGERPRINT,
                    delta={"content": text},
                    finish_reason=None,
                )
                yield f"data: {json.dumps(chunk)}\n\n"
            finish_reason = "stop" if was_truncated else backend_result.finish_reason
            yield f"data: {json.dumps(build_chat_completion_chunk(response_id=response_id, created=created, model=model_path, system_fingerprint=DEFAULT_SYSTEM_FINGERPRINT, delta={}, finish_reason=finish_reason))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    backend_result = _run_turn(backend_inputs)

    assistant_text = backend_result.assistant_text or ""
    assistant_text, was_truncated = apply_stop_sequences(assistant_text, request.stop)
    analysis_text: str | None = None
    if request.return_analysis:
        analysis_text = backend_result.analysis_text

    prompt_tokens = backend_result.prompt_tokens
    completion_tokens = backend_result.completion_tokens
    finish_reason = "stop" if was_truncated else backend_result.finish_reason
    return build_chat_completion_response(
        response_id=response_id,
        created=created,
        model=model_path,
        system_fingerprint=DEFAULT_SYSTEM_FINGERPRINT,
        assistant_text=assistant_text,
        analysis_text=analysis_text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@app.get("/v1/models")
async def list_models():
    """List available local models using the configured models/profiles.

    Returns:
        OpenAI-compatible list response for local models.
    """
    model_ids = collect_configured_model_ids(
        default_models_dir=DEFAULT_MODELS_DIR,
        default_profiles_file=DEFAULT_PROFILES_FILE,
        load_profiles_fn=load_profiles,
        logger=logger,
    )
    return build_models_list_response(model_ids=model_ids)


@app.get("/v1/health")
async def health_check() -> dict[str, str | bool | int | None]:
    """Report server liveness and model load state.

    Returns:
        Health metadata including model load state and paths.
    """
    model_loaded = _generator is not None
    return build_health_response(
        model_loaded=model_loaded,
        model_path=_loaded_model_path,
        prompt_config_path=_generator_prompt_config_path,
        loaded_at_unix=_loaded_at_unix,
    )


_register_placeholder_endpoints()


def main() -> None:
    settings = parse_server_startup_settings(
        default_profiles_file=DEFAULT_PROFILES_FILE
    )
    os.environ["MLX_HARMONY_PROFILES_FILE"] = settings.profiles_file
    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=load_profiles,
        resolve_startup_profile_fn=resolve_startup_profile,
        load_generator_fn=_get_generator,
        logger=logger,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()
