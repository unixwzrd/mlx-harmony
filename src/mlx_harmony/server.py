"""FastAPI server endpoints and shared runtime plumbing for MLX Harmony."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_harmony.api_contract import (
    ChatRequest,
)
from mlx_harmony.api_routes import (
    build_invalid_request_response,
    register_chat_completions_route,
    register_health_route,
    register_models_route,
    register_placeholder_post_endpoints,
    register_request_validation_handler,
)
from mlx_harmony.api_service import (
    GeneratorRuntimeCache,
    TurnRuntimeState,
    collect_configured_model_ids,
    handle_chat_completions_request,
    parse_server_startup_settings,
    resolve_startup_profile,
    run_server_startup,
)
from mlx_harmony.backend_api import run_backend_chat
from mlx_harmony.chat_history import (
    make_message_id,
    make_timestamp,
    write_debug_info,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.config import load_profiles
from mlx_harmony.logging import get_logger

app = FastAPI(title="MLX Harmony API")
logger = get_logger(__name__)

_generator_runtime = GeneratorRuntimeCache(logger=logger)
_turn_runtime = TurnRuntimeState()

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

def _get_generator(model: str, prompt_config_path: str | None):
    """Load or reuse a cached generator using the shared CLI init path.

    Args:
        model: Model path to load.
        prompt_config_path: Prompt config file path, if any.

    Returns:
        Loaded TokenGenerator instance.
    """
    return _generator_runtime.get_generator(model, prompt_config_path)


async def _chat_completions_handler(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    return handle_chat_completions_request(
        request=request,
        default_profiles_file=DEFAULT_PROFILES_FILE,
        default_server_debug_log=DEFAULT_SERVER_DEBUG_LOG,
        system_fingerprint=DEFAULT_SYSTEM_FINGERPRINT,
        get_loaded_model_path=_generator_runtime.get_loaded_model_path,
        get_loaded_prompt_config_path=_generator_runtime.get_loaded_prompt_config_path,
        load_profiles_fn=load_profiles,
        get_generator_fn=_get_generator,
        read_turn_state=_turn_runtime.read,
        write_turn_state=_turn_runtime.write,
        run_backend_chat_fn=run_backend_chat,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        invalid_request_response_builder=build_invalid_request_response,
    )

register_chat_completions_route(app=app, handler=_chat_completions_handler)
register_models_route(
    app=app,
    get_model_ids=lambda: collect_configured_model_ids(
        default_models_dir=DEFAULT_MODELS_DIR,
        default_profiles_file=DEFAULT_PROFILES_FILE,
        load_profiles_fn=load_profiles,
        logger=logger,
    ),
)
register_health_route(
    app=app,
    get_model_loaded=_generator_runtime.is_model_loaded,
    get_model_path=_generator_runtime.get_loaded_model_path,
    get_prompt_config_path=_generator_runtime.get_loaded_prompt_config_path,
    get_loaded_at_unix=_generator_runtime.get_loaded_at_unix,
)
register_request_validation_handler(
    app=app,
    invalid_request_response_builder=build_invalid_request_response,
)
register_placeholder_post_endpoints(app=app, endpoints=PLACEHOLDER_POST_ENDPOINTS)


def main() -> None:
    settings = parse_server_startup_settings(
        default_profiles_file=DEFAULT_PROFILES_FILE
    )
    import uvicorn

    run_server_startup(
        app=app,
        settings=settings,
        load_profiles_fn=load_profiles,
        load_generator_fn=_get_generator,
        logger=logger,
        run_server_fn=uvicorn.run,
        resolve_startup_profile_fn=resolve_startup_profile,
    )


if __name__ == "__main__":
    main()
