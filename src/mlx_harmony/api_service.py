"""Shared API service functions used by HTTP and non-HTTP entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol

from fastapi.responses import JSONResponse, StreamingResponse

from mlx_harmony.api_contract import (
    ChatRequest,
    RequestValidationError,
    apply_stop_sequences,
    build_chat_completion_chunk,
    build_chat_completion_response,
    validate_chat_request_supported,
)
from mlx_harmony.backend_runtime import (
    BackendState,
    GeneratorRuntimeCache,
    TurnRuntimeState,
    collect_mlx_memory_stats,
    execute_backend_turn,
    prepare_backend_inputs,
)
from mlx_harmony.chat_utils import (
    get_turn_limits,
    resolve_api_profile_paths,
    resolve_startup_profile_paths,
)


class _WarnLogger(Protocol):
    """Minimal logger protocol for warning emission in API services."""

    def warning(self, msg: str, *args: Any) -> None:
        """Log warning-level messages."""


class _PreloadLogger(_WarnLogger, Protocol):
    """Logger protocol for preload flow."""

    def error(self, msg: str, *args: Any) -> None:
        """Log error-level messages."""


class _ServerRunner(Protocol):
    """Callable protocol for starting an ASGI server runtime."""

    def __call__(
        self,
        app: Any,
        *,
        host: str,
        port: int,
        log_level: str,
        reload: bool,
        workers: int,
    ) -> None:
        """Run an ASGI app with server settings."""


@dataclass(frozen=True)
class ServerStartupSettings:
    """Normalized server startup settings parsed from args/env."""

    host: str
    port: int
    log_level: str
    reload: bool
    workers: int
    profiles_file: str
    model: str | None
    profile: str | None
    prompt_config: str | None
    preload: bool


def parse_server_startup_settings(
    *,
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
    default_profiles_file: str,
) -> ServerStartupSettings:
    """Parse startup settings for the API server from CLI args and environment.

    Args:
        argv: Optional argument vector to parse. Uses process arguments when `None`.
        environ: Optional environment mapping to read defaults from.
        default_profiles_file: Default profiles file path.

    Returns:
        Parsed startup settings object.
    """

    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Run the MLX Harmony API server.")
    parser.add_argument(
        "--host",
        default=env.get("MLX_HARMONY_HOST", "0.0.0.0"),
        help="Host interface to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env.get("MLX_HARMONY_PORT", "8000")),
        help="Port to bind the server.",
    )
    parser.add_argument(
        "--log-level",
        default=env.get("MLX_HARMONY_LOG_LEVEL", "info"),
        help="Uvicorn log level (debug, info, warning, error).",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=env.get("MLX_HARMONY_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes (development only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(env.get("MLX_HARMONY_WORKERS", "1")),
        help="Number of worker processes (use 1 for in-process model cache).",
    )
    parser.add_argument(
        "--profiles-file",
        default=env.get("MLX_HARMONY_PROFILES_FILE", default_profiles_file),
        help="Profiles file path used for model selection.",
    )
    parser.add_argument(
        "--model",
        default=env.get("MLX_HARMONY_MODEL_PATH", None),
        help="Model path for preload (optional).",
    )
    parser.add_argument(
        "--profile",
        default=env.get("MLX_HARMONY_PROFILE", None),
        help="Profile name for preload (optional).",
    )
    parser.add_argument(
        "--prompt-config",
        default=env.get("MLX_HARMONY_PROMPT_CONFIG", None),
        help="Prompt config path for preload (optional).",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=env.get("MLX_HARMONY_PRELOAD", "false").lower() == "true",
        help="Preload model at startup (default: false).",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    return ServerStartupSettings(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        workers=args.workers,
        profiles_file=args.profiles_file,
        model=args.model,
        profile=args.profile,
        prompt_config=args.prompt_config,
        preload=args.preload,
    )


def collect_local_model_ids(
    *,
    models_dir: str,
    profiles_file: str,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
    logger: _WarnLogger | None = None,
) -> list[str]:
    """Collect model identifiers from local model directories or profiles file.

    Args:
        models_dir: Directory path containing local model subdirectories.
        profiles_file: Profiles file path used as fallback source.
        load_profiles_fn: Callable that loads profile mappings.
        logger: Optional logger to report profile load failures.

    Returns:
        Ordered model identifiers suitable for `/v1/models` responses.
    """

    model_ids: list[str] = []
    base = Path(models_dir)
    if base.exists():
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                model_ids.append(str(entry))
    if model_ids:
        return model_ids

    if not os.path.exists(profiles_file):
        return model_ids

    try:
        profiles = load_profiles_fn(profiles_file)
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("Failed to load profiles from %s: %s", profiles_file, exc)
        return model_ids

    for name, profile in profiles.items():
        model_ids.append(str(profile.get("model", name)))
    return model_ids


def collect_configured_model_ids(
    *,
    default_models_dir: str,
    default_profiles_file: str,
    environ: Mapping[str, str] | None = None,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
    logger: _WarnLogger | None = None,
) -> list[str]:
    """Collect model identifiers using configured environment defaults.

    Args:
        default_models_dir: Fallback models directory.
        default_profiles_file: Fallback profiles file.
        environ: Optional environment mapping for key lookups.
        load_profiles_fn: Callable that loads profile mappings.
        logger: Optional logger for profile-load warnings.

    Returns:
        Model identifiers to expose via `/v1/models`.
    """

    env = os.environ if environ is None else environ
    return collect_local_model_ids(
        models_dir=env.get("MLX_HARMONY_MODELS_DIR", default_models_dir),
        profiles_file=env.get("MLX_HARMONY_PROFILES_FILE", default_profiles_file),
        load_profiles_fn=load_profiles_fn,
        logger=logger,
    )


def resolve_request_profile_paths(
    *,
    model: str | None,
    prompt_config: str | None,
    profile: str | None,
    profiles_file: str | None,
    default_profiles_file: str,
    loaded_model_path: str | None,
    loaded_prompt_config_path: str | None,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
) -> tuple[str, str | None, dict[str, object] | None]:
    """Resolve request model/prompt profile paths for one chat completion call.

    Args:
        model: Request model id/path.
        prompt_config: Optional request prompt config path.
        profile: Optional request profile name.
        profiles_file: Optional request profiles file override.
        default_profiles_file: Default profiles file path.
        loaded_model_path: Already-loaded model path in server state.
        loaded_prompt_config_path: Already-loaded prompt config path in server state.
        load_profiles_fn: Callable used to load profile mappings.

    Returns:
        Tuple of `(model_path, prompt_config_path, profile_data)`.
    """

    return resolve_api_profile_paths(
        model=model,
        prompt_config=prompt_config,
        profile=profile,
        profiles_file=profiles_file,
        default_profiles_file=default_profiles_file,
        loaded_model_path=loaded_model_path,
        loaded_prompt_config_path=loaded_prompt_config_path,
        load_profiles_fn=load_profiles_fn,
    )


def resolve_startup_profile(
    *,
    model: str | None,
    profile: str | None,
    prompt_config: str | None,
    profiles_file: str,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
) -> tuple[str | None, str | None]:
    """Resolve startup model/prompt profile paths for optional preload.

    Args:
        model: Startup model id/path override.
        profile: Optional startup profile name.
        prompt_config: Optional startup prompt config path.
        profiles_file: Profiles file path to resolve profile names.
        load_profiles_fn: Callable used to load profile mappings.

    Returns:
        Tuple of `(model_path, prompt_config_path)`.
    """

    return resolve_startup_profile_paths(
        model=model,
        profile=profile,
        prompt_config=prompt_config,
        profiles_file=profiles_file,
        load_profiles_fn=load_profiles_fn,
    )


def preload_server_model_if_requested(
    *,
    settings: ServerStartupSettings,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
    resolve_startup_profile_fn: Callable[..., tuple[str | None, str | None]] = resolve_startup_profile,
    load_generator_fn: Callable[[str, str | None], Any],
    logger: _PreloadLogger,
) -> None:
    """Preload a model at startup when enabled in parsed settings.

    Args:
        settings: Parsed startup settings.
        load_profiles_fn: Callable used by profile resolution.
        resolve_startup_profile_fn: Startup profile resolver implementation.
        load_generator_fn: Callable that loads/caches the model generator.
        logger: Logger used for preload diagnostics.
    """

    if not settings.preload:
        return
    try:
        model_path, prompt_config_path = resolve_startup_profile_fn(
            model=settings.model,
            profile=settings.profile,
            prompt_config=settings.prompt_config,
            profiles_file=settings.profiles_file,
            load_profiles_fn=load_profiles_fn,
        )
    except ValueError as exc:
        logger.error("Failed to preload model: %s", exc)
        return

    if model_path:
        load_generator_fn(model_path, prompt_config_path)
        return
    logger.warning("Preload requested but no model/profile provided; skipping preload.")


def run_server_startup(
    *,
    app: Any,
    settings: ServerStartupSettings,
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
    load_generator_fn: Callable[[str, str | None], Any],
    logger: _PreloadLogger,
    run_server_fn: _ServerRunner,
    environ: MutableMapping[str, str] | None = None,
    resolve_startup_profile_fn: Callable[..., tuple[str | None, str | None]] = resolve_startup_profile,
) -> None:
    """Apply startup side effects and start the ASGI server runtime.

    Args:
        app: ASGI application instance.
        settings: Parsed startup settings.
        load_profiles_fn: Callable used by startup profile resolution.
        load_generator_fn: Callable used to preload the model when enabled.
        logger: Logger used for preload diagnostics.
        run_server_fn: Server runner implementation (for example `uvicorn.run`).
        environ: Optional environment mapping to update.
        resolve_startup_profile_fn: Startup profile resolver implementation.
    """

    env = os.environ if environ is None else environ
    env["MLX_HARMONY_PROFILES_FILE"] = settings.profiles_file
    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=load_profiles_fn,
        resolve_startup_profile_fn=resolve_startup_profile_fn,
        load_generator_fn=load_generator_fn,
        logger=logger,
    )
    run_server_fn(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
        workers=settings.workers,
    )


def run_server_backend_turn(
    *,
    request: ChatRequest,
    generator: Any,
    model_path: str,
    profile_data: dict[str, object] | None,
    state: BackendState,
    debug_path: Path,
    collect_memory: bool,
    last_saved_hyperparameters: dict[str, float | int | bool | str] | None,
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
    write_debug_metrics: Any,
    write_debug_response: Any,
    write_debug_info: Any,
    write_debug_token_texts: Any,
    write_debug_tokens: Any,
    run_backend_chat_fn: Any,
) -> tuple[Any, BackendState]:
    """Run one server-side generation turn via shared backend runtime helpers.

    Args:
        request: Parsed chat-completions request payload.
        generator: Loaded token generator.
        model_path: Resolved model path.
        profile_data: Optional profile metadata for request resolution.
        state: Current backend state snapshot.
        debug_path: Output path for debug artifacts.
        collect_memory: Whether MLX memory metrics should be collected.
        last_saved_hyperparameters: Previous hyperparameter snapshot.
        make_message_id: Callback for message id generation.
        make_timestamp: Callback for timestamp generation.
        write_debug_metrics: Debug metrics writer callback.
        write_debug_response: Debug response writer callback.
        write_debug_info: Debug metadata writer callback.
        write_debug_token_texts: Debug token text writer callback.
        write_debug_tokens: Debug token writer callback.
        run_backend_chat_fn: Shared backend chat execution function.

    Returns:
        Tuple of `(backend_result, updated_state)`.
    """

    backend_inputs = prepare_backend_inputs(
        request=request,
        generator=generator,
        model_path=model_path,
        profile_data=profile_data,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
    )
    max_tool_iterations, max_resume_attempts = get_turn_limits(
        getattr(generator, "prompt_config", None)
    )
    return execute_backend_turn(
        generator=generator,
        inputs=backend_inputs,
        state=state,
        last_saved_hyperparameters=last_saved_hyperparameters,
        debug_path=debug_path,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        collect_memory_stats=lambda: collect_mlx_memory_stats(enabled=collect_memory),
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        max_tool_iterations=max_tool_iterations,
        max_resume_attempts=max_resume_attempts,
        run_backend_chat_fn=run_backend_chat_fn,
    )


def finalize_backend_response_fields(
    *,
    assistant_text: str | None,
    analysis_text: str | None,
    finish_reason: str | None,
    stop: str | list[str] | None,
    include_analysis: bool,
) -> tuple[str, str | None, str | None]:
    """Finalize assistant/analysis/finish fields for API responses.

    Args:
        assistant_text: Raw assistant text from backend generation.
        analysis_text: Raw analysis text from backend generation.
        finish_reason: Raw finish reason from backend generation.
        stop: Optional stop sequence(s) from request payload.
        include_analysis: Whether analysis text should be returned.

    Returns:
        Tuple of ``(assistant_text, analysis_text, finish_reason)`` after
        applying stop truncation and analysis visibility policy.
    """
    resolved_text, was_truncated = apply_stop_sequences(assistant_text or "", stop)
    resolved_analysis = analysis_text if include_analysis else None
    resolved_finish_reason = "stop" if was_truncated else finish_reason
    return resolved_text, resolved_analysis, resolved_finish_reason


def build_non_stream_chat_response(
    *,
    backend_result: Any,
    request: ChatRequest,
    response_id: str,
    created: int,
    model_path: str,
    system_fingerprint: str,
) -> dict[str, Any]:
    """Build a non-stream chat completion payload from backend outputs.

    Args:
        backend_result: Backend generation result object.
        request: Parsed chat request payload.
        response_id: Stable response id for this request.
        created: Unix timestamp for response creation.
        model_path: Resolved model id/path for this request.
        system_fingerprint: Server fingerprint string.

    Returns:
        OpenAI-compatible non-stream chat completion payload.
    """
    assistant_text, analysis_text, finish_reason = finalize_backend_response_fields(
        assistant_text=backend_result.assistant_text,
        analysis_text=backend_result.analysis_text,
        finish_reason=backend_result.finish_reason,
        stop=request.stop,
        include_analysis=request.return_analysis,
    )
    return build_chat_completion_response(
        response_id=response_id,
        created=created,
        model=model_path,
        system_fingerprint=system_fingerprint,
        assistant_text=assistant_text,
        analysis_text=analysis_text,
        finish_reason=finish_reason,
        prompt_tokens=backend_result.prompt_tokens,
        completion_tokens=backend_result.completion_tokens,
    )


def build_streaming_chat_response(
    *,
    backend_result: Any,
    request: ChatRequest,
    response_id: str,
    created: int,
    model_path: str,
    system_fingerprint: str,
) -> list[str]:
    """Build SSE `data:` payload lines for a streaming chat completion turn.

    Args:
        backend_result: Backend generation result object.
        request: Parsed chat request payload.
        response_id: Stable response id for this request.
        created: Unix timestamp for response creation.
        model_path: Resolved model id/path for this request.
        system_fingerprint: Server fingerprint string.

    Returns:
        Ordered SSE payload lines ending with `[DONE]`.
    """
    lines: list[str] = []
    lines.append(
        "data: "
        + json.dumps(
            build_chat_completion_chunk(
                response_id=response_id,
                created=created,
                model=model_path,
                system_fingerprint=system_fingerprint,
                delta={"role": "assistant", "content": ""},
                finish_reason=None,
            )
        )
        + "\n\n"
    )
    text, _analysis_text, finish_reason = finalize_backend_response_fields(
        assistant_text=backend_result.assistant_text,
        analysis_text=backend_result.analysis_text,
        finish_reason=backend_result.finish_reason,
        stop=request.stop,
        include_analysis=False,
    )
    if text:
        lines.append(
            "data: "
            + json.dumps(
                build_chat_completion_chunk(
                    response_id=response_id,
                    created=created,
                    model=model_path,
                    system_fingerprint=system_fingerprint,
                    delta={"content": text},
                    finish_reason=None,
                )
            )
            + "\n\n"
        )
    lines.append(
        "data: "
        + json.dumps(
            build_chat_completion_chunk(
                response_id=response_id,
                created=created,
                model=model_path,
                system_fingerprint=system_fingerprint,
                delta={},
                finish_reason=finish_reason,
            )
        )
        + "\n\n"
    )
    lines.append("data: [DONE]\n\n")
    return lines


def handle_chat_completions_request(
    *,
    request: ChatRequest,
    default_profiles_file: str,
    default_server_debug_log: str,
    system_fingerprint: str,
    get_loaded_model_path: Callable[[], str | None],
    get_loaded_prompt_config_path: Callable[[], str | None],
    load_profiles_fn: Callable[[str], dict[str, dict[str, Any]]],
    get_generator_fn: Callable[[str, str | None], Any],
    read_turn_state: Callable[[], BackendState],
    write_turn_state: Callable[[BackendState], None],
    run_backend_chat_fn: Any,
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
    write_debug_metrics: Any,
    write_debug_response: Any,
    write_debug_info: Any,
    write_debug_token_texts: Any,
    write_debug_tokens: Any,
    invalid_request_response_builder: Callable[..., JSONResponse],
) -> JSONResponse | StreamingResponse:
    """Handle one OpenAI-compatible chat completions request.

    Args:
        request: Parsed chat request payload.
        default_profiles_file: Default profile mapping path.
        default_server_debug_log: Default debug artifact path.
        system_fingerprint: Fingerprint string returned in responses.
        get_loaded_model_path: Callback returning current loaded model path.
        get_loaded_prompt_config_path: Callback returning current prompt config path.
        load_profiles_fn: Profiles loader callable.
        get_generator_fn: Model generator resolver/cacher.
        read_turn_state: Callback returning current turn state snapshot.
        write_turn_state: Callback storing updated turn state snapshot.
        run_backend_chat_fn: Shared backend chat runner.
        make_message_id: Message id generator callback.
        make_timestamp: Timestamp generator callback.
        write_debug_metrics: Debug metrics writer callback.
        write_debug_response: Debug response writer callback.
        write_debug_info: Debug metadata writer callback.
        write_debug_token_texts: Debug token-text writer callback.
        write_debug_tokens: Debug token writer callback.
        invalid_request_response_builder: Response builder for 400 payloads.

    Returns:
        JSON or streaming response matching OpenAI chat-completions behavior.
    """

    try:
        validate_chat_request_supported(request)
    except RequestValidationError as exc:
        return invalid_request_response_builder(str(exc))
    try:
        model_path, prompt_config_path, profile_data = resolve_request_profile_paths(
            model=request.model,
            prompt_config=request.prompt_config,
            profile=request.profile,
            profiles_file=request.profiles_file,
            default_profiles_file=default_profiles_file,
            loaded_model_path=get_loaded_model_path(),
            loaded_prompt_config_path=get_loaded_prompt_config_path(),
            load_profiles_fn=load_profiles_fn,
        )
    except ValueError as exc:
        return invalid_request_response_builder(str(exc))

    generator = get_generator_fn(model_path, prompt_config_path)
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", default_server_debug_log))
    collect_memory = os.getenv("MLX_HARMONY_SERVER_COLLECT_MEMORY", "0") == "1"

    def _run_turn() -> Any:
        state = read_turn_state()
        result, updated_state = run_server_backend_turn(
            request=request,
            generator=generator,
            model_path=model_path,
            profile_data=profile_data,
            state=state,
            last_saved_hyperparameters={},
            debug_path=debug_path,
            collect_memory=collect_memory,
            make_message_id=make_message_id,
            make_timestamp=make_timestamp,
            write_debug_metrics=write_debug_metrics,
            write_debug_response=write_debug_response,
            write_debug_info=write_debug_info,
            write_debug_token_texts=write_debug_token_texts,
            write_debug_tokens=write_debug_tokens,
            run_backend_chat_fn=run_backend_chat_fn,
        )
        write_turn_state(updated_state)
        return result

    if request.stream:

        def generate_stream():
            backend_result = _run_turn()
            yield from build_streaming_chat_response(
                backend_result=backend_result,
                request=request,
                response_id=response_id,
                created=created,
                model_path=model_path,
                system_fingerprint=system_fingerprint,
            )

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    backend_result = _run_turn()
    return build_non_stream_chat_response(
        backend_result=backend_result,
        request=request,
        response_id=response_id,
        created=created,
        model_path=model_path,
        system_fingerprint=system_fingerprint,
    )
