"""Shared API service functions used by HTTP and non-HTTP entrypoints."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from mlx_harmony.chat_utils import (
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


def collect_mlx_memory_stats(*, enabled: bool) -> dict[str, float | int | str]:
    """Collect MLX Metal device stats when enabled.

    Args:
        enabled: Whether memory collection is enabled for this run.

    Returns:
        Flat memory stats dict suitable for debug artifacts.
    """

    if not enabled:
        return {}
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, float | int | str] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats
