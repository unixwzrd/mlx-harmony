"""Unit tests for shared API service helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_harmony.api_service import (
    ServerStartupSettings,
    collect_configured_model_ids,
    collect_local_model_ids,
    collect_mlx_memory_stats,
    parse_server_startup_settings,
    preload_server_model_if_requested,
    resolve_request_profile_paths,
    resolve_startup_profile,
)


class _StubLogger:
    """Simple logger stub that records warning calls."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[Any, ...]]] = []

    def warning(self, msg: str, *args: Any) -> None:
        """Record warning call arguments."""

        self.messages.append((msg, args))

    def error(self, msg: str, *args: Any) -> None:
        """Record error call arguments."""

        self.messages.append((msg, args))


def test_collect_local_model_ids_prefers_models_directory(tmp_path: Path) -> None:
    """Return model ids from models directory before reading profiles file."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "a").mkdir()
    (models_dir / "b").mkdir()
    profiles_file = tmp_path / "profiles.json"
    profiles_file.write_text("{}", encoding="utf-8")

    calls: list[str] = []

    def _load_profiles(_path: str) -> dict[str, dict[str, Any]]:
        calls.append(_path)
        return {"unused": {"model": "unused"}}

    model_ids = collect_local_model_ids(
        models_dir=str(models_dir),
        profiles_file=str(profiles_file),
        load_profiles_fn=_load_profiles,
    )
    assert str(models_dir / "a") in model_ids
    assert str(models_dir / "b") in model_ids
    assert calls == []


def test_collect_local_model_ids_falls_back_to_profiles(tmp_path: Path) -> None:
    """Return model ids from profiles file when models directory is empty."""

    profiles_file = tmp_path / "profiles.json"
    profiles_file.write_text("{}", encoding="utf-8")

    def _load_profiles(_path: str) -> dict[str, dict[str, Any]]:
        return {
            "default": {"model": "models/default"},
            "fallback": {},
        }

    model_ids = collect_local_model_ids(
        models_dir=str(tmp_path / "missing-models"),
        profiles_file=str(profiles_file),
        load_profiles_fn=_load_profiles,
    )
    assert "models/default" in model_ids
    assert "fallback" in model_ids


def test_collect_local_model_ids_logs_profile_load_failure(tmp_path: Path) -> None:
    """Log profile load errors and return an empty model list."""

    profiles_file = tmp_path / "profiles.json"
    profiles_file.write_text("{}", encoding="utf-8")
    logger = _StubLogger()

    def _load_profiles(_path: str) -> dict[str, dict[str, Any]]:
        raise RuntimeError("boom")

    model_ids = collect_local_model_ids(
        models_dir=str(tmp_path / "missing-models"),
        profiles_file=str(profiles_file),
        load_profiles_fn=_load_profiles,
        logger=logger,
    )
    assert model_ids == []
    assert logger.messages


def test_collect_configured_model_ids_reads_environment_overrides(tmp_path: Path) -> None:
    """Read models/profiles sources from environment-style mappings."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1").mkdir()
    env = {
        "MLX_HARMONY_MODELS_DIR": str(models_dir),
        "MLX_HARMONY_PROFILES_FILE": str(tmp_path / "profiles.json"),
    }

    model_ids = collect_configured_model_ids(
        default_models_dir="models",
        default_profiles_file="configs/profiles.example.json",
        environ=env,
        load_profiles_fn=lambda _path: {},
    )
    assert model_ids == [str(models_dir / "m1")]


def test_collect_configured_model_ids_uses_defaults_when_env_missing(tmp_path: Path) -> None:
    """Use provided defaults when environment variables are not set."""

    models_dir = tmp_path / "default-models"
    models_dir.mkdir()
    (models_dir / "m2").mkdir()

    model_ids = collect_configured_model_ids(
        default_models_dir=str(models_dir),
        default_profiles_file="configs/profiles.example.json",
        environ={},
        load_profiles_fn=lambda _path: {},
    )
    assert model_ids == [str(models_dir / "m2")]


def test_resolve_request_profile_paths_delegates_to_chat_utils(monkeypatch: Any) -> None:
    """Delegate request profile resolution through shared chat-utils helper."""

    expected = ("models/test", "configs/test.json", {"name": "p"})

    def _fake_resolve_api_profile_paths(**kwargs: Any) -> tuple[str, str | None, dict[str, object]]:
        assert kwargs["model"] == "models/test"
        assert kwargs["profile"] == "default"
        return expected

    monkeypatch.setattr(
        "mlx_harmony.api_service.resolve_api_profile_paths",
        _fake_resolve_api_profile_paths,
    )

    actual = resolve_request_profile_paths(
        model="models/test",
        prompt_config=None,
        profile="default",
        profiles_file=None,
        default_profiles_file="configs/profiles.json",
        loaded_model_path=None,
        loaded_prompt_config_path=None,
        load_profiles_fn=lambda _path: {},
    )
    assert actual == expected


def test_resolve_startup_profile_delegates_to_chat_utils(monkeypatch: Any) -> None:
    """Delegate startup profile resolution through shared chat-utils helper."""

    expected = ("models/startup", "configs/startup.json")

    def _fake_resolve_startup_profile_paths(**kwargs: Any) -> tuple[str | None, str | None]:
        assert kwargs["profile"] == "startup"
        assert kwargs["profiles_file"] == "configs/profiles.json"
        return expected

    monkeypatch.setattr(
        "mlx_harmony.api_service.resolve_startup_profile_paths",
        _fake_resolve_startup_profile_paths,
    )

    actual = resolve_startup_profile(
        model=None,
        profile="startup",
        prompt_config=None,
        profiles_file="configs/profiles.json",
        load_profiles_fn=lambda _path: {},
    )
    assert actual == expected


def test_parse_server_startup_settings_reads_env_defaults() -> None:
    """Use environment defaults when argv overrides are not provided."""

    env = {
        "MLX_HARMONY_HOST": "127.0.0.1",
        "MLX_HARMONY_PORT": "9000",
        "MLX_HARMONY_LOG_LEVEL": "debug",
        "MLX_HARMONY_RELOAD": "true",
        "MLX_HARMONY_WORKERS": "2",
        "MLX_HARMONY_PROFILES_FILE": "configs/profiles.json",
        "MLX_HARMONY_MODEL_PATH": "models/test",
        "MLX_HARMONY_PROFILE": "default",
        "MLX_HARMONY_PROMPT_CONFIG": "configs/test.json",
        "MLX_HARMONY_PRELOAD": "true",
    }
    settings = parse_server_startup_settings(
        argv=[],
        environ=env,
        default_profiles_file="configs/profiles.example.json",
    )
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.log_level == "debug"
    assert settings.reload is True
    assert settings.workers == 2
    assert settings.profiles_file == "configs/profiles.json"
    assert settings.model == "models/test"
    assert settings.profile == "default"
    assert settings.prompt_config == "configs/test.json"
    assert settings.preload is True


def test_parse_server_startup_settings_applies_argv_overrides() -> None:
    """Allow CLI arguments to override environment defaults."""

    settings = parse_server_startup_settings(
        argv=["--host", "0.0.0.0", "--port", "8100", "--no-reload", "--workers", "3"],
        environ={},
        default_profiles_file="configs/profiles.example.json",
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 8100
    assert settings.reload is False
    assert settings.workers == 3
    assert settings.profiles_file == "configs/profiles.example.json"


def test_preload_server_model_if_requested_noop_when_disabled() -> None:
    """Do nothing when preload flag is disabled."""

    settings = ServerStartupSettings(
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
        profiles_file="configs/profiles.json",
        model=None,
        profile=None,
        prompt_config=None,
        preload=False,
    )
    logger = _StubLogger()
    called: list[str] = []

    def _load_generator(_model: str, _prompt: str | None) -> None:
        called.append("load")

    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=lambda _path: {},
        load_generator_fn=_load_generator,
        logger=logger,
    )
    assert called == []
    assert logger.messages == []


def test_preload_server_model_if_requested_loads_generator_when_resolved() -> None:
    """Load generator when startup profile resolution returns a model path."""

    settings = ServerStartupSettings(
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
        profiles_file="configs/profiles.json",
        model="models/test",
        profile=None,
        prompt_config=None,
        preload=True,
    )
    logger = _StubLogger()
    loaded: list[tuple[str, str | None]] = []

    def _resolve_startup_profile(**_kwargs: Any) -> tuple[str | None, str | None]:
        return ("models/test", "configs/test.json")

    def _load_generator(model: str, prompt: str | None) -> None:
        loaded.append((model, prompt))

    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=lambda _path: {},
        resolve_startup_profile_fn=_resolve_startup_profile,
        load_generator_fn=_load_generator,
        logger=logger,
    )
    assert loaded == [("models/test", "configs/test.json")]


def test_preload_server_model_if_requested_logs_errors_and_warnings() -> None:
    """Log preload failures and unresolved preload targets."""

    settings = ServerStartupSettings(
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
        profiles_file="configs/profiles.json",
        model=None,
        profile="default",
        prompt_config=None,
        preload=True,
    )
    logger = _StubLogger()

    def _failing_resolver(**_kwargs: Any) -> tuple[str | None, str | None]:
        raise ValueError("bad profile")

    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=lambda _path: {},
        resolve_startup_profile_fn=_failing_resolver,
        load_generator_fn=lambda _model, _prompt: None,
        logger=logger,
    )
    assert any("Failed to preload model" in msg for msg, _ in logger.messages)

    logger.messages.clear()

    def _empty_resolver(**_kwargs: Any) -> tuple[str | None, str | None]:
        return (None, None)

    preload_server_model_if_requested(
        settings=settings,
        load_profiles_fn=lambda _path: {},
        resolve_startup_profile_fn=_empty_resolver,
        load_generator_fn=lambda _model, _prompt: None,
        logger=logger,
    )
    assert any("Preload requested but no model/profile provided" in msg for msg, _ in logger.messages)


def test_collect_mlx_memory_stats_disabled_returns_empty() -> None:
    """Return empty stats when collection is disabled."""

    assert collect_mlx_memory_stats(enabled=False) == {}
