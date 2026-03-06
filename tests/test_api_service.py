"""Unit tests for shared API service helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mlx_harmony.api_service import (
    ServerStartupSettings,
    build_non_stream_chat_response,
    build_streaming_chat_response,
    collect_configured_model_ids,
    collect_local_model_ids,
    finalize_backend_response_fields,
    parse_server_startup_settings,
    preload_server_model_if_requested,
    resolve_request_profile_paths,
    resolve_startup_profile,
    run_server_backend_turn,
    run_server_startup,
)
from mlx_harmony.backend_runtime import (
    BackendState,
    GeneratorRuntimeCache,
    TurnRuntimeState,
    collect_mlx_memory_stats,
    load_runtime_generator,
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

    def info(self, msg: str, *args: Any) -> None:
        """Record info call arguments."""

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


def test_generator_runtime_cache_reuses_generator(monkeypatch: Any) -> None:
    """Reuse cached generator for identical model and prompt-config path."""

    init_calls: list[tuple[str, str | None]] = []

    def _fake_initialize_generator(**kwargs: Any) -> Any:
        init_calls.append((kwargs["model_path"], kwargs["prompt_config_path"]))
        return SimpleNamespace(model_path=kwargs["model_path"])

    monkeypatch.setattr("mlx_harmony.backend_runtime.load_prompt_config", lambda _path: None)
    monkeypatch.setattr(
        "mlx_harmony.backend_runtime.initialize_generator",
        _fake_initialize_generator,
    )

    cache = GeneratorRuntimeCache(logger=_StubLogger())
    first = cache.get_generator("models/a", None)
    second = cache.get_generator("models/a", None)

    assert first is second
    assert init_calls == [("models/a", None)]
    assert cache.is_model_loaded() is True
    assert cache.get_loaded_model_path() == "models/a"
    assert cache.get_loaded_prompt_config_path() is None
    assert cache.get_loaded_at_unix() is not None


def test_generator_runtime_cache_reloads_for_new_prompt_config(monkeypatch: Any) -> None:
    """Reload generator cache when prompt-config path changes."""

    init_calls: list[tuple[str, str | None]] = []

    def _fake_initialize_generator(**kwargs: Any) -> Any:
        init_calls.append((kwargs["model_path"], kwargs["prompt_config_path"]))
        return SimpleNamespace(model_path=kwargs["model_path"])

    monkeypatch.setattr("mlx_harmony.backend_runtime.load_prompt_config", lambda _path: None)
    monkeypatch.setattr(
        "mlx_harmony.backend_runtime.initialize_generator",
        _fake_initialize_generator,
    )

    cache = GeneratorRuntimeCache(logger=_StubLogger())
    cache.get_generator("models/a", "configs/one.json")
    cache.get_generator("models/a", "configs/two.json")

    assert init_calls == [
        ("models/a", "configs/one.json"),
        ("models/a", "configs/two.json"),
    ]
    assert cache.get_loaded_prompt_config_path() == "configs/two.json"


def test_turn_runtime_state_defaults_and_updates() -> None:
    """Store and retrieve backend turn runtime state snapshots."""

    state = TurnRuntimeState()
    initial = state.read()
    assert initial.last_prompt_start_time is None
    assert initial.generation_index == 0

    updated = BackendState(last_prompt_start_time=3.5, generation_index=7)
    state.write(updated)
    final = state.read()
    assert final.last_prompt_start_time == 3.5
    assert final.generation_index == 7


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


def test_load_runtime_generator_uses_prompt_config_mlock(monkeypatch: Any) -> None:
    """Use prompt-config mlock by default when explicit override is not passed."""
    init_calls: list[dict[str, Any]] = []

    def _fake_initialize_generator(**kwargs: Any) -> Any:
        init_calls.append(kwargs)
        return SimpleNamespace(model_path=kwargs["model_path"])

    prompt_cfg = SimpleNamespace(mlock=True)
    monkeypatch.setattr(
        "mlx_harmony.backend_runtime.initialize_generator",
        _fake_initialize_generator,
    )

    generator = load_runtime_generator(
        model_path="models/test",
        prompt_config_path="configs/test.json",
        prompt_config=prompt_cfg,
    )
    assert generator.model_path == "models/test"
    assert init_calls[0]["mlock"] is True
    assert init_calls[0]["lazy"] is False


def test_load_runtime_generator_prefers_explicit_mlock_override(monkeypatch: Any) -> None:
    """Use explicit mlock override when provided by caller."""
    init_calls: list[dict[str, Any]] = []

    def _fake_initialize_generator(**kwargs: Any) -> Any:
        init_calls.append(kwargs)
        return SimpleNamespace(model_path=kwargs["model_path"])

    prompt_cfg = SimpleNamespace(mlock=True)
    monkeypatch.setattr(
        "mlx_harmony.backend_runtime.initialize_generator",
        _fake_initialize_generator,
    )

    load_runtime_generator(
        model_path="models/test",
        prompt_config_path="configs/test.json",
        prompt_config=prompt_cfg,
        mlock=False,
    )
    assert init_calls[0]["mlock"] is False


def test_run_server_startup_sets_profiles_env_and_runs_server() -> None:
    """Set profiles env key and invoke server runner with parsed settings."""

    settings = ServerStartupSettings(
        host="127.0.0.1",
        port=8001,
        log_level="debug",
        reload=False,
        workers=1,
        profiles_file="configs/profiles.test.json",
        model=None,
        profile=None,
        prompt_config=None,
        preload=False,
    )
    env: dict[str, str] = {}
    run_calls: list[tuple[Any, dict[str, Any]]] = []

    def _run_server(app: Any, **kwargs: Any) -> None:
        run_calls.append((app, kwargs))

    run_server_startup(
        app="app-object",
        settings=settings,
        load_profiles_fn=lambda _path: {},
        load_generator_fn=lambda _model, _prompt: None,
        logger=_StubLogger(),
        run_server_fn=_run_server,
        environ=env,
    )

    assert env["MLX_HARMONY_PROFILES_FILE"] == "configs/profiles.test.json"
    assert run_calls == [
        (
            "app-object",
            {
                "host": "127.0.0.1",
                "port": 8001,
                "log_level": "debug",
                "reload": False,
                "workers": 1,
            },
        )
    ]


def test_run_server_startup_triggers_preload_when_enabled() -> None:
    """Call model preload path before invoking server runner."""

    settings = ServerStartupSettings(
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
        profiles_file="configs/profiles.json",
        model="models/test",
        profile=None,
        prompt_config="configs/prompt.json",
        preload=True,
    )
    loaded: list[tuple[str, str | None]] = []

    def _resolve_startup_profile(**_kwargs: Any) -> tuple[str | None, str | None]:
        return ("models/test", "configs/prompt.json")

    def _load_generator(model: str, prompt: str | None) -> None:
        loaded.append((model, prompt))

    run_server_startup(
        app="app-object",
        settings=settings,
        load_profiles_fn=lambda _path: {},
        load_generator_fn=_load_generator,
        logger=_StubLogger(),
        run_server_fn=lambda _app, **_kwargs: None,
        environ={},
        resolve_startup_profile_fn=_resolve_startup_profile,
    )

    assert loaded == [("models/test", "configs/prompt.json")]


def test_run_server_backend_turn_delegates_prepare_and_execute(monkeypatch: Any) -> None:
    """Wire backend-turn helpers with prepared inputs and memory callback."""

    prepared_inputs = {"conversation": [], "hyperparameters": {}}
    captured_execute_kwargs: dict[str, Any] = {}

    def _fake_prepare_backend_inputs(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["model_path"] == "models/test"
        return prepared_inputs

    def _fake_execute_backend_turn(**kwargs: Any) -> tuple[str, Any]:
        captured_execute_kwargs.update(kwargs)
        return ("backend-result", "updated-state")

    monkeypatch.setattr(
        "mlx_harmony.api_service.prepare_backend_inputs",
        _fake_prepare_backend_inputs,
    )
    monkeypatch.setattr(
        "mlx_harmony.api_service.execute_backend_turn",
        _fake_execute_backend_turn,
    )

    result, state = run_server_backend_turn(
        request=type("R", (), {"messages": []})(),
        generator=object(),
        model_path="models/test",
        profile_data=None,
        state="state",
        debug_path=Path("logs/debug.log"),
        collect_memory=False,
        last_saved_hyperparameters={},
        make_message_id=lambda: "m",
        make_timestamp=lambda: "t",
        write_debug_metrics=None,
        write_debug_response=None,
        write_debug_info=None,
        write_debug_token_texts=None,
        write_debug_tokens=None,
        run_backend_chat_fn=None,
    )
    assert result == "backend-result"
    assert state == "updated-state"
    assert captured_execute_kwargs["inputs"] is prepared_inputs
    assert captured_execute_kwargs["state"] == "state"
    assert captured_execute_kwargs["collect_memory_stats"]() == {}
    assert captured_execute_kwargs["max_tool_iterations"] == 10
    assert captured_execute_kwargs["max_resume_attempts"] == 2


def test_finalize_backend_response_fields_applies_stop_and_finish_reason() -> None:
    """Apply stop truncation and map finish reason to `stop` when truncated."""
    assistant_text, analysis_text, finish_reason = finalize_backend_response_fields(
        assistant_text="hello<END>ignored",
        analysis_text="thinking",
        finish_reason="length",
        stop="<END>",
        include_analysis=True,
    )
    assert assistant_text == "hello"
    assert analysis_text == "thinking"
    assert finish_reason == "stop"


def test_finalize_backend_response_fields_omits_analysis_when_disabled() -> None:
    """Hide analysis text when analysis channel output is disabled."""
    assistant_text, analysis_text, finish_reason = finalize_backend_response_fields(
        assistant_text="hello",
        analysis_text="thinking",
        finish_reason="stop",
        stop=None,
        include_analysis=False,
    )
    assert assistant_text == "hello"
    assert analysis_text is None
    assert finish_reason == "stop"


def test_build_non_stream_chat_response_applies_stop_and_analysis_policy() -> None:
    """Build non-stream payload with shared response postprocessing rules."""
    request = SimpleNamespace(stop="<STOP>", return_analysis=False)
    backend_result = SimpleNamespace(
        assistant_text="hello<STOP>ignored",
        analysis_text="hidden analysis",
        finish_reason="length",
        prompt_tokens=12,
        completion_tokens=8,
    )

    response = build_non_stream_chat_response(
        backend_result=backend_result,
        request=request,
        response_id="chatcmpl-test",
        created=123,
        model_path="models/test",
        system_fingerprint="fp-test",
    )

    assert response["choices"][0]["message"]["content"] == "hello"
    assert "analysis" not in response["choices"][0]["message"]
    assert response["choices"][0]["finish_reason"] == "stop"


def test_build_streaming_chat_response_emits_done_and_final_finish_reason() -> None:
    """Build streaming payload lines with final stop override when truncated."""
    request = SimpleNamespace(stop="<END>", return_analysis=True)
    backend_result = SimpleNamespace(
        assistant_text="answer<END>rest",
        analysis_text="analysis",
        finish_reason="length",
        prompt_tokens=9,
        completion_tokens=3,
    )

    lines = build_streaming_chat_response(
        backend_result=backend_result,
        request=request,
        response_id="chatcmpl-stream",
        created=456,
        model_path="models/test",
        system_fingerprint="fp-stream",
    )

    assert lines[-1] == "data: [DONE]\n\n"
    assert len(lines) == 4

    first_chunk = lines[0].removeprefix("data: ").strip()
    middle_chunk = lines[1].removeprefix("data: ").strip()
    final_chunk = lines[2].removeprefix("data: ").strip()
    first_data = json.loads(first_chunk)
    middle_data = json.loads(middle_chunk)
    final_data = json.loads(final_chunk)

    assert first_data["choices"][0]["delta"]["role"] == "assistant"
    assert middle_data["choices"][0]["delta"]["content"] == "answer"
    assert final_data["choices"][0]["finish_reason"] == "stop"
