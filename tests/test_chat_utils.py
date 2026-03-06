"""Unit tests for shared chat utility helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_harmony.chat_utils import (
    build_hyperparameters,
    build_hyperparameters_from_request,
    get_turn_limits,
    resolve_api_profile_paths,
    resolve_startup_profile_paths,
    sampling_fields_from_hyperparameters,
    transport_fields_from_hyperparameters,
)


def test_resolve_api_profile_paths_uses_loaded_defaults() -> None:
    """API resolver should fall back to already-loaded server defaults."""
    model_path, prompt_path, profile_data = resolve_api_profile_paths(
        model=None,
        prompt_config=None,
        profile=None,
        profiles_file=None,
        default_profiles_file="configs/profiles.example.json",
        loaded_model_path="models/test-model",
        loaded_prompt_config_path="configs/test.json",
    )
    assert model_path == "models/test-model"
    assert prompt_path == "configs/test.json"
    assert profile_data is None


def test_resolve_api_profile_paths_errors_when_no_model() -> None:
    """API resolver should reject requests with no model and no loaded default."""
    with pytest.raises(ValueError, match="No model provided"):
        resolve_api_profile_paths(
            model=None,
            prompt_config=None,
            profile=None,
            profiles_file=None,
            default_profiles_file="configs/profiles.example.json",
            loaded_model_path=None,
            loaded_prompt_config_path=None,
        )


def test_resolve_startup_profile_paths_with_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup resolver should map profile to model and prompt config path."""
    monkeypatch.setenv("MLX_HARMONY_CONFIG_DIR", str(tmp_path))
    profiles = tmp_path / "profiles.json"
    prompt_cfg = tmp_path / "prompt.json"
    prompt_cfg.write_text("{}", encoding="utf-8")
    profiles.write_text(
        (
            '{\n'
            '  "demo": {\n'
            '    "model": "models/demo",\n'
            '    "prompt_config": "prompt.json"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    model_path, prompt_path = resolve_startup_profile_paths(
        model=None,
        profile="demo",
        prompt_config=None,
        profiles_file=str(profiles),
    )
    assert model_path == "models/demo"
    assert prompt_path == str(prompt_cfg)


def test_sampling_fields_from_hyperparameters_normalizes_types() -> None:
    """Sampling fields should normalize numeric values from shared hyperparameters."""
    sampling = sampling_fields_from_hyperparameters(
        {
            "temperature": "0.75",
            "max_tokens": "256",
            "top_p": 0.9,
            "min_p": "0.05",
            "top_k": "40",
            "repetition_penalty": "1.1",
            "repetition_context_size": "32",
            "xtc_probability": "0.25",
            "xtc_threshold": "0.1",
            "seed": "1234",
        }
    )
    assert sampling == {
        "temperature": 0.75,
        "max_tokens": 256,
        "top_p": 0.9,
        "min_p": 0.05,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "repetition_context_size": 32,
        "xtc_probability": 0.25,
        "xtc_threshold": 0.1,
        "seed": 1234,
    }


def test_sampling_fields_from_hyperparameters_handles_missing_values() -> None:
    """Sampling fields should remain unset when values are not provided."""
    sampling = sampling_fields_from_hyperparameters({})
    assert sampling == {
        "temperature": None,
        "max_tokens": None,
        "top_p": None,
        "min_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "repetition_context_size": None,
        "xtc_probability": None,
        "xtc_threshold": None,
        "seed": None,
    }


def test_transport_fields_from_hyperparameters_normalizes_transport_values() -> None:
    """Transport fields should normalize shared runtime values consistently."""
    transport = transport_fields_from_hyperparameters(
        {
            "temperature": "0.7",
            "max_tokens": "256",
            "top_p": "0.9",
            "min_p": "0.05",
            "top_k": "40",
            "repetition_penalty": "1.1",
            "repetition_context_size": "20",
            "xtc_probability": "0.25",
            "xtc_threshold": "0.1",
            "seed": "1234",
            "loop_detection": "full",
            "reseed_each_turn": "true",
        }
    )
    assert transport == {
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9,
        "min_p": 0.05,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "repetition_context_size": 20,
        "xtc_probability": 0.25,
        "xtc_threshold": 0.1,
        "seed": 1234,
        "loop_detection": "full",
        "reseed_each_turn": True,
    }


def test_transport_fields_from_hyperparameters_handles_missing_values() -> None:
    """Transport fields should remain unset when no values are provided."""
    transport = transport_fields_from_hyperparameters({})
    assert transport == {
        "temperature": None,
        "max_tokens": None,
        "top_p": None,
        "min_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "repetition_context_size": None,
        "xtc_probability": None,
        "xtc_threshold": None,
        "seed": None,
        "loop_detection": None,
        "reseed_each_turn": None,
    }


def test_build_hyperparameters_preserves_falsy_loaded_values() -> None:
    """Loaded zero/false values should not be dropped during merge."""
    args = SimpleNamespace(
        max_tokens=None,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=None,
        repetition_penalty=None,
        repetition_context_size=None,
        xtc_probability=None,
        xtc_threshold=None,
        seed=None,
        loop_detection=None,
        reseed_each_turn=None,
    )
    loaded = {
        "temperature": 0.0,
        "top_k": 0,
        "xtc_probability": 0.0,
        "reseed_each_turn": False,
    }
    merged = build_hyperparameters(
        args=args,
        loaded_hyperparameters=loaded,
        prompt_config=None,
        is_harmony=False,
    )
    assert merged["temperature"] == 0.0
    assert merged["top_k"] == 0
    assert merged["xtc_probability"] == 0.0
    assert merged["reseed_each_turn"] is False


def test_build_hyperparameters_from_request_uses_request_values() -> None:
    """Request-provided values should override config defaults."""
    request = SimpleNamespace(
        max_tokens=128,
        temperature=0.2,
        top_p=0.8,
        min_p=0.05,
        top_k=30,
        repetition_penalty=1.05,
        repetition_context_size=24,
        xtc_probability=0.2,
        xtc_threshold=0.1,
        loop_detection="full",
        seed=77,
        reseed_each_turn=True,
    )
    cfg = SimpleNamespace(
        max_tokens=512,
        temperature=0.9,
        top_p=0.95,
        min_p=0.0,
        top_k=40,
        repetition_penalty=1.1,
        repetition_context_size=20,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        loop_detection="cheap",
        seed=-1,
        reseed_each_turn=False,
    )
    merged = build_hyperparameters_from_request(
        request=request,
        prompt_config=cfg,
        is_harmony=False,
    )
    assert merged == {
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.8,
        "min_p": 0.05,
        "top_k": 30,
        "repetition_penalty": 1.05,
        "repetition_context_size": 24,
        "xtc_probability": 0.2,
        "xtc_threshold": 0.1,
        "seed": 77,
        "loop_detection": "full",
        "reseed_each_turn": True,
    }


def test_get_turn_limits_defaults_when_unset() -> None:
    """Turn limits should use shared defaults when config omits values."""
    assert get_turn_limits(None) == (10, 2)
    assert get_turn_limits(SimpleNamespace()) == (10, 2)


def test_get_turn_limits_uses_prompt_config_values() -> None:
    """Turn limits should honor prompt-config overrides."""
    cfg = SimpleNamespace(max_tool_iterations=15, max_resume_attempts=4)
    assert get_turn_limits(cfg) == (15, 4)
