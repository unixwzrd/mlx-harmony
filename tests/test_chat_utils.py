"""Unit tests for shared chat utility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_harmony.chat_utils import resolve_api_profile_paths, resolve_startup_profile_paths


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

