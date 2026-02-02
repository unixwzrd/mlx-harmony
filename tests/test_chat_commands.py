"""Unit tests for chat command parsing/rendering."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_harmony.chat_commands import (
    build_help_text,
    normalize_set_command,
    parse_command,
    parse_hyperparameter_update,
    render_hyperparameters,
    render_models_list,
)


def test_build_help_text_contains_supported_commands() -> None:
    help_text = build_help_text()
    assert "\\help" in help_text
    assert "\\models" in help_text
    assert "\\set <param>=<value>" in help_text


def test_render_hyperparameters_defaults_message() -> None:
    rendered = render_hyperparameters({})
    assert "(using defaults)" in rendered


def test_render_hyperparameters_sorted_output() -> None:
    rendered = render_hyperparameters({"top_p": 0.9, "temperature": 0.7})
    assert rendered.index("temperature = 0.7") < rendered.index("top_p = 0.9")


def test_normalize_set_command_strips_prefixes() -> None:
    assert normalize_set_command("\\set temperature=0.7") == "temperature=0.7"
    assert normalize_set_command("/set top_k=40") == "top_k=40"


def test_parse_hyperparameter_update_float_param() -> None:
    ok, message, updates = parse_hyperparameter_update("temperature", "0.7")
    assert ok is True
    assert "Set temperature = 0.7" in message
    assert updates == {"temperature": 0.7}


def test_parse_hyperparameter_update_int_param() -> None:
    ok, message, updates = parse_hyperparameter_update("top_k", "40")
    assert ok is True
    assert "Set top_k = 40" in message
    assert updates == {"top_k": 40}


def test_parse_hyperparameter_update_rejects_invalid_value() -> None:
    ok, message, updates = parse_hyperparameter_update("temperature", "abc")
    assert ok is False
    assert "Invalid value" in message
    assert updates == {}


def test_parse_command_set_and_unknown_commands() -> None:
    handled, should_apply, _, updates = parse_command("\\set temperature=0.5", {})
    assert handled is True
    assert should_apply is True
    assert updates == {"temperature": 0.5}

    handled, should_apply, message, updates = parse_command("\\unknown", {})
    assert handled is True
    assert should_apply is False
    assert "Unknown out-of-band command" in message
    assert updates == {}


def test_render_models_list_reads_models_directory(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "model-a").mkdir(parents=True)
    (models_dir / "model-b").mkdir(parents=True)
    (models_dir / "README.txt").write_text("ignore non-dir", encoding="utf-8")

    rendered = render_models_list(str(models_dir), profiles_file=None)

    assert "Available models" in rendered
    assert str(models_dir / "model-a") in rendered
    assert str(models_dir / "model-b") in rendered
    assert "README.txt" not in rendered


def test_render_models_list_falls_back_to_profiles(tmp_path: Path) -> None:
    profiles_file = tmp_path / "profiles.json"
    profiles_file.write_text(
        json.dumps(
            {
                "alpha": {"model": "models/alpha"},
                "beta": {"model": "models/beta"},
            }
        ),
        encoding="utf-8",
    )

    rendered = render_models_list(models_dir=None, profiles_file=str(profiles_file))

    assert "models/alpha" in rendered
    assert "models/beta" in rendered
