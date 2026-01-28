"""
Unit tests for the config module.

Tests placeholder expansion, config loading, and profile management.
"""
import json
from pathlib import Path

import pytest

from mlx_harmony.config import (
    apply_placeholders,
    load_profiles,
    load_prompt_config,
    parse_dialogue_text,
)


class TestPlaceholderExpansion:
    """Test placeholder expansion functionality."""

    def test_date_placeholder(self):
        """Test <|DATE|> placeholder expansion."""
        text = "Today is <|DATE|>"
        result = apply_placeholders(text, {})
        assert "<|DATE|>" not in result
        assert len(result) > len("Today is ")
        # Should be in YYYY-MM-DD format
        assert result.count("-") == 2

    def test_datetime_placeholder(self):
        """Test <|DATETIME|> placeholder expansion."""
        text = "Current time: <|DATETIME|>"
        result = apply_placeholders(text, {})
        assert "<|DATETIME|>" not in result
        assert "T" in result or "-" in result  # ISO format

    def test_time_placeholders(self):
        """Test time placeholders (TIME, TIMEZ, TIMEA, TIMEU)."""
        text = "<|TIME|> <|TIMEZ|> <|TIMEA|> <|TIMEU|>"
        result = apply_placeholders(text, {})
        assert "<|TIME|>" not in result
        assert "<|TIMEZ|>" not in result
        assert "<|TIMEA|>" not in result
        assert "<|TIMEU|>" not in result
        # Should have some time-like content
        assert len(result) > 10

    def test_user_placeholders_curly_braces(self):
        """Test user-defined placeholders in {key} format."""
        text = "Hello {user}, I'm {assistant}."
        placeholders = {"user": "Alice", "assistant": "Bob"}
        result = apply_placeholders(text, placeholders)
        assert result == "Hello Alice, I'm Bob."

    def test_user_placeholders_angle_brackets(self):
        """Test user-defined placeholders in <|KEY|> format (case-insensitive)."""
        text = "Hello <|USER|>, I'm <|assistant|>."
        placeholders = {"user": "Alice", "assistant": "Bob"}
        result = apply_placeholders(text, placeholders)
        assert "Alice" in result
        assert "Bob" in result
        assert "<|USER|>" not in result
        assert "<|assistant|>" not in result

    def test_mixed_placeholders(self):
        """Test mixing built-in and user-defined placeholders."""
        text = "Hello {user}, today is <|DATE|> at <|TIMEZ|>."
        placeholders = {"user": "Alice"}
        result = apply_placeholders(text, placeholders)
        assert "Alice" in result
        assert "<|DATE|>" not in result
        assert "<|TIMEZ|>" not in result


class TestPromptConfigLoading:
    """Test prompt config loading from JSON."""

    def test_load_minimal_config(self, temp_dir: Path):
        """Test loading a minimal config."""
        config_file = temp_dir / "minimal.json"
        config_data = {
            "system_model_identity": "You are a helpful assistant.",
            "temperature": 0.7,
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_prompt_config(str(config_file))
        assert config is not None
        assert config.system_model_identity == "You are a helpful assistant."
        assert config.temperature == 0.7

    def test_load_full_config(self, sample_prompt_config: dict, temp_dir: Path):
        """Test loading a full config with all fields."""
        config_file = temp_dir / "full.json"
        config_file.write_text(json.dumps(sample_prompt_config), encoding="utf-8")

        config = load_prompt_config(str(config_file))
        assert config is not None
        assert config.system_model_identity == "You are TestBot, a helpful AI assistant."
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.max_context_tokens == 2048
        assert config.truncate_thinking == 500
        assert config.truncate_response == 500
        assert config.logs_dir == "test_logs"
        assert config.chats_dir == "test_chats"

    def test_prompt_config_validation_ranges(self, temp_dir: Path):
        """Invalid sampling fields should fail validation."""
        config_file = temp_dir / "invalid.json"
        config_file.write_text(
            json.dumps(
                {
                    "temperature": 3.0,
                    "top_p": 1.5,
                    "min_p": -0.1,
                    "top_k": -1,
                    "truncate_thinking": -5,
                    "max_context_tokens": 0,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            load_prompt_config(str(config_file))

    def test_load_example_dialogues(self, temp_dir: Path):
        """Test loading example_dialogues as a list of conversations."""
        config_file = temp_dir / "examples.json"
        config_data = {
            "placeholders": {"assistant": "Mia"},
            "example_dialogues": [
                [
                    {"role": "user", "content": "Hello {assistant}"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            ],
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_prompt_config(str(config_file))
        assert config is not None
        assert config.example_dialogues is not None
        assert isinstance(config.example_dialogues, list)
        assert config.example_dialogues[0][0]["content"] == "Hello Mia"

    def test_load_system_and_developer_placeholders_without_user_placeholders(
        self, temp_dir: Path
    ):
        """System/developer placeholders expand even when placeholders map is empty."""
        config_file = temp_dir / "system-dev-placeholders.json"
        config_data = {
            "system_model_identity": "System time <|TIMEU|>",
            "developer_instructions": "Developer time <|TIMEU|>",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_prompt_config(str(config_file))
        assert config is not None
        assert "<|TIMEU|>" not in (config.system_model_identity or "")
        assert "<|TIMEU|>" not in (config.developer_instructions or "")

    def test_load_deterministic_time(self, temp_dir: Path):
        """Test deterministic time placeholder loading."""
        config_file = temp_dir / "deterministic-time.json"
        config_data = {
            "deterministic_time_enabled": True,
            "deterministic_time_iso": "2026-01-27T12:00:00Z",
            "system_model_identity": "System at <|TIMEU|>",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        config = load_prompt_config(str(config_file))
        assert config is not None
        assert config.deterministic_time_enabled is True
        assert config.deterministic_time_iso == "2026-01-27T12:00:00Z"
        assert config.placeholders["TIMEU"] == "12:00:00 UTC"
        assert config.system_model_identity == "System at 12:00:00 UTC"

    def test_deterministic_time_defaults_seed_and_time(
        self, temp_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        """Deterministic mode defaults missing time and seed values with warnings."""
        config_file = temp_dir / "deterministic-defaults.json"
        config_data = {
            "deterministic_time_enabled": True,
            "system_model_identity": "System at <|TIMEU|>",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        with caplog.at_level("WARNING"):
            config = load_prompt_config(str(config_file))
        assert config is not None
        assert config.deterministic_time_iso == "2000-01-01T00:00:00Z"
        assert config.placeholders["TIMEU"] == "00:00:00 UTC"
        assert config.seed == 0
        assert config.reseed_each_turn is False
        assert "deterministic_time_iso" in caplog.text
        assert "defaulting seed=0" in caplog.text

    def test_load_nonexistent_config(self):
        """Test loading a non-existent config returns None."""
        config = load_prompt_config("nonexistent.json")
        assert config is None

    def test_load_invalid_json(self, temp_dir: Path):
        """Test loading invalid JSON handles gracefully."""
        config_file = temp_dir / "invalid.json"
        config_file.write_text("{ invalid json }", encoding="utf-8")

        # Should handle gracefully (either return None or raise a clear error)
        try:
            config = load_prompt_config(str(config_file))
            # If it doesn't raise, it should return None
            assert config is None
        except (json.JSONDecodeError, ValueError):
            # Or raise a clear error
            pass


class TestProfileLoading:
    """Test profile loading from JSON."""

    def test_load_profiles(self, temp_dir: Path):
        """Test loading profiles from JSON."""
        profiles_file = temp_dir / "profiles.json"
        profiles_data = {
            "test-profile": {
                "model": "mlx-community/Qwen1.5-0.5B-Chat-4bit",
                "prompt_config": None,
            },
            "profile-with-config": {
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "prompt_config": str(temp_dir / "config.json"),
            },
        }
        profiles_file.write_text(json.dumps(profiles_data), encoding="utf-8")

        profiles = load_profiles(str(profiles_file))
        assert "test-profile" in profiles
        assert "profile-with-config" in profiles
        assert profiles["test-profile"]["model"] == "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    def test_load_nonexistent_profiles(self):
        """Test loading non-existent profiles raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_profiles("nonexistent.json")


class TestDialogueParsing:
    """Test dialogue text parsing."""

    def test_parse_simple_dialogue(self):
        """Test parsing a simple dialogue."""
        text = "user: Hello\nassistant: Hi there!"
        messages = parse_dialogue_text(text)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_parse_multiple_turns(self):
        """Test parsing multiple conversation turns."""
        text = """user: Hello
assistant: Hi!

user: How are you?
assistant: I'm doing well!"""
        messages = parse_dialogue_text(text)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_parse_empty_dialogue(self):
        """Test parsing empty dialogue returns empty list."""
        messages = parse_dialogue_text("")
        assert messages == []
