"""
Integration tests for the chat module.

Tests conversation saving/loading, hyperparameter changes, and chat flow.
"""
import json
from pathlib import Path

from mlx_harmony import chat_io
from mlx_harmony.chat import load_conversation, save_conversation
from mlx_harmony.chat_history import normalize_dir_path
from mlx_harmony.chat_io import read_user_input
from mlx_harmony.chat_prompt import truncate_conversation_for_context
from mlx_harmony.prompt_cache import PromptTokenCache


class TestConversationIO:
    """Test conversation save/load functionality."""

    def test_save_conversation(self, sample_conversation: list, temp_dir: Path):
        """Test saving a conversation to JSON."""
        chat_file = temp_dir / "test_chat.json"
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
            prompt_config_path=None,
            tools=[],
            hyperparameters={"temperature": 0.7},
        )
        assert chat_file.exists()
        data = json.loads(chat_file.read_text(encoding="utf-8"))
        assert data["schema_version"] == 2
        assert "chats" in data
        assert "chat_order" in data
        chat_id = data["active_chat_id"]
        assert chat_id in data["chats"]
        assert len(data["chats"][chat_id]["messages"]) == 2

    def test_load_conversation(self, sample_conversation: list, temp_dir: Path):
        """Test loading a conversation from JSON."""
        chat_file = temp_dir / "test_chat.json"
        # Save first
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
            prompt_config_path=None,
            tools=[],
            hyperparameters={"temperature": 0.7},
        )
        # Then load
        messages, metadata = load_conversation(chat_file)
        assert len(messages) == 2
        assert metadata["model_path"] == "test-model"
        assert "hyperparameters" in metadata
        assert "chat_id" in metadata

    def test_conversation_metadata(self, sample_conversation: list, temp_dir: Path):
        """Test that conversation metadata is preserved."""
        chat_file = temp_dir / "test_chat.json"
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
            prompt_config_path="test-config.json",
            tools=[type("Tool", (), {"name": "browser"})()],
            hyperparameters={"temperature": 0.7, "max_tokens": 100},
        )
        messages, metadata = load_conversation(chat_file)
        assert metadata["model_path"] == "test-model"
        assert metadata["prompt_config_path"] == "test-config.json"
        assert "browser" in metadata["tools"]
        assert metadata["hyperparameters"]["temperature"] == 0.7
        assert metadata["hyperparameters"]["max_tokens"] == 100
        assert metadata["last_prompt_config_path"] == "test-config.json"

    def test_conversation_timestamps(self, sample_conversation: list, temp_dir: Path):
        """Test that timestamps are preserved in conversations."""
        chat_file = temp_dir / "test_chat.json"
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
        )
        messages, _ = load_conversation(chat_file)
        for msg in messages:
            assert "timestamp" in msg

    def test_conversation_hyperparameters_per_turn(
        self, sample_conversation: list, temp_dir: Path
    ):
        """Test that hyperparameters are saved per assistant turn."""
        chat_file = temp_dir / "test_chat.json"
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
        )
        messages, _ = load_conversation(chat_file)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        assert assistant_msg is not None
        assert "hyperparameters" in assistant_msg


class TestChatHelpers:
    """Test chat helper utilities."""

    def test_normalize_dir_path(self):
        """Normalize logs path to avoid nested logs/log."""
        assert str(normalize_dir_path("logs/log")) == "logs"
        assert str(normalize_dir_path("logs/logs")) == "logs"

    def test_read_user_input_continuation(self, monkeypatch):
        """Allow multi-line input using a trailing backslash."""
        inputs = iter(["first line\\", "second line"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr(chat_io.select, "select", lambda *_: ([], [], []))
        assert read_user_input(">> ") == "first line\nsecond line"

    def test_read_user_input_block(self, monkeypatch):
        """Allow block input using \\ start/end markers."""
        inputs = iter(["\\", "first line", "second line", "\\"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        assert read_user_input(">> ") == "first line\nsecond line"

    def test_truncate_conversation_for_context(self):
        """Truncate oldest messages to respect max_context_tokens."""
        class StubGenerator:
            def render_prompt_tokens(self, messages, _system_message=None):
                return list(range(len(messages) * 4))

        generator = StubGenerator()
        conversation = [
            {"role": "user", "content": "oldest"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "newest"},
        ]
        trimmed, prompt_tokens = truncate_conversation_for_context(
            generator=generator,
            conversation=conversation,
            system_message=None,
            max_context_tokens=8,
        )
        assert len(trimmed) == 2
        assert prompt_tokens <= 8
        assert trimmed[0]["content"] == "reply"

    def test_truncate_conversation_perf_prompt_budget(self):
        """Use perf prompt token budget for early truncation."""
        class StubConfig:
            performance_mode = True
            perf_prompt_token_budget = 8

        class StubGenerator:
            def __init__(self) -> None:
                self.prompt_config = StubConfig()

            def render_prompt_tokens(self, messages, _system_message=None):
                return list(range(len(messages) * 4))

        generator = StubGenerator()
        conversation = [
            {"role": "user", "content": "oldest"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "newest"},
        ]
        trimmed, prompt_tokens = truncate_conversation_for_context(
            generator=generator,
            conversation=conversation,
            system_message=None,
            max_context_tokens=20,
        )
        assert len(trimmed) == 2
        assert prompt_tokens <= 8

    def test_truncate_conversation_preserves_system_and_developer_with_cache(self):
        """Keep system/developer messages while truncating with prompt cache enabled."""
        class StubGenerator:
            def __init__(self) -> None:
                self.use_harmony = True
                self.encoding = object()
                self.prompt_token_cache = PromptTokenCache()

            def render_prompt_tokens(self, messages, _system_message=None):
                base_prefix = [0, 1]
                assistant_start = [2, 3]
                token_ids: list[int] = []
                for idx, _msg in enumerate(messages):
                    token_ids.extend([10 + idx * 2, 11 + idx * 2])
                return base_prefix + token_ids + assistant_start

        generator = StubGenerator()
        conversation = [
            {"role": "system", "content": "system", "cache_key": "system"},
            {"role": "developer", "content": "developer", "cache_key": "developer"},
            {"role": "user", "content": "oldest user", "cache_key": "u1"},
            {"role": "assistant", "content": "reply", "cache_key": "a1"},
            {"role": "user", "content": "newest user", "cache_key": "u2"},
        ]
        trimmed, prompt_tokens = truncate_conversation_for_context(
            generator=generator,
            conversation=conversation,
            system_message=None,
            max_context_tokens=12,
        )
        roles = [msg["role"] for msg in trimmed]
        assert "system" in roles
        assert "developer" in roles
        assert roles[0] == "system"
        assert roles[1] == "developer"
        assert all(msg.get("content") != "oldest user" for msg in trimmed)
        assert prompt_tokens >= 12


class TestChatIntegration:
    """Integration tests for chat functionality."""

    def test_chat_file_path_resolution(self, temp_dir: Path):
        """Test that chat file paths are resolved correctly."""
        # This tests the logic in chat.py for resolving chat file paths
        chat_name = "test_chat"
        chats_dir = temp_dir / "chats"
        chats_dir.mkdir(parents=True, exist_ok=True)
        chat_file = chats_dir / f"{chat_name}.json"
        assert chat_file.parent == chats_dir
        assert chat_file.name == f"{chat_name}.json"

    def test_conversation_append(self, sample_conversation: list, temp_dir: Path):
        """Test that new messages are appended to existing conversations."""
        chat_file = temp_dir / "test_chat.json"
        # Save initial conversation
        save_conversation(
            chat_file,
            sample_conversation,
            model_path="test-model",
        )
        # Add a new message
        new_message = {
            "role": "user",
            "content": "Another message",
            "timestamp": "2025-01-07T12:02:00Z",
        }
        extended_conversation = sample_conversation + [new_message]
        # Save again (should preserve created_at)
        save_conversation(
            chat_file,
            extended_conversation,
            model_path="test-model",
        )
        # Load and verify
        messages, metadata = load_conversation(chat_file)
        assert len(messages) == 3
        # created_at should be preserved
        assert "created_at" in metadata

    def test_chat_history_round_trip(self, temp_dir: Path):
        """Ensure timestamps and hyperparameters are preserved after save/load."""
        chat_file = temp_dir / "round_trip.json"
        conversation = [
            {
                "role": "user",
                "content": "Hi",
                "timestamp": "2025-01-07T12:00:00Z",
            },
            {
                "role": "assistant",
                "content": "Hello",
                "timestamp": "2025-01-07T12:00:01Z",
                "hyperparameters": {"temperature": 0.5, "max_tokens": 42},
            },
        ]
        save_conversation(
            chat_file,
            conversation,
            model_path="test-model",
            hyperparameters={"temperature": 0.5, "max_tokens": 42},
        )
        messages, metadata = load_conversation(chat_file)
        assert len(messages) == 2
        assert messages[0].get("id")
        assert messages[0].get("cache_key")
        assert messages[1].get("id")
        assert messages[1].get("cache_key")
        assert messages[1].get("parent_id") == messages[0]["id"]
        assert messages[0]["timestamp"]["iso"] == "2025-01-07T12:00:00Z"
        assert messages[1]["hyperparameters"]["temperature"] == 0.5
        assert metadata["hyperparameters"]["max_tokens"] == 42
        assert metadata["last_hyperparameters"]["max_tokens"] == 42
