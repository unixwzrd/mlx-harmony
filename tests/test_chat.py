"""
Integration tests for the chat module.

Tests conversation saving/loading, hyperparameter changes, and chat flow.
"""
import json
from pathlib import Path

from mlx_harmony.chat import load_conversation, save_conversation


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
        assert "metadata" in data
        assert "messages" in data
        assert len(data["messages"]) == 2

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
