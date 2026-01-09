"""
Pytest configuration and shared fixtures for mlx-harmony tests.
"""
from pathlib import Path

import pytest

# Small test model from HuggingFace (0.5B parameters, ~300MB)
# This is cached by HuggingFace, so subsequent test runs are fast
TEST_MODEL = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


@pytest.fixture(scope="session")
def test_model_path() -> str:
    """
    Returns the HuggingFace model path for testing.

    This model is small (~300MB) and will be downloaded and cached
    by HuggingFace on first use.
    """
    return TEST_MODEL


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Returns the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_configs_dir() -> Path:
    """Returns the test configs directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Returns a temporary directory for test output."""
    return tmp_path


@pytest.fixture
def sample_prompt_config() -> dict:
    """Returns a sample prompt config for testing."""
    return {
        "system_model_identity": "You are {assistant}, a helpful AI assistant.",
        "reasoning_effort": "Medium",
        "conversation_start_date": "<|DATE|>",
        "knowledge_cutoff": "2024-06",
        "developer_instructions": "Be helpful and concise.",
        "assistant_greeting": "Hello {user}, I'm {assistant}. How can I help?",
        "placeholders": {
            "assistant": "TestBot",
            "user": "Tester",
        },
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "truncate_thinking": 500,
        "truncate_response": 500,
        "logs_dir": "test_logs",
        "chats_dir": "test_chats",
    }


@pytest.fixture
def sample_conversation() -> list:
    """Returns a sample conversation for testing."""
    return [
        {
            "role": "user",
            "content": "Hello!",
            "timestamp": "2025-01-07T12:00:00Z",
        },
        {
            "role": "assistant",
            "content": "Hello! How can I help you?",
            "timestamp": "2025-01-07T12:00:01Z",
            "hyperparameters": {"temperature": 0.7, "max_tokens": 100},
        },
    ]
