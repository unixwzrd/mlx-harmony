"""
Integration tests for the HTTP API server.

Tests FastAPI endpoints, request/response handling, and streaming.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlx_harmony.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_generator():
    """Create a mock TokenGenerator for testing."""
    mock_gen = MagicMock()
    mock_gen.model_path = "test-model"
    mock_gen.tokenizer = MagicMock()
    mock_gen.tokenizer.decode = lambda tokens: "".join(
        ["test"] * len(tokens)
    )  # Simple mock decode
    mock_gen.generate = MagicMock(return_value=iter([1, 2, 3, 4, 5]))  # Mock token IDs
    return mock_gen


class TestChatCompletions:
    """Test /v1/chat/completions endpoint."""

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_basic(self, mock_get_gen, client: TestClient, mock_generator):
        """Test basic chat completions request."""
        mock_get_gen.return_value = mock_generator

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 10,
            "temperature": 0.7,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

        # Verify generator was called correctly
        mock_get_gen.assert_called_once()
        mock_generator.generate.assert_called_once()

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_sampling_params(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Test chat completions with all sampling parameters."""
        mock_get_gen.return_value = mock_generator

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.8,
            "max_tokens": 20,
            "top_p": 0.9,
            "min_p": 0.1,
            "top_k": 10,
            "repetition_penalty": 1.2,
            "repetition_context_size": 25,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify generator was called with all parameters
        call_kwargs = mock_generator.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 20
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["min_p"] == 0.1
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["repetition_penalty"] == 1.2
        assert call_kwargs["repetition_context_size"] == 25

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_multiple_messages(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Test chat completions with multiple messages."""
        mock_get_gen.return_value = mock_generator

        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "max_tokens": 10,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify messages were passed correctly
        call_args = mock_generator.generate.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        assert messages is not None
        assert len(messages) == 4

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_streaming(self, mock_get_gen, client: TestClient, mock_generator):
        """Test streaming chat completions."""
        mock_get_gen.return_value = mock_generator

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 10,
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming response
        lines = response.text.split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l.strip() != "data: [DONE]"]

        assert len(data_lines) > 0

        # Parse first chunk
        first_chunk = json.loads(data_lines[0][6:])  # Remove "data: " prefix
        assert "choices" in first_chunk
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]

        # Verify final chunk is [DONE]
        assert any("data: [DONE]" in line for line in lines)

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_profile(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Test chat completions with profile."""
        profiles_data = {
            "test-profile": {
                "model": "profile-model",
                "prompt_config": None,
            }
        }

        mock_get_gen.return_value = mock_generator

        # Mock load_profiles to return our test profiles
        # The server loads from "configs/profiles.example.json" by default
        with patch("mlx_harmony.server.load_profiles") as mock_load_profiles:
            mock_load_profiles.return_value = profiles_data

            request_data = {
                "model": "test-model",  # This should be overridden by profile
                "profile": "test-profile",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            }

            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200

            # Verify load_profiles was called (it's called in the endpoint)
            mock_load_profiles.assert_called_once()
            # Verify it was called with the default profiles path
            assert mock_load_profiles.call_args[0][0] == "configs/profiles.example.json"

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_profiles_file_override(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Test chat completions with profiles_file override."""
        profiles_data = {
            "test-profile": {
                "model": "profile-model",
                "prompt_config": None,
            }
        }

        mock_get_gen.return_value = mock_generator

        with patch("mlx_harmony.server.load_profiles") as mock_load_profiles:
            mock_load_profiles.return_value = profiles_data

            request_data = {
                "model": "test-model",
                "profile": "test-profile",
                "profiles_file": "configs/custom-profiles.json",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            }

            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200

            mock_load_profiles.assert_called_once()
            assert mock_load_profiles.call_args[0][0] == "configs/custom-profiles.json"

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_profiles_env_override(
        self, mock_get_gen, client: TestClient, mock_generator, monkeypatch: pytest.MonkeyPatch
    ):
        """Test chat completions with profiles file from environment."""
        profiles_data = {
            "test-profile": {
                "model": "profile-model",
                "prompt_config": None,
            }
        }

        mock_get_gen.return_value = mock_generator
        monkeypatch.setenv("MLX_HARMONY_PROFILES_FILE", "configs/env-profiles.json")

        with patch("mlx_harmony.server.load_profiles") as mock_load_profiles:
            mock_load_profiles.return_value = profiles_data

            request_data = {
                "model": "test-model",
                "profile": "test-profile",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            }

            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200

            mock_load_profiles.assert_called_once()
            assert mock_load_profiles.call_args[0][0] == "configs/env-profiles.json"

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_invalid_profile(self, mock_get_gen, client: TestClient):
        """Test chat completions with invalid profile."""
        # Mock load_profiles to return empty profiles
        with patch("mlx_harmony.server.load_profiles") as mock_load_profiles:
            mock_load_profiles.return_value = {}

            request_data = {
                "model": "test-model",
                "profile": "nonexistent-profile",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            }

            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 400
            assert "not found" in response.json()["detail"].lower()


class TestServerErrors:
    """Test error handling in server."""

    def test_chat_completions_missing_model(self, client: TestClient):
        """Test chat completions with missing model."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        response = client.post("/v1/chat/completions", json=request_data)
        # FastAPI should validate and return 422 for missing required field
        assert response.status_code in [400, 422]

    def test_chat_completions_missing_messages(self, client: TestClient):
        """Test chat completions with missing messages."""
        request_data = {
            "model": "test-model",
        }

        response = client.post("/v1/chat/completions", json=request_data)
        # FastAPI should validate and return 422 for missing required field
        assert response.status_code in [400, 422]

    def test_chat_completions_invalid_messages(self, client: TestClient):
        """Test chat completions with invalid messages format."""
        request_data = {
            "model": "test-model",
            "messages": "not a list",  # Invalid: should be a list
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code in [400, 422]


@pytest.mark.requires_model
@pytest.mark.slow
class TestServerWithRealModel:
    """Integration tests with a real model (slower tests)."""

    def test_chat_completions_real_model(self, client: TestClient, test_model_path: str):
        """Test chat completions with a real model."""
        request_data = {
            "model": test_model_path,
            "messages": [{"role": "user", "content": "Say hello!"}],
            "max_tokens": 5,
            "temperature": 0.0,  # Deterministic
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_chat_completions_streaming_real_model(
        self, client: TestClient, test_model_path: str
    ):
        """Test streaming chat completions with a real model."""
        request_data = {
            "model": test_model_path,
            "messages": [{"role": "user", "content": "Count to three:"}],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming response
        lines = response.text.split("\n")
        data_lines = [
            l
            for l in lines
            if l.startswith("data: ") and l.strip() not in ["data: [DONE]", "data: "]
        ]

        assert len(data_lines) > 0

        # Verify chunks are valid JSON
        for line in data_lines[:5]:  # Check first 5 chunks
            chunk_data = json.loads(line[6:])  # Remove "data: " prefix
            assert "choices" in chunk_data
