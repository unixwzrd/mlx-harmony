"""
Integration tests for the HTTP API server.

Tests FastAPI endpoints, request/response handling, and streaming.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import mlx_harmony.server as server_module
from mlx_harmony.backend_api import BackendChatResult
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
    mock_gen.render_prompt_tokens = MagicMock(return_value=[1, 2, 3])
    mock_gen.render_prompt = MagicMock(return_value="<prompt>")
    mock_gen.use_harmony = False
    mock_gen.prompt_config = None
    return mock_gen


@pytest.fixture(autouse=True)
def stub_run_backend_chat(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if "requires_model" in request.keywords:
        return

    def _fake_run_backend_chat(*, conversation: list[dict], **_kwargs):
        parent_id = conversation[-1].get("id") if conversation else None
        assistant_id = "assistant-test-id"
        conversation.append(
            {
                "id": assistant_id,
                "parent_id": parent_id,
                "cache_key": assistant_id,
                "role": "assistant",
                "content": "stub response",
                "analysis": "stub analysis",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        return BackendChatResult(
            assistant_text="stub response",
            analysis_text="stub analysis",
            prompt_tokens=5,
            completion_tokens=7,
            finish_reason="stop",
            last_prompt_start_time=None,
        )

    monkeypatch.setattr(server_module, "run_backend_chat", _fake_run_backend_chat)


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
        assert data["system_fingerprint"] == "mlx-harmony-local"
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

        # Verify generator was resolved
        mock_get_gen.assert_called_once()

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

        # Verify generator was resolved
        mock_get_gen.assert_called_once()

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

        # Verify generator was resolved
        mock_get_gen.assert_called_once()

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_streaming(self, mock_get_gen, client: TestClient, mock_generator):
        """Test streaming chat completions."""
        mock_get_gen.return_value = mock_generator
        mock_generator.encoding = None

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 10,
            "stream": True,
        }

        backend_result = BackendChatResult(
            assistant_text="test",
            analysis_text=None,
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
            last_prompt_start_time=1.0,
        )

        with patch("mlx_harmony.server.run_backend_chat") as mock_backend:
            mock_backend.return_value = backend_result
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
        assert first_chunk["system_fingerprint"] == "mlx-harmony-local"
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]

        # Verify final chunk is [DONE]
        assert any("data: [DONE]" in line for line in lines)

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_stream_nonstream_use_shared_backend_executor(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Ensure stream and non-stream paths both use shared backend helpers."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 10,
        }
        backend_inputs = {"conversation": [], "hyperparameters": {}}
        backend_result = BackendChatResult(
            assistant_text="shared response",
            analysis_text=None,
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
            last_prompt_start_time=1.0,
        )
        with (
            patch("mlx_harmony.server.prepare_backend_inputs") as mock_prepare,
            patch("mlx_harmony.server.execute_backend_turn") as mock_execute,
        ):
            mock_prepare.return_value = backend_inputs
            mock_execute.return_value = (
                backend_result,
                server_module.BackendState(last_prompt_start_time=1.0, generation_index=1),
            )

            non_stream_response = client.post("/v1/chat/completions", json=request_data)
            assert non_stream_response.status_code == 200

            stream_request_data = dict(request_data)
            stream_request_data["stream"] = True
            stream_response = client.post("/v1/chat/completions", json=stream_request_data)
            assert stream_response.status_code == 200
            assert "data: [DONE]" in stream_response.text

        assert mock_prepare.call_count == 2
        assert mock_execute.call_count == 2
        for call in mock_execute.call_args_list:
            assert call.kwargs["inputs"] is backend_inputs

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
        self, mock_get_gen, client: TestClient, mock_generator, tmp_path: Path
    ):
        """Test chat completions with profiles_file override."""
        profiles_data = {
            "test-profile": {
                "model": "profile-model",
                "prompt_config": None,
            }
        }

        mock_get_gen.return_value = mock_generator

        profiles_file = tmp_path / "custom-profiles.json"
        profiles_file.write_text("{}", encoding="utf-8")

        with patch("mlx_harmony.server.load_profiles") as mock_load_profiles:
            mock_load_profiles.return_value = profiles_data

            request_data = {
                "model": "test-model",
                "profile": "test-profile",
                "profiles_file": str(profiles_file),
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            }

            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200

            mock_load_profiles.assert_called_once()
            assert mock_load_profiles.call_args[0][0] == str(profiles_file)

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_profiles_env_override(
        self,
        mock_get_gen,
        client: TestClient,
        mock_generator,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """Test chat completions with profiles file from environment."""
        profiles_data = {
            "test-profile": {
                "model": "profile-model",
                "prompt_config": None,
            }
        }

        mock_get_gen.return_value = mock_generator
        profiles_file = tmp_path / "env-profiles.json"
        profiles_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("MLX_HARMONY_PROFILES_FILE", str(profiles_file))

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
            assert mock_load_profiles.call_args[0][0] == str(profiles_file)

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
            assert "not found" in response.json()["error"]["message"].lower()

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_with_seed(self, mock_get_gen, client: TestClient, mock_generator):
        """Test chat completions accepts deterministic seed parameter."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "seed": 42,
            "max_tokens": 10,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_n_greater_than_one(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject unsupported n>1 until multi-choice output is implemented."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "n": 2,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "limited to 1" in response.json()["error"]["message"]

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_applies_stop_string_non_stream(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Truncate assistant content at stop sequence for non-stream calls."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stop": " response",
            "return_analysis": True,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"] == "stub"
        assert message["analysis"] == "stub analysis"

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_applies_stop_list_stream(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Truncate stream chunk content at the first matching stop sequence."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stop": [" analysis", " response"],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data_lines = [
            line[6:]
            for line in response.text.split("\n")
            if line.startswith("data: ") and line not in {"data: [DONE]", "data: "}
        ]
        payloads = [json.loads(line) for line in data_lines]
        content_chunks = [
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in payloads
            if chunk["choices"][0]["delta"].get("content") is not None
        ]
        assert "stub" in content_chunks
        final_chunk = payloads[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "stop"

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_invalid_stop_payload(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject invalid stop types/values explicitly."""
        mock_get_gen.return_value = mock_generator
        invalid_payloads = [
            {"stop": ""},
            {"stop": []},
            {"stop": ["ok", ""]},
        ]
        for payload in invalid_payloads:
            request_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello!"}],
                **payload,
            }
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 400
            assert "stop" in response.json()["error"]["message"]

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_unsupported_presence_penalty(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject unsupported non-zero presence_penalty."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "presence_penalty": 0.5,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "presence_penalty" in response.json()["error"]["message"]

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_unsupported_tools(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject unsupported tools parameter explicitly."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "tools" in response.json()["error"]["message"]

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_accepts_response_format_json_object(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Accept response_format json_object for OpenAI compatibility."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "response_format": {"type": "json_object"},
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_unsupported_response_format_type(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject unsupported response_format types explicitly."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "response_format": {"type": "json_schema"},
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "response_format" in response.json()["error"]["message"]

    @patch("mlx_harmony.server._get_generator")
    def test_chat_completions_rejects_unsupported_logprobs(
        self, mock_get_gen, client: TestClient, mock_generator
    ):
        """Reject unsupported logprobs parameter explicitly."""
        mock_get_gen.return_value = mock_generator
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "logprobs": True,
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "logprobs" in response.json()["error"]["message"]


class TestServerErrors:
    """Test error handling in server."""

    def test_chat_completions_missing_model(self, client: TestClient):
        """Test chat completions with missing model."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"
        assert "model" in body["error"]["message"].lower()


class TestServerHealth:
    """Test the health endpoint."""

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "health"
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["model_path"] is None
        assert body["prompt_config_path"] is None
        assert body["loaded_at_unix"] is None

    def test_chat_completions_missing_messages(self, client: TestClient):
        """Test chat completions with missing messages."""
        request_data = {
            "model": "test-model",
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "messages"

    def test_chat_completions_invalid_messages(self, client: TestClient):
        """Test chat completions with invalid messages format."""
        request_data = {
            "model": "test-model",
            "messages": "not a list",  # Invalid: should be a list
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "messages"

    def test_models_endpoint_reads_models_dir(self, client: TestClient, tmp_path: Path) -> None:
        """Return OpenAI-compatible model list from models directory."""
        (tmp_path / "model-a").mkdir()
        (tmp_path / "model-b").mkdir()
        with patch.dict("os.environ", {"MLX_HARMONY_MODELS_DIR": str(tmp_path)}):
            response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        model_ids = [entry["id"] for entry in body["data"]]
        assert str(tmp_path / "model-a") in model_ids
        assert str(tmp_path / "model-b") in model_ids

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/completions",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/speech",
            "/v1/images/generations",
            "/v1/images/edits",
            "/v1/images/variations",
            "/v1/moderations",
            "/v1/files",
            "/v1/batches",
            "/v1/responses",
        ],
    )
    def test_placeholder_endpoints_return_openai_style_501(
        self,
        client: TestClient,
        path: str,
    ) -> None:
        """Verify placeholder endpoints return a standardized 501 error envelope."""
        response = client.post(path, json={})
        assert response.status_code == 501
        body = response.json()
        assert body["error"]["type"] == "not_implemented_error"
        assert body["error"]["code"] == "not_implemented"


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
