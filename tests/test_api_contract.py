"""Unit tests for shared API request contracts."""

from __future__ import annotations

import pytest

from mlx_harmony.api_contract import (
    ChatRequest,
    RequestValidationError,
    apply_stop_sequences,
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_health_response,
    build_models_list_response,
    build_openai_error_response,
    validate_chat_request_supported,
)


def test_validate_chat_request_supported_accepts_json_object_response_format() -> None:
    """Allow supported response_format variants used by OpenAI clients."""
    request = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
    )
    validate_chat_request_supported(request)


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ({"n": 2}, "limited to 1"),
        ({"stop": ""}, "stop"),
        ({"stop": []}, "stop"),
        ({"tools": [{"type": "function"}]}, "tools"),
        ({"response_format": {"type": "json_schema"}}, "response_format"),
        ({"logprobs": True}, "logprobs"),
    ],
)
def test_validate_chat_request_supported_rejects_unsupported_fields(
    payload: dict[str, object],
    expected_message: str,
) -> None:
    """Reject unsupported request features with explicit errors."""
    request = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        **payload,
    )
    with pytest.raises(RequestValidationError) as exc_info:
        validate_chat_request_supported(request)
    assert expected_message in str(exc_info.value)


def test_apply_stop_sequences_returns_first_match() -> None:
    """Truncate at the first stop marker that appears in output text."""
    text, truncated = apply_stop_sequences("alpha beta gamma", [" gamma", " beta"])
    assert text == "alpha"
    assert truncated is True


def test_build_chat_completion_chunk_shape() -> None:
    """Build stream chunk payload with expected OpenAI-compatible fields."""
    chunk = build_chat_completion_chunk(
        response_id="chatcmpl-123",
        created=123,
        model="test-model",
        system_fingerprint="mlx-harmony-local",
        delta={"content": "hello"},
        finish_reason=None,
    )
    assert chunk["id"] == "chatcmpl-123"
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["system_fingerprint"] == "mlx-harmony-local"
    assert chunk["choices"][0]["delta"]["content"] == "hello"


def test_build_chat_completion_response_shape() -> None:
    """Build non-stream response payload with expected fields and usage."""
    response = build_chat_completion_response(
        response_id="chatcmpl-123",
        created=123,
        model="test-model",
        system_fingerprint="mlx-harmony-local",
        assistant_text="hi",
        analysis_text="thinking",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
    )
    assert response["id"] == "chatcmpl-123"
    assert response["object"] == "chat.completion"
    assert response["system_fingerprint"] == "mlx-harmony-local"
    assert response["choices"][0]["message"]["analysis"] == "thinking"
    assert response["usage"]["total_tokens"] == 15


def test_build_models_list_response_shape() -> None:
    """Build model list payload in OpenAI-compatible shape."""
    response = build_models_list_response(model_ids=["model-a", "model-b"])
    assert response["object"] == "list"
    assert response["data"][0]["id"] == "model-a"
    assert response["data"][1]["object"] == "model"


def test_build_health_response_shape() -> None:
    """Build server health payload with expected metadata fields."""
    response = build_health_response(
        model_loaded=True,
        model_path="models/test",
        prompt_config_path="configs/test.json",
        loaded_at_unix=123,
    )
    assert response["object"] == "health"
    assert response["status"] == "ok"
    assert response["model_loaded"] is True
    assert response["model_path"] == "models/test"


def test_build_openai_error_response_shape() -> None:
    """Build OpenAI-style error envelope payload."""
    response = build_openai_error_response(
        message="Not implemented",
        error_type="not_implemented_error",
        param="tools",
        code="not_implemented",
    )
    assert response["error"]["message"] == "Not implemented"
    assert response["error"]["type"] == "not_implemented_error"
    assert response["error"]["param"] == "tools"
    assert response["error"]["code"] == "not_implemented"
