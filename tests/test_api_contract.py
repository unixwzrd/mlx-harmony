"""Unit tests for shared API request contracts."""

from __future__ import annotations

import pytest

from mlx_harmony.api_contract import (
    ChatRequest,
    RequestValidationError,
    apply_stop_sequences,
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

