"""Tests for HTTP API client payload construction and transport behavior."""

from __future__ import annotations

import json
from unittest.mock import patch

from mlx_harmony.api_client import ApiClient, ApiClientConfig


class _StubHTTPResponse:
    """Simple context-manager response stub for urllib tests."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> _StubHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        """Return encoded JSON payload bytes."""
        return json.dumps(self._payload).encode("utf-8")


def _client_config() -> ApiClientConfig:
    """Build a default API client config for tests."""
    return ApiClientConfig(
        host="127.0.0.1",
        port=8000,
        model="model-x",
        profile="profile-y",
        prompt_config="configs/test.json",
        max_tokens=None,
        timeout=5,
        return_analysis=False,
        requests_log=None,
    )


def test_send_messages_includes_explicit_sampling_fields() -> None:
    """Explicitly provided sampling fields should be serialized to JSON payload."""
    captured: dict[str, object] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["data"] = request.data
        captured["timeout"] = timeout
        return _StubHTTPResponse({"choices": []}, status=200)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        body = ApiClient(_client_config()).send_messages(
            [{"role": "user", "content": "hello"}],
            temperature=0.3,
            max_tokens=77,
            top_p=0.95,
            min_p=0.05,
            top_k=40,
            repetition_penalty=1.1,
            repetition_context_size=20,
            xtc_probability=0.25,
            xtc_threshold=0.1,
            seed=1234,
            loop_detection="full",
            reseed_each_turn=True,
            return_analysis=True,
        )

    payload = json.loads(captured["data"].decode("utf-8"))  # type: ignore[union-attr]
    assert body == {"choices": []}
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 77
    assert payload["top_p"] == 0.95
    assert payload["min_p"] == 0.05
    assert payload["top_k"] == 40
    assert payload["repetition_penalty"] == 1.1
    assert payload["repetition_context_size"] == 20
    assert payload["xtc_probability"] == 0.25
    assert payload["xtc_threshold"] == 0.1
    assert payload["seed"] == 1234
    assert payload["loop_detection"] == "full"
    assert payload["reseed_each_turn"] is True
    assert payload["return_analysis"] is True
    assert payload["model"] == "model-x"
    assert payload["profile"] == "profile-y"
    assert payload["prompt_config"] == "configs/test.json"


def test_send_messages_omits_unset_sampling_fields() -> None:
    """Unset sampling values should not be emitted into payload."""
    captured: dict[str, object] = {}
    config = _client_config()
    config = ApiClientConfig(
        host=config.host,
        port=config.port,
        model=None,
        profile=None,
        prompt_config=None,
        max_tokens=None,
        timeout=config.timeout,
        return_analysis=False,
        requests_log=None,
    )

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["data"] = request.data
        captured["timeout"] = timeout
        return _StubHTTPResponse({"choices": []}, status=200)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        ApiClient(config).send_messages([{"role": "user", "content": "hello"}])

    payload = json.loads(captured["data"].decode("utf-8"))  # type: ignore[union-attr]
    assert payload == {"messages": [{"role": "user", "content": "hello"}]}
