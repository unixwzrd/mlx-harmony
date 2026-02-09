"""Tests for backend adapter contract behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mlx_harmony.chat_backend import LocalBackend, ServerBackend
from mlx_harmony.generation.client import GenerationResult


def _backend_kwargs() -> dict[str, Any]:
    """Build default kwargs for backend adapter calls."""
    return {
        "conversation": [{"role": "user", "content": "hello"}],
        "hyperparameters": {
            "temperature": 0.7,
            "max_tokens": 64,
            "top_p": 0.9,
            "min_p": 0.0,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
        },
        "last_saved_hyperparameters": {
            "temperature": 0.7,
            "max_tokens": 64,
        },
        "last_user_text": "hello",
        "max_context_tokens": 8192,
        "last_prompt_start_time": 123.4,
        "generation_index": 6,
        "max_tool_iterations": 10,
        "max_resume_attempts": 2,
        "generator": object(),
        "tools": [],
        "assistant_name": "Assistant",
        "thinking_limit": None,
        "response_limit": None,
        "render_markdown": False,
        "debug": False,
        "debug_path": Path("debug.log"),
        "debug_tokens": None,
        "enable_artifacts": True,
        "make_message_id": lambda: "msg-id",
        "make_timestamp": lambda: "2026-01-01T00:00:00Z",
        "display_assistant": lambda *_args, **_kwargs: None,
        "display_thinking": lambda *_args, **_kwargs: None,
        "truncate_text": lambda text, _limit: text,
        "collect_memory_stats": lambda: {},
        "write_debug_metrics": lambda **_kwargs: None,
        "write_debug_response": lambda **_kwargs: None,
        "write_debug_info": lambda **_kwargs: None,
        "write_debug_token_texts": lambda **_kwargs: None,
        "write_debug_tokens": lambda **_kwargs: None,
    }


@dataclass(frozen=True)
class _StubTurnResult:
    """Simple stand-in for chat turn return values."""

    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    generation_index: int
    last_prompt_start_time: float | None
    prompt_tokens: int | None
    completion_tokens: int | None


class _StubGenerationClient:
    """Capture requests and return a configured generation response."""

    def __init__(self, response: GenerationResult) -> None:
        self.response = response
        self.requests: list[Any] = []

    def generate(self, request: Any) -> GenerationResult:
        self.requests.append(request)
        return self.response


def test_local_backend_returns_expected_contract() -> None:
    """Local backend should return turn data via BackendResult."""
    kwargs = _backend_kwargs()
    stub_result = _StubTurnResult(
        hyperparameters=kwargs["hyperparameters"],
        last_saved_hyperparameters=kwargs["last_saved_hyperparameters"],
        generation_index=7,
        last_prompt_start_time=125.0,
        prompt_tokens=101,
        completion_tokens=202,
    )
    with patch("mlx_harmony.chat_backend.run_chat_turn", return_value=stub_result) as run_mock:
        result = LocalBackend().generate(**kwargs)

    run_mock.assert_called_once()
    assert result.handled_conversation is True
    assert result.hyperparameters == kwargs["hyperparameters"]
    assert result.last_saved_hyperparameters == kwargs["last_saved_hyperparameters"]
    assert result.generation_index == 7
    assert result.prompt_tokens == 101
    assert result.completion_tokens == 202


def test_server_backend_maps_request_and_returns_expected_contract() -> None:
    """Server backend should map hyperparameters into GenerationRequest."""
    kwargs = _backend_kwargs()
    kwargs["conversation"] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "prior"},
    ]
    client = _StubGenerationClient(
        GenerationResult(
            text="remote assistant text",
            analysis_text="analysis",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=22,
        )
    )
    result = ServerBackend(client).generate(**kwargs)

    assert len(client.requests) == 1
    request = client.requests[0]
    assert [m["role"] for m in request.messages] == ["system", "user", "assistant"]
    assert request.temperature == 0.7
    assert request.max_tokens == 64
    assert request.top_p == 0.9
    assert request.top_k == 40
    assert request.repetition_penalty == 1.1
    assert request.repetition_context_size == 20

    assert result.handled_conversation is False
    assert result.assistant_text == "remote assistant text"
    assert result.analysis_text == "analysis"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 22
    assert result.generation_index == kwargs["generation_index"] + 1

