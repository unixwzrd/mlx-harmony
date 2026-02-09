"""Shared OpenAI-style API request contracts and validation helpers.

This module centralizes request models and protocol-level validation logic so
both HTTP and non-HTTP entrypoints can reuse the same behavior.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class RequestValidationError(ValueError):
    """Raised when a request payload uses unsupported or invalid fields."""


class ChatMessage(BaseModel):
    """One chat message in OpenAI-compatible request payloads.

    Attributes:
        role: Message role (for example ``system``, ``user``, or ``assistant``).
        content: Message text content.
    """

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class ChatRequest(BaseModel):
    """OpenAI-style chat completions request schema used by backend entrypoints.

    Unknown fields are ignored to preserve client compatibility while the
    backend implements additional parameters incrementally.
    """

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: list[ChatMessage]
    n: Optional[int] = 1
    stop: Optional[str | list[str]] = None
    temperature: Optional[float] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    seed: Optional[int] = None
    profile: Optional[str] = None
    prompt_config: Optional[str] = None
    profiles_file: Optional[str] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[str | dict[str, Any]] = None
    stream: bool = False
    return_analysis: bool = False


def validate_chat_request_supported(request: ChatRequest) -> None:
    """Validate chat fields that are currently unsupported by the backend.

    Args:
        request: Parsed chat completion payload.

    Raises:
        RequestValidationError: If unsupported or malformed parameters are
            provided.
    """

    if request.n not in (None, 1):
        raise RequestValidationError("Parameter 'n' is currently limited to 1.")

    if request.stop is not None:
        if isinstance(request.stop, list):
            if not request.stop:
                raise RequestValidationError("Parameter 'stop' must not be an empty list.")
            if any(not isinstance(item, str) or item == "" for item in request.stop):
                raise RequestValidationError(
                    "Parameter 'stop' list entries must be non-empty strings."
                )
        elif not isinstance(request.stop, str) or request.stop == "":
            raise RequestValidationError(
                "Parameter 'stop' must be a non-empty string or list of non-empty strings."
            )

    if request.tools:
        raise RequestValidationError("Parameter 'tools' is not yet supported.")

    if request.tool_choice is not None:
        raise RequestValidationError("Parameter 'tool_choice' is not yet supported.")

    if request.response_format is not None:
        if not isinstance(request.response_format, dict):
            raise RequestValidationError("Parameter 'response_format' must be an object.")
        response_format_type = request.response_format.get("type")
        if response_format_type not in {"text", "json_object"}:
            raise RequestValidationError(
                "Parameter 'response_format' currently supports only 'text' and 'json_object'."
            )

    if request.presence_penalty not in (None, 0, 0.0):
        raise RequestValidationError("Parameter 'presence_penalty' is not yet supported.")

    if request.frequency_penalty not in (None, 0, 0.0):
        raise RequestValidationError("Parameter 'frequency_penalty' is not yet supported.")


def apply_stop_sequences(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    """Apply OpenAI-style stop sequence truncation to assistant text.

    Args:
        text: Assistant output text.
        stop: One stop token or a list of stop tokens.

    Returns:
        A tuple of ``(truncated_text, was_truncated)``.
    """

    if not text or stop is None:
        return text, False

    stops = [stop] if isinstance(stop, str) else stop
    cut_indices = [text.find(token) for token in stops if token]
    cut_indices = [index for index in cut_indices if index >= 0]
    if not cut_indices:
        return text, False

    cut_at = min(cut_indices)
    return text[:cut_at], True

