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
    logprobs: Optional[bool | int] = None
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

    if request.logprobs is not None:
        raise RequestValidationError("Parameter 'logprobs' is not yet supported.")

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


def build_chat_completion_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    system_fingerprint: str,
    delta: dict[str, Any],
    finish_reason: str | None,
    index: int = 0,
) -> dict[str, Any]:
    """Build one OpenAI-compatible `chat.completion.chunk` payload.

    Args:
        response_id: Chat completion identifier.
        created: Unix timestamp for completion creation.
        model: Model identifier returned in the response.
        system_fingerprint: Backend fingerprint for reproducibility metadata.
        delta: Delta payload for streamed content.
        finish_reason: Optional finish reason for this chunk.
        index: Choice index.

    Returns:
        Serialized chunk object compatible with SSE `data:` payloads.
    """

    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_chat_completion_response(
    *,
    response_id: str,
    created: int,
    model: str,
    system_fingerprint: str,
    assistant_text: str,
    analysis_text: str | None,
    finish_reason: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    index: int = 0,
) -> dict[str, Any]:
    """Build one OpenAI-compatible non-stream chat completion payload.

    Args:
        response_id: Chat completion identifier.
        created: Unix timestamp for completion creation.
        model: Model identifier returned in the response.
        system_fingerprint: Backend fingerprint for reproducibility metadata.
        assistant_text: Final assistant text content.
        analysis_text: Optional analysis channel content.
        finish_reason: Finish reason for the choice.
        prompt_tokens: Prompt token count.
        completion_tokens: Completion token count.
        index: Choice index.

    Returns:
        Serialized non-stream chat completion response object.
    """

    message: dict[str, Any] = {"role": "assistant", "content": assistant_text}
    if analysis_text:
        message["analysis"] = analysis_text
    total_tokens = prompt_tokens + completion_tokens
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "choices": [
            {
                "index": index,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def build_models_list_response(*, model_ids: list[str]) -> dict[str, Any]:
    """Build an OpenAI-compatible models list response payload.

    Args:
        model_ids: Ordered model identifiers.

    Returns:
        Serialized `/v1/models` response object.
    """

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
            for model_id in model_ids
        ],
    }


def build_health_response(
    *,
    model_loaded: bool,
    model_path: str | None,
    prompt_config_path: str | None,
    loaded_at_unix: int | None,
) -> dict[str, Any]:
    """Build the API health payload.

    Args:
        model_loaded: Whether a model is currently loaded.
        model_path: Loaded model path if available.
        prompt_config_path: Prompt config path if available.
        loaded_at_unix: Model load timestamp if available.

    Returns:
        Serialized `/v1/health` response object.
    """

    return {
        "object": "health",
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "loaded_at_unix": loaded_at_unix,
    }


def build_openai_error_response(
    *,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-style error response payload.

    Args:
        message: Human-readable error message.
        error_type: Error type identifier.
        param: Optional request parameter associated with the error.
        code: Optional machine-readable code.

    Returns:
        Error payload with the OpenAI-compatible `error` envelope.
    """

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def build_openai_error_response(
    *,
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-style error response payload.

    Args:
        message: Human-readable error message.
        error_type: Error type identifier.
        param: Optional request parameter associated with the error.
        code: Optional machine-readable code.

    Returns:
        Error payload with the OpenAI-compatible `error` envelope.
    """

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
