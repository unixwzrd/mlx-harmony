from __future__ import annotations

"""HTTP generation client abstractions for server-backed execution.

This module intentionally models request/response transport for API calls.
Local in-process generation stays in the CLI turn pipeline and does not live
here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from mlx_harmony.api_client import ApiClient, ApiClientConfig


@dataclass(frozen=True)
class GenerationRequest:
    """Generation request payload sent to the server."""

    messages: list[dict[str, str]]
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 0.0
    repetition_context_size: int = 20


@dataclass(frozen=True)
class GenerationResult:
    """Normalized generation result from the server."""

    text: str
    analysis_text: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class GenerationClient(Protocol):
    """Protocol for generation transports."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a response for a request."""
        ...


class ServerGenerationClient:
    """Generation transport backed by the HTTP API server."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        model: Optional[str],
        profile: Optional[str],
        prompt_config: Optional[str],
        max_tokens: Optional[int],
        timeout: int,
        requests_log: Optional[str],
        return_analysis: bool = True,
    ) -> None:
        """Create an HTTP generation client.

        Args:
            host: Server host.
            port: Server port.
            model: Optional model identifier.
            profile: Optional profile name.
            prompt_config: Optional prompt config name/path.
            max_tokens: Optional default max tokens for requests.
            timeout: HTTP timeout seconds.
            requests_log: Optional path for raw request/response logs.
            return_analysis: Whether to request analysis channel text.
        """
        config = ApiClientConfig(
            host=host,
            port=port,
            model=model,
            profile=profile,
            prompt_config=prompt_config,
            max_tokens=max_tokens,
            timeout=timeout,
            return_analysis=return_analysis,
            requests_log=None if requests_log is None else Path(requests_log),
        )
        self._client = ApiClient(config)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Send a generation request and normalize the response."""
        body = self._client.send_messages(
            request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            return_analysis=True,
        )
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("Server response missing choices")
        message = choices[0].get("message", {})
        text = message.get("content", "") if isinstance(message, dict) else ""
        analysis_text = message.get("analysis") if isinstance(message, dict) else None
        finish_reason = choices[0].get("finish_reason") or "stop"
        usage = body.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return GenerationResult(
            text=text,
            analysis_text=str(analysis_text) if analysis_text else None,
            finish_reason=str(finish_reason),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
