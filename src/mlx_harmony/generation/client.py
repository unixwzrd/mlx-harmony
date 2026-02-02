from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from mlx_harmony.generator import TokenGenerator
from mlx_harmony.api_client import ApiClient, ApiClientConfig


@dataclass(frozen=True)
class GenerationRequest:
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
    text: str
    analysis_text: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class GenerationClient(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


class LocalGenerationClient:
    def __init__(self, generator: "TokenGenerator") -> None:
        self._generator = generator

    def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt_tokens = len(self._generator.render_prompt_tokens(request.messages))
        token_ids = list(
            self._generator.generate(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                min_p=request.min_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                repetition_context_size=request.repetition_context_size,
            )
        )
        completion_tokens = len(token_ids)
        if hasattr(self._generator, "tokenizer") and hasattr(self._generator.tokenizer, "decode"):
            text = self._generator.tokenizer.decode([int(t) for t in token_ids])
        else:
            text = ""
        finish_reason = self._generator.last_finish_reason or "stop"
        if not isinstance(finish_reason, str):
            finish_reason = "stop"
        return GenerationResult(
            text=text,
            analysis_text=None,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class ServerGenerationClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        model: Optional[str],
        profile: Optional[str],
        prompt_config: Optional[str],
        max_tokens: int,
        timeout: int,
        requests_log: Optional[str],
        return_analysis: bool = True,
    ) -> None:
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
