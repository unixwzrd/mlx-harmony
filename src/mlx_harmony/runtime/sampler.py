from __future__ import annotations

from typing import Protocol


class SamplerProtocol(Protocol):
    """Protocol for sampler callables used in generation."""

    def __call__(self, logprobs: object) -> object:
        ...


class LogitsProcessorProtocol(Protocol):
    """Protocol for logits processors used in generation."""

    def __call__(self, tokens: object, logits: object) -> object:
        ...
