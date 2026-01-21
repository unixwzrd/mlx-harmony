from __future__ import annotations

from typing import Protocol, Sequence

from mlx_harmony.runtime.tokenizer import TokenizerProtocol


class ModelBackend(Protocol):
    def prepare_prompt(
        self,
        *,
        prompt_tokens: list[int] | None,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        system_message: str | None,
    ) -> tuple[str | list[int], list[int] | None]:
        """Return prompt input plus optional prompt token list."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs to text for debug/logging."""

    def get_tokenizer(self) -> TokenizerProtocol:
        """Return the tokenizer instance for this backend."""

    def get_stop_tokens(self) -> list[int]:
        """Return token IDs that should stop generation for this backend."""
