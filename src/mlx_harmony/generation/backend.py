from __future__ import annotations

from typing import Protocol


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

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text for debug/logging."""
