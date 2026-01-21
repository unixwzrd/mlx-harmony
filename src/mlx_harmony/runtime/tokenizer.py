from __future__ import annotations

from typing import Protocol, Sequence


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer implementations used by the generation loop."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into text."""
        ...

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID into text."""
        ...
