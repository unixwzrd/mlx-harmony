"""
Streaming detokenizer utilities for ByteLevel BPE tokenizers.
"""

from __future__ import annotations


class SimpleStreamingDetokenizer:
    """Simple streaming detokenizer for ByteLevel BPE tokenizers."""

    def __init__(self, tokenizer: "ByteLevelBPETokenizer"):
        """Initialize streaming detokenizer."""
        self.tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        """Reset the detokenizer state."""
        self.tokens: list[int] = []
        self._unflushed_bytes: str = ""
        self.text: str = ""
        self.last_segment: str = ""

    def add_token(self, token_id: int) -> None:
        """Add a token to the stream."""
        self.tokens.append(token_id)
        if token_id not in self.tokenizer.id_to_token:
            raise ValueError(
                f"Token ID {token_id} not found in vocabulary. "
                f"Vocabulary size: {len(self.tokenizer.vocab)}."
            )
        token_str = self.tokenizer.id_to_token[token_id]

        if token_str.startswith("<|") and token_str.endswith("|>"):
            self.last_segment = ""
            return

        self._unflushed_bytes += token_str
        decoded = self.tokenizer._byte_decode(self._unflushed_bytes)
        if not decoded.endswith("\ufffd"):
            decoded = decoded.replace("Ġ", " ")
            self.text += decoded
            self.last_segment = decoded
            self._unflushed_bytes = ""
        else:
            self.last_segment = ""

    def finalize(self) -> None:
        """Finalize decoding (handle remaining bytes)."""
        if self._unflushed_bytes:
            decoded = self.tokenizer._byte_decode(self._unflushed_bytes)
            decoded = decoded.replace("Ġ", " ")
            self.text += decoded
            self.last_segment = decoded
            self._unflushed_bytes = ""
