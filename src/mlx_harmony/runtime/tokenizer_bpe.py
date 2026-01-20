"""
Pure Python ByteLevel BPE tokenizer implementation for mlx-harmony.
"""

from __future__ import annotations

from mlx_harmony.runtime.tokenizer_streaming import SimpleStreamingDetokenizer


class ByteLevelBPETokenizer:
    """Pure Python ByteLevel BPE tokenizer implementation."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: dict[str, str] | None = None,
        chat_template: str | None = None,
    ):
        """
        Initialize ByteLevel BPE tokenizer.

        Args:
            vocab: Dictionary mapping token strings to token IDs
            merges: List of BPE merge pairs (tuple of two strings)
            special_tokens: Optional dict of special token names to token strings
            chat_template: Optional chat template string for message formatting
        """
        self.vocab = vocab
        self.merges = merges

        self.id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}
        self.bpe_ranks: dict[tuple[str, str], int] = {
            (pair[0], pair[1]): idx for idx, pair in enumerate(merges)
        }

        self.special_tokens = special_tokens or {}
        self.eos_token_id = vocab.get(self.special_tokens.get("eos_token", "<|endoftext|>"), None)
        self.bos_token_id = vocab.get(self.special_tokens.get("bos_token", "<|endoftext|>"), None)
        self.pad_token_id = vocab.get(self.special_tokens.get("pad_token", "<|endoftext|>"), None)

        self.chat_template = chat_template
        self.default_chat_template = None

        self._bpe_cache: dict[str, list[str]] = {}
        self._byte_encoder = self._make_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        self.detokenizer = SimpleStreamingDetokenizer(self)

    @staticmethod
    def _make_byte_encoder() -> dict[int, str]:
        """
        Create GPT-2 style byte encoder.

        Maps bytes (0-255) to Unicode characters. Based on GPT-2's byte pair encoding.
        The space character (0x20) maps to 'Ġ' (U+0120).
        """
        byte_encoder: dict[int, str] = {}
        n = 0

        for i in range(256):
            if 33 <= i <= 126 or 161 <= i <= 255:
                byte_encoder[i] = chr(i)
            else:
                byte_encoder[i] = chr(256 + n)
                n += 1

        return byte_encoder

    def _byte_encode(self, text: str) -> str:
        """Encode text to byte-level Unicode characters."""
        return "".join(self._byte_encoder[b] for b in text.encode("utf-8"))

    def _byte_decode(self, byte_chars: str) -> str:
        """Decode byte-level Unicode characters to text."""
        byte_arr = bytearray()
        for char in byte_chars:
            byte_val = self._byte_decoder.get(char)
            if byte_val is not None:
                byte_arr.append(byte_val)
            else:
                byte_arr.extend(char.encode("utf-8"))
        try:
            return byte_arr.decode("utf-8")
        except UnicodeDecodeError:
            return byte_arr.decode("utf-8", errors="replace")

    def _apply_bpe(self, word: str) -> list[str]:
        """Apply BPE merges to a word."""
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        pairs: list[tuple[str, str]] = []
        word_chars = list(word)
        if len(word_chars) > 1:
            pairs = list(zip(word_chars[:-1], word_chars[1:]))

        if not pairs:
            self._bpe_cache[word] = word_chars
            return word_chars

        while True:
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            new_word_chars: list[str] = []
            i = 0
            while i < len(word_chars):
                if (
                    i < len(word_chars) - 1
                    and (word_chars[i], word_chars[i + 1]) == bigram
                ):
                    new_word_chars.append(word_chars[i] + word_chars[i + 1])
                    i += 2
                else:
                    new_word_chars.append(word_chars[i])
                    i += 1

            word_chars = new_word_chars
            if len(word_chars) == 1:
                break

            pairs = list(zip(word_chars[:-1], word_chars[1:]))

        self._bpe_cache[word] = word_chars
        return word_chars

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        add_prefix_space: bool = False,
    ) -> list[int]:
        """Encode text to token IDs using ByteLevel BPE."""
        if not text:
            return []

        if add_prefix_space and not text.startswith(" "):
            text = " " + text

        byte_text = self._byte_encode(text)
        bpe_tokens = self._apply_bpe(byte_text)

        token_ids: list[int] = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                raise ValueError(
                    f"Token '{token}' not found in vocabulary. "
                    f"Vocabulary size: {len(self.vocab)}. "
                    f"This indicates a BPE encoding error or vocabulary mismatch."
                )

        if add_special_tokens:
            if self.bos_token_id is not None:
                token_ids.insert(0, self.bos_token_id)
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int] | int,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        tokens: list[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in (
                self.eos_token_id,
                self.bos_token_id,
                self.pad_token_id,
            ):
                continue

            if token_id not in self.id_to_token:
                raise ValueError(
                    f"Token ID {token_id} not found in vocabulary. "
                    f"Vocabulary size: {len(self.vocab)}. "
                    f"Valid token ID range includes added_tokens from tokenizer.json."
                )

            token_str = self.id_to_token[token_id]
            is_special_token = (
                token_str.startswith("<|") and token_str.endswith("|>")
            ) or token_id in (self.eos_token_id, self.bos_token_id, self.pad_token_id)

            if skip_special_tokens and is_special_token:
                continue

            if is_special_token and not skip_special_tokens:
                continue

            tokens.append(token_str)

        byte_text = "".join(tokens)
        text = self._byte_decode(byte_text)
        text = text.replace("Ġ", " ")
        return text

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Apply chat template to messages (simple string replacement)."""
        if not self.chat_template:
            prompt_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")
            if add_generation_prompt:
                prompt_parts.append("Assistant: ")
            return "".join(prompt_parts)

        template = self.chat_template

        formatted_messages: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            elif role == "system":
                formatted_messages.append(f"System: {content}")

        message_text = "\n".join(formatted_messages)
        if add_generation_prompt:
            message_text += "\nAssistant: "

        template = template.replace("{{ messages }}", message_text)
        template = template.replace("{{ bos_token }}", self.special_tokens.get("bos_token", ""))
        template = template.replace("{{ eos_token }}", self.special_tokens.get("eos_token", ""))

        return template

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token ID."""
        return self._eos_token_id

    @eos_token_id.setter
    def eos_token_id(self, value: int | None) -> None:
        """Set EOS token ID."""
        self._eos_token_id = value

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID."""
        return self._bos_token_id

    @bos_token_id.setter
    def bos_token_id(self, value: int | None) -> None:
        """Set BOS token ID."""
        self._bos_token_id = value

    @property
    def pad_token_id(self) -> int | None:
        """Return PAD token ID."""
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value: int | None) -> None:
        """Set PAD token ID."""
        self._pad_token_id = value

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary as dict (for compatibility with mlx-lm interface)."""
        return self.vocab.copy()
