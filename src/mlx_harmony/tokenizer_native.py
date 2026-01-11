"""
Pure Python native tokenizer implementation for mlx-harmony.

This module implements ByteLevel BPE tokenization in pure Python without external
dependencies (no Rust tokenizers, no sentencepiece, no jinja2). Based on GPT-2
style ByteLevel BPE as seen in mlx-examples and mlx-lm.

Supports:
- ByteLevel BPE tokenization (GPT-2/GPT-OSS style)
- Simple chat template handling (string replacement, no Jinja2)
- Streaming detokenization
"""

from __future__ import annotations

import json
from pathlib import Path


class SimpleStreamingDetokenizer:
    """Simple streaming detokenizer for ByteLevel BPE tokenizers."""

    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        """Initialize streaming detokenizer."""
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        """Reset the detokenizer state."""
        self.tokens: list[int] = []
        self._unflushed_bytes: str = ""
        self.text: str = ""
        self.last_segment: str = ""

    def add_token(self, token_id: int):
        """Add a token to the stream."""
        self.tokens.append(token_id)
        # Decode incrementally
        if token_id not in self.tokenizer.id_to_token:
            raise ValueError(
                f"Token ID {token_id} not found in vocabulary. "
                f"Vocabulary size: {len(self.tokenizer.vocab)}."
            )
        token_str = self.tokenizer.id_to_token[token_id]

        # Special tokens (Harmony formatting tokens) are literal strings, not byte-encoded
        # Skip them during streaming - they're not part of the actual text output
        if token_str.startswith("<|") and token_str.endswith("|>"):
            # This is a Harmony special token (e.g., <|start|>, <|end|>, <|message|>)
            # Skip it - it's not part of the decoded text
            self.last_segment = ""
            return

        # Regular BPE tokens are byte-encoded, add to unflushed bytes for decoding
        self._unflushed_bytes += token_str
        # Try to decode bytes (handle incomplete UTF-8 sequences for streaming)
        decoded = self.tokenizer._byte_decode(self._unflushed_bytes)
        if not decoded.endswith("\ufffd"):  # Not incomplete UTF-8 (replacement char)
            # Handle 'Ġ' prefix (space marker)
            decoded = decoded.replace("Ġ", " ")
            self.text += decoded
            self.last_segment = decoded
            self._unflushed_bytes = ""
        else:
            # Incomplete UTF-8, wait for more tokens (this is expected during streaming)
            self.last_segment = ""

    def finalize(self):
        """Finalize decoding (handle remaining bytes)."""
        if self._unflushed_bytes:
            decoded = self.tokenizer._byte_decode(self._unflushed_bytes)
            decoded = decoded.replace("Ġ", " ")
            self.text += decoded
            self.last_segment = decoded
            self._unflushed_bytes = ""


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

        # Build reverse vocab (token ID -> token string)
        self.id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}

        # Build BPE ranks from merges (lower rank = merge earlier)
        self.bpe_ranks: dict[tuple[str, str], int] = {
            (pair[0], pair[1]): idx for idx, pair in enumerate(merges)
        }

        # Special tokens
        self.special_tokens = special_tokens or {}
        self.eos_token_id = vocab.get(self.special_tokens.get("eos_token", "<|endoftext|>"), None)
        self.bos_token_id = vocab.get(self.special_tokens.get("bos_token", "<|endoftext|>"), None)
        self.pad_token_id = vocab.get(self.special_tokens.get("pad_token", "<|endoftext|>"), None)

        # Chat template (simple string-based, no Jinja2)
        self.chat_template = chat_template
        self.default_chat_template = None

        # BPE encoding cache
        self._bpe_cache: dict[str, list[str]] = {}

        # Build byte encoder/decoder (GPT-2 style)
        self._byte_encoder = self._make_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        # Create streaming detokenizer
        self.detokenizer = SimpleStreamingDetokenizer(self)

    @staticmethod
    def _make_byte_encoder() -> dict[int, str]:
        """
        Create GPT-2 style byte encoder.

        Maps bytes (0-255) to Unicode characters. Based on GPT-2's byte pair encoding.
        The space character (0x20) maps to 'Ġ' (U+0120), which is why you see
        'Ġ' prefix in GPT-2 tokenizers.

        Algorithm from: https://github.com/openai/gpt-2/blob/master/src/encoder.py
        """
        byte_encoder = {}
        n = 0

        # Range mapping:
        # 0-32, 127-160: Non-printable bytes -> Unicode chars 256+
        # 33-126, 161-255: Printable bytes -> Direct Unicode mapping
        for i in range(256):
            if 33 <= i <= 126 or 161 <= i <= 255:
                # Printable: map directly
                byte_encoder[i] = chr(i)
            else:
                # Non-printable: map to Unicode chars 256+
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
                # Fallback: treat as regular Unicode character
                byte_arr.extend(char.encode("utf-8"))
        try:
            return byte_arr.decode("utf-8")
        except UnicodeDecodeError:
            # Handle incomplete UTF-8 sequences (replace with replacement char)
            return byte_arr.decode("utf-8", errors="replace")

    def _apply_bpe(self, word: str) -> list[str]:
        """
        Apply BPE merges to a word.

        Based on CLIP tokenizer's BPE implementation, adapted for ByteLevel.
        """
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        # Start with individual characters
        pairs = []
        word_chars = list(word)
        if len(word_chars) > 1:
            pairs = list(zip(word_chars[:-1], word_chars[1:]))

        if not pairs:
            self._bpe_cache[word] = word_chars
            return word_chars

        # Apply merges in order of rank (lower rank = merge first)
        while True:
            # Find the lowest-ranked bigram
            bigram = min(
                pairs,
                key=lambda pair: self.bpe_ranks.get(pair, float("inf")),
            )
            if bigram not in self.bpe_ranks:
                break

            # Merge the bigram
            new_word_chars = []
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
        """
        Encode text to token IDs using ByteLevel BPE.

        ByteLevel BPE encoding process (GPT-2 style):
        1. Add prefix space if requested
        2. Convert entire text to UTF-8 bytes
        3. Map bytes to Unicode characters using byte encoder (space -> 'Ġ')
        4. Apply BPE merges to the byte-level Unicode string
        5. Map BPE tokens to token IDs

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            add_prefix_space: Whether to add prefix space (ByteLevel behavior)

        Returns:
            List of token IDs
        """
        if not text:
            return []

        # Add prefix space if requested (ByteLevel behavior)
        if add_prefix_space and not text.startswith(" "):
            text = " " + text

        # Step 1: Convert text to UTF-8 bytes, then to byte-level Unicode chars
        # This automatically handles whitespace -> 'Ġ' mapping
        byte_text = self._byte_encode(text)

        # Step 2: Apply BPE merges to the byte-level Unicode string
        # The BPE algorithm will merge pairs according to the merges list
        # Tokens with 'Ġ' prefix come from space bytes and are in the vocab
        bpe_tokens = self._apply_bpe(byte_text)

        # Step 3: Map BPE tokens to token IDs
        token_ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Unknown token - this shouldn't happen with proper BPE
                # Fail fast instead of silently replacing with unknown token
                raise ValueError(
                    f"Token '{token}' not found in vocabulary. "
                    f"Vocabulary size: {len(self.vocab)}. "
                    f"This indicates a BPE encoding error or vocabulary mismatch."
                )

        # Add special tokens
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
        """
        Decode token IDs to text.

        Handles Harmony special tokens (token IDs >= 200000) by skipping them
        if skip_special_tokens is True, or converting them to their string representation.

        Args:
            token_ids: Token ID(s) to decode
            skip_special_tokens: Whether to skip special tokens (including Harmony tokens)

        Returns:
            Decoded text
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        # Convert token IDs to token strings
        tokens = []
        for token_id in token_ids:
            # Handle base tokenizer special tokens
            if skip_special_tokens and token_id in (
                self.eos_token_id,
                self.bos_token_id,
                self.pad_token_id,
            ):
                continue

            # Check if token ID is in vocabulary (includes added_tokens from tokenizer.json)
            if token_id not in self.id_to_token:
                # Token ID is not in vocabulary - this shouldn't happen for valid tokens
                # If skip_special_tokens is False, we could add a placeholder, but for now
                # we'll raise an error to catch any issues
                raise ValueError(
                    f"Token ID {token_id} not found in vocabulary. "
                    f"Vocabulary size: {len(self.vocab)}. "
                    f"Valid token ID range includes added_tokens from tokenizer.json."
                )

            token_str = self.id_to_token[token_id]

            # Check if this is a special token (Harmony formatting tokens are literal strings)
            is_special_token = (
                token_str.startswith("<|") and token_str.endswith("|>")
            ) or token_id in (self.eos_token_id, self.bos_token_id, self.pad_token_id)

            # Skip special tokens if requested
            if skip_special_tokens and is_special_token:
                continue

            # Special tokens are literal strings, not byte-encoded
            # Regular BPE tokens are byte-encoded and need decoding
            if is_special_token and not skip_special_tokens:
                # For special tokens, we can't mix them with byte-decoded text
                # If user wants to see them, we'd need to return them separately
                # For now, skip them even if skip_special_tokens=False to avoid corruption
                # (The Harmony parser handles these tokens separately anyway)
                continue

            tokens.append(token_str)

        # Join tokens and decode bytes (only regular BPE tokens should be here now)
        byte_text = "".join(tokens)
        text = self._byte_decode(byte_text)

        # Handle 'Ġ' prefix (space marker in ByteLevel BPE)
        text = text.replace("Ġ", " ")

        return text

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Apply chat template to messages (simple string replacement, no Jinja2).

        This is a simplified chat template implementation that uses string
        replacement instead of Jinja2 templating. It supports common patterns
        like {{ messages }}, {{ bos_token }}, {{ eos_token }}.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add assistant prompt at the end

        Returns:
            Formatted prompt string
        """
        if not self.chat_template:
            # Fallback: simple format
            prompt_parts = []
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

        # Simple template replacement (no Jinja2)
        # Replace {{ messages }} with formatted messages
        template = self.chat_template

        # Format messages (basic implementation)
        formatted_messages = []
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

        # Replace template variables
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
    def eos_token_id(self, value: int | None):
        """Set EOS token ID."""
        self._eos_token_id = value

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID."""
        return self._bos_token_id

    @bos_token_id.setter
    def bos_token_id(self, value: int | None):
        """Set BOS token ID."""
        self._bos_token_id = value

    @property
    def pad_token_id(self) -> int | None:
        """Return PAD token ID."""
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value: int | None):
        """Set PAD token ID."""
        self._pad_token_id = value

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary as dict (for compatibility with mlx-lm interface)."""
        return self.vocab.copy()


def load_tokenizer_native(
    model_path: str | Path,
) -> ByteLevelBPETokenizer:
    """
    Load a ByteLevel BPE tokenizer from a model directory.

    Args:
        model_path: Path to model directory containing tokenizer.json

    Returns:
        ByteLevelBPETokenizer instance

    Raises:
        FileNotFoundError: If tokenizer.json is not found
        ValueError: If tokenizer format is not supported
    """
    model_path = Path(model_path)

    # Load tokenizer.json
    tokenizer_json_path = model_path / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json_path}")

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract model configuration
    model_config = tokenizer_data.get("model", {})
    model_type = model_config.get("type", "")

    if model_type != "BPE":
        raise ValueError(
            f"Unsupported tokenizer type: {model_type}. Only BPE tokenizers are supported."
        )

    # Extract vocab
    vocab = model_config.get("vocab", {})
    if not vocab:
        raise ValueError("Vocabulary not found in tokenizer.json")

    # Extract merges
    merges = model_config.get("merges", [])
    if not merges:
        raise ValueError("BPE merges not found in tokenizer.json")

    # Convert merges to list of tuples
    merge_pairs = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merge_pairs.append((merge[0], merge[1]))
        elif isinstance(merge, str):
            # Handle space-separated merge format
            parts = merge.split()
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))
        else:
            raise ValueError(f"Invalid merge format: {merge}")

    # Extract and add special tokens from added_tokens array
    # These are tokens with IDs outside the main vocab range (e.g., Harmony tokens 200006+)
    special_tokens = {}
    added_tokens = tokenizer_data.get("added_tokens", [])
    for token_info in added_tokens:
        if isinstance(token_info, dict):
            content = token_info.get("content", "")
            token_id = token_info.get("id")
            special_type = token_info.get("special", False)

            # Add all added tokens to vocab (they have IDs outside the main vocab range)
            if token_id is not None:
                vocab[content] = token_id

                # Identify special token types
                if special_type:
                    if "eos" in content.lower() or "endoftext" in content.lower() or "return" in content.lower():
                        special_tokens["eos_token"] = content
                    elif "bos" in content.lower() or "startoftext" in content.lower():
                        special_tokens["bos_token"] = content
                    elif "pad" in content.lower():
                        special_tokens["pad_token"] = content
                    elif "unk" in content.lower():
                        special_tokens["unk_token"] = content

    # Try to find special tokens in vocab (fallback)
    if "eos_token" not in special_tokens:
        for token_str in ["<|endoftext|>", "</s>", "<eos>"]:
            if token_str in vocab:
                special_tokens["eos_token"] = token_str
                break

    if "bos_token" not in special_tokens:
        for token_str in ["<|startoftext|>", "<s>", "<bos>"]:
            if token_str in vocab:
                special_tokens["bos_token"] = token_str
                break

    # Load chat template from tokenizer_config.json (if available)
    # Fail fast if file exists but can't be read/parsed
    chat_template = None
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            chat_template = config.get("chat_template")

    return ByteLevelBPETokenizer(
        vocab=vocab,
        merges=merge_pairs,
        special_tokens=special_tokens,
        chat_template=chat_template,
    )
