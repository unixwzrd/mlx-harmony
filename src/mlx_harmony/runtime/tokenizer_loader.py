"""
Tokenizer loader for ByteLevel BPE tokenizers.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlx_harmony.runtime.tokenizer_bpe import ByteLevelBPETokenizer


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

    tokenizer_json_path = model_path / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_json_path}")

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    model_config = tokenizer_data.get("model", {})
    model_type = model_config.get("type", "")

    if model_type != "BPE":
        raise ValueError(
            f"Unsupported tokenizer type: {model_type}. Only BPE tokenizers are supported."
        )

    vocab = model_config.get("vocab", {})
    if not vocab:
        raise ValueError("Vocabulary not found in tokenizer.json")

    merges = model_config.get("merges", [])
    if not merges:
        raise ValueError("BPE merges not found in tokenizer.json")

    merge_pairs: list[tuple[str, str]] = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merge_pairs.append((merge[0], merge[1]))
        elif isinstance(merge, str):
            parts = merge.split()
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))
        else:
            raise ValueError(f"Invalid merge format: {merge}")

    special_tokens: dict[str, str] = {}
    added_tokens = tokenizer_data.get("added_tokens", [])
    for token_info in added_tokens:
        if isinstance(token_info, dict):
            content = token_info.get("content", "")
            token_id = token_info.get("id")
            special_type = token_info.get("special", False)

            if token_id is not None:
                vocab[content] = token_id

                if special_type:
                    if (
                        "eos" in content.lower()
                        or "endoftext" in content.lower()
                        or "return" in content.lower()
                    ):
                        special_tokens["eos_token"] = content
                    elif "bos" in content.lower() or "startoftext" in content.lower():
                        special_tokens["bos_token"] = content
                    elif "pad" in content.lower():
                        special_tokens["pad_token"] = content
                    elif "unk" in content.lower():
                        special_tokens["unk_token"] = content

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
