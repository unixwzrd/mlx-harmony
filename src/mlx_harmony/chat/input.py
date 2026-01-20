from __future__ import annotations

import sys
from typing import Any

from mlx_harmony.chat.voice import listen_for_user_input
from mlx_harmony.cli.chat_commands import parse_command
from mlx_harmony.conversation.conversation_io import read_user_input, try_save_conversation
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def read_chat_input(
    *,
    moshi_stt: Any | None,
    moshi_config: Any | None,
) -> str:
    if moshi_stt is not None:
        user_input = listen_for_user_input(moshi_stt, moshi_config)
        if user_input:
            print(f"\n>> {user_input}")
        return user_input
    return read_user_input("\n>> ")


def handle_user_command(
    *,
    user_input: str,
    hyperparameters: dict[str, float | int | bool],
    chat_file_path: str | None,
    conversation: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None,
    tools: list[Any],
) -> bool:
    handled, should_apply, message, updates = parse_command(user_input, hyperparameters)
    if not handled:
        return False
    if message:
        print(message)
    if should_apply and updates:
        hyperparameters.update(updates)
        if chat_file_path and conversation:
            error = try_save_conversation(
                chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                tools,
                hyperparameters,
            )
            if error:
                logger.warning(
                    "Failed to save updated hyperparameters: %s (check file path permissions)",
                    error,
                )
    return True


def apply_user_token_limit(
    *,
    text: str,
    prompt_config: Any | None,
    generator: Any,
) -> str | None:
    if not prompt_config:
        return text
    max_user_tokens = prompt_config.max_user_tokens
    if max_user_tokens is None:
        max_user_tokens = prompt_config.max_tokens
    if max_user_tokens is None:
        return text
    if not text.strip():
        return text
    tokens = generator.tokenizer.encode(text)
    token_count = len(tokens)
    limit = max_user_tokens
    if token_count <= limit:
        return text
    if sys.stdin.isatty():
        print(
            f"[WARN] Input is {token_count} tokens (limit {limit})."
            " Proceed with truncated input? [y/N]: ",
            end="",
            flush=True,
        )
        response = read_user_input("").strip().lower()
        if not response.startswith("y"):
            print("[INFO] Input discarded. Please try again.")
            return None
    else:
        logger.warning(
            "Input is %s tokens (limit %s). Truncating for non-interactive input.",
            token_count,
            limit,
        )
    truncated = generator.tokenizer.decode(tokens[:limit])
    print("[INFO] Using truncated input.")
    return truncated
