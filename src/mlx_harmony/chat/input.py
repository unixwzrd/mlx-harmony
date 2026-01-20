from __future__ import annotations

from typing import Any

from mlx_harmony.cli.chat_commands import parse_command
from mlx_harmony.cli.chat_voice import listen_for_user_input
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
