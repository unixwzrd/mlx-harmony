from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from mlx_harmony.conversation.ids import make_message_id, make_timestamp
from mlx_harmony.conversation.paths import normalize_chat_name
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def find_last_hyperparameters(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the most recent hyperparameters stored in the conversation."""
    for msg in reversed(conversation):
        if msg.get("role") == "assistant" and "hyperparameters" in msg:
            return msg["hyperparameters"]
    return {}


def ensure_message_links(conversation: list[dict[str, Any]]) -> None:
    """Ensure each message has id, parent_id, and cache_key fields."""
    prev_id: str | None = None
    for msg in conversation:
        msg_id = msg.get("id")
        if not msg_id:
            msg_id = make_message_id()
            msg["id"] = msg_id
        if "parent_id" not in msg:
            msg["parent_id"] = prev_id
        if not msg.get("cache_key"):
            msg["cache_key"] = msg_id
        prev_id = msg_id


def normalize_timestamp(ts: str | dict[str, str | float] | None) -> dict[str, str | float]:
    """Normalize timestamps to the canonical dict form used in chat logs."""
    if ts is None:
        return make_timestamp()
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return {
                "unix": dt.timestamp(),
                "iso": ts,
            }
        except Exception as exc:
            logger.warning(
                "Failed to parse timestamp '%s': %s (using current time instead)",
                ts,
                exc,
            )
            return make_timestamp()
    if isinstance(ts, dict) and "unix" in ts and "iso" in ts:
        return ts
    return make_timestamp()


def restore_chat_metadata(
    *,
    chat_arg: str | None,
    chat_file_path: Path | None,
    model_path: str,
    prompt_config_path: str | None,
    prompt_config: Any | None,
    loaded_metadata: dict[str, Any],
    load_prompt_config: Callable[[str], Any | None],
    resolve_dirs: Callable[[Any | None], tuple[Path, Path]],
) -> tuple[str, str | None, Any | None, Path | None, Path | None, Path | None, dict[str, Any]]:
    """Restore model/config paths and directories from loaded chat metadata."""
    original_chat_file_path = chat_file_path

    if not prompt_config_path and loaded_metadata.get("prompt_config_path"):
        prompt_config_path = loaded_metadata["prompt_config_path"]
        prompt_config = load_prompt_config(prompt_config_path) if prompt_config_path else None
        if prompt_config_path and prompt_config is None:
            raise RuntimeError(f"Prompt config not found: {prompt_config_path}")
        logger.info("Using prompt config from chat: %s", prompt_config_path)

    chats_dir: Path | None = None
    logs_dir: Path | None = None
    if prompt_config_path:
        chats_dir, logs_dir = resolve_dirs(prompt_config)
        if chat_arg:
            updated_chat_name = normalize_chat_name(chat_arg)
            new_chat_file_path = chats_dir / f"{updated_chat_name}.json"
            if original_chat_file_path and new_chat_file_path != original_chat_file_path:
                logger.info("Chat will be saved to: %s (per updated config)", new_chat_file_path)
            chat_file_path = new_chat_file_path

    if loaded_metadata.get("model_path") and not model_path:
        model_path = loaded_metadata["model_path"]
        logger.info("Using model from chat: %s", model_path)

    loaded_hyperparameters = loaded_metadata.get("hyperparameters", {})
    if loaded_hyperparameters:
        logger.info("Loaded hyperparameters from chat: %s", loaded_hyperparameters)

    return (
        model_path,
        prompt_config_path,
        prompt_config,
        chats_dir,
        logs_dir,
        chat_file_path,
        loaded_hyperparameters,
    )


def load_chat_session(
    *,
    load_file_path: Path | None,
    chat_file_path: Path | None,
    chat_arg: str | None,
    model_path: str,
    prompt_config_path: str | None,
    prompt_config: Any | None,
    load_conversation: Callable[[Path], tuple[list[dict[str, Any]], dict[str, Any]]],
    load_prompt_config: Callable[[str], Any | None],
    resolve_dirs: Callable[[Any | None], tuple[Path, Path]],
) -> tuple[
    list[dict[str, Any]],
    str,
    str | None,
    Any | None,
    Path | None,
    Path | None,
    Path | None,
    dict[str, Any],
    int | None,
    str | None,
    str | None,
]:
    """Load or initialize a chat session and merge any saved metadata."""
    conversation: list[dict[str, Any]] = []
    loaded_metadata: dict[str, Any] = {}
    loaded_hyperparameters: dict[str, Any] = {}
    loaded_max_context_tokens: int | None = None
    loaded_model_path: str | None = None
    loaded_chat_id: str | None = None

    updated_chats_dir: Path | None = None
    updated_logs_dir: Path | None = None

    if load_file_path and load_file_path.exists():
        try:
            conversation, loaded_metadata = load_conversation(load_file_path)
            ensure_message_links(conversation)
            logger.info("Loaded existing chat from: %s", load_file_path)
            if chat_file_path and load_file_path != chat_file_path:
                logger.info("Chat will be saved to: %s", chat_file_path)
            logger.info("Found %d previous messages (turns)", len(conversation))
            loaded_max_context_tokens = loaded_metadata.get("max_context_tokens")
            loaded_model_path = loaded_metadata.get("model_path")
            loaded_chat_id = loaded_metadata.get("chat_id")

            (
                model_path,
                prompt_config_path,
                prompt_config,
                updated_chats_dir,
                updated_logs_dir,
                chat_file_path,
                loaded_hyperparameters,
            ) = restore_chat_metadata(
                chat_arg=chat_arg,
                chat_file_path=chat_file_path,
                model_path=model_path,
                prompt_config_path=prompt_config_path,
                prompt_config=prompt_config,
                loaded_metadata=loaded_metadata,
                load_prompt_config=load_prompt_config,
                resolve_dirs=resolve_dirs,
            )
        except Exception as exc:
            logger.error("Failed to load chat: %s", exc)
            raise SystemExit(1) from exc
    elif chat_file_path:
        logger.info("Creating new chat: %s", chat_file_path)

    return (
        conversation,
        model_path,
        prompt_config_path,
        prompt_config,
        updated_chats_dir,
        updated_logs_dir,
        chat_file_path,
        loaded_hyperparameters,
        loaded_max_context_tokens,
        loaded_model_path,
        loaded_chat_id,
    )
