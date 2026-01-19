from __future__ import annotations

import json
import select
import sys
from pathlib import Path
from typing import Any

from mlx_harmony.conversation.conversation_history import (
    ensure_message_links,
    find_last_hyperparameters,
    make_chat_id,
    make_timestamp,
    normalize_timestamp,
)
from mlx_harmony.conversation.conversation_migration import (
    build_chat_container,
    migrate_chat_data,
    validate_chat_container,
)
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def _drain_pasted_lines() -> list[str]:
    """Return any additional lines already buffered on stdin."""
    lines: list[str] = []
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        next_line = sys.stdin.readline()
        if next_line == "":
            break
        lines.append(next_line.rstrip("\n"))
    return lines


def read_user_input(prompt: str, continuation_prompt: str = "... ") -> str:
    """
    Read user input, supporting pasted multi-line content and explicit continuations.
    """
    first_line = input(prompt)
    if first_line.strip() == "\\":
        return _read_multiline_block(continuation_prompt)

    lines = [first_line]
    lines.extend(_drain_pasted_lines())

    while lines and lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input(continuation_prompt))

    return "\n".join(lines)


def read_user_input_from_first_line(
    first_line: str,
    continuation_prompt: str = "... ",
) -> str:
    """
    Read user input starting from a pre-read first line, supporting paste blocks.
    """
    if first_line.strip() == "\\":
        return _read_multiline_block(continuation_prompt)

    lines = [first_line]
    lines.extend(_drain_pasted_lines())

    while lines and lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input(continuation_prompt))

    return "\n".join(lines)


def _read_multiline_block(continuation_prompt: str) -> str:
    """Read lines until a line containing only '\\\\' is entered."""
    lines: list[str] = []
    while True:
        next_line = input(continuation_prompt)
        if next_line.strip() == "\\":
            break
        lines.append(next_line)
    return "\n".join(lines)


def save_conversation(
    path: str | Path,
    messages: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None = None,
    prompt_config: dict[str, Any] | None = None,
    tools: list | None = None,
    hyperparameters: dict[str, float | int | None] | None = None,
    max_context_tokens: int | None = None,
    chat_id: str | None = None,
) -> None:
    """
    Save conversation to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ensure_message_links(messages)
    created_at = make_timestamp()
    existing_container: dict[str, Any] | None = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing_raw = json.load(f)
            existing_container, existing_chat_id, _ = migrate_chat_data(existing_raw)
            if not chat_id:
                chat_id = existing_chat_id
            existing_created_at = (
                existing_container.get("chats", {})
                .get(chat_id, {})
                .get("metadata", {})
                .get("created_at")
            )
            if existing_created_at:
                created_at = normalize_timestamp(existing_created_at)
        except Exception as exc:
            raise ValueError(f"Existing chat file is not valid JSON: {path}") from exc

    now = make_timestamp()
    messages_with_timestamps: list[dict[str, Any]] = []
    latest_hyperparameters = hyperparameters or find_last_hyperparameters(messages)

    if not chat_id:
        chat_id = make_chat_id()

    for msg in messages:
        msg_copy = msg.copy()
        if "timestamp" not in msg_copy:
            msg_copy["timestamp"] = now
        else:
            msg_copy["timestamp"] = normalize_timestamp(msg_copy["timestamp"])
        messages_with_timestamps.append(msg_copy)

    metadata = {
        "chat_id": chat_id,
        "model_path": model_path,
        "last_model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "last_prompt_config_path": prompt_config_path,
        "prompt_config": prompt_config,
        "last_prompt_config": prompt_config,
        "tools": [t.name for t in tools] if tools else [],
        "hyperparameters": latest_hyperparameters,
        "last_hyperparameters": latest_hyperparameters,
        "max_context_tokens": max_context_tokens,
        "created_at": created_at,
        "updated_at": make_timestamp(),
    }

    data = build_chat_container(
        chat_id=chat_id,
        metadata=metadata,
        messages=messages_with_timestamps,
        existing_container=existing_container,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_conversation(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load conversation from a JSON file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in conversation file: {path}") from exc

    container, active_chat_id, report = migrate_chat_data(data)
    container, warnings = validate_chat_container(container, active_chat_id)
    if report["changes"] or report["renames"] or report["added"]:
        logger.info("Chat schema migration report:")
        for item in report["changes"]:
            logger.info("  change: %s", item)
        for item in report["renames"]:
            logger.info("  rename: %s", item)
        for item in report["added"]:
            logger.info("  added: %s", item)
    for warning in warnings:
        logger.warning("Chat consistency warning: %s", warning)
    chat_entry = container["chats"].get(active_chat_id)
    if not chat_entry:
        raise ValueError(f"Chat id not found in conversation file: {path}")

    messages = chat_entry.get("messages", [])
    metadata = chat_entry.get("metadata", {})
    metadata["chat_id"] = active_chat_id

    return messages, metadata


def try_save_conversation(
    path: str | Path,
    messages: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None = None,
    prompt_config: dict[str, Any] | None = None,
    tools: list | None = None,
    hyperparameters: dict[str, float | int | None] | None = None,
    max_context_tokens: int | None = None,
    chat_id: str | None = None,
) -> str | None:
    """
    Attempt to save a conversation and return an error message if it fails.
    """
    try:
        save_conversation(
            path,
            messages,
            model_path,
            prompt_config_path,
            prompt_config,
            tools,
            hyperparameters,
            max_context_tokens,
            chat_id,
        )
        return None
    except Exception as e:
        return str(e)
