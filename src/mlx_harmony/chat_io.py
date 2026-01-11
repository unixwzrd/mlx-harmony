from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_harmony.chat_history import make_timestamp, normalize_timestamp


def save_conversation(
    path: str | Path,
    messages: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None = None,
    tools: list | None = None,
    hyperparameters: dict[str, float | int | None] | None = None,
) -> None:
    """
    Save conversation to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    created_at = make_timestamp()
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
                existing_created_at = existing.get("metadata", {}).get("created_at", None)
                created_at = normalize_timestamp(existing_created_at)
        except Exception:
            pass

    now = make_timestamp()
    messages_with_timestamps: list[dict[str, Any]] = []
    latest_hyperparameters = hyperparameters or {}

    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "hyperparameters" in msg:
            latest_hyperparameters = msg["hyperparameters"]
            break

    for msg in messages:
        msg_copy = msg.copy()
        if "timestamp" not in msg_copy:
            msg_copy["timestamp"] = now
        else:
            msg_copy["timestamp"] = normalize_timestamp(msg_copy["timestamp"])
        messages_with_timestamps.append(msg_copy)

    metadata = {
        "model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "tools": [t.name for t in tools] if tools else [],
        "hyperparameters": latest_hyperparameters,
        "created_at": created_at,
        "updated_at": make_timestamp(),
    }

    data = {
        "metadata": metadata,
        "messages": messages_with_timestamps,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_conversation(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load conversation from a JSON file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    metadata = data.get("metadata", {})

    return messages, metadata
