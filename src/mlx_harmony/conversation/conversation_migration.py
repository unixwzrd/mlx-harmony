from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_harmony.conversation.conversation_history import (
    find_last_hyperparameters,
    make_chat_id,
    make_timestamp,
    normalize_timestamp,
)
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)

CURRENT_SCHEMA_VERSION = 2


def _new_report() -> dict[str, list[str]]:
    return {"changes": [], "renames": [], "added": []}


def _ensure_metadata_fields(
    metadata: dict[str, Any],
    *,
    chat_id: str,
    messages: list[dict[str, Any]],
    report: dict[str, list[str]],
) -> dict[str, Any]:
    if metadata.get("model") and not metadata.get("model_path"):
        metadata["model_path"] = metadata["model"]
        report["renames"].append("metadata.model -> metadata.model_path")
    if (
        metadata.get("prompt_config")
        and not metadata.get("prompt_config_path")
        and isinstance(metadata.get("prompt_config"), str)
    ):
        metadata["prompt_config_path"] = metadata["prompt_config"]
        report["renames"].append("metadata.prompt_config -> metadata.prompt_config_path")

    if not metadata.get("chat_id"):
        metadata["chat_id"] = chat_id
        report["added"].append("metadata.chat_id")

    if "created_at" not in metadata:
        metadata["created_at"] = make_timestamp()
        report["added"].append("metadata.created_at")
    else:
        metadata["created_at"] = normalize_timestamp(metadata["created_at"])

    metadata["updated_at"] = normalize_timestamp(metadata.get("updated_at"))
    if "updated_at" not in metadata:
        report["added"].append("metadata.updated_at")

    last_hyperparameters: dict[str, Any] = {}
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "hyperparameters" in msg:
            last_hyperparameters = msg["hyperparameters"]
            break
    if "hyperparameters" in metadata and metadata["hyperparameters"]:
        last_hyperparameters = metadata["hyperparameters"]

    metadata["last_hyperparameters"] = metadata.get(
        "last_hyperparameters", last_hyperparameters
    )
    if "last_hyperparameters" not in metadata:
        report["added"].append("metadata.last_hyperparameters")
    metadata["last_model_path"] = metadata.get(
        "last_model_path", metadata.get("model_path")
    )
    if "last_model_path" not in metadata:
        report["added"].append("metadata.last_model_path")
    metadata["last_prompt_config_path"] = metadata.get(
        "last_prompt_config_path", metadata.get("prompt_config_path")
    )
    if "last_prompt_config_path" not in metadata:
        report["added"].append("metadata.last_prompt_config_path")
    if "prompt_config" in metadata:
        metadata["last_prompt_config"] = metadata.get(
            "last_prompt_config", metadata.get("prompt_config")
        )
        if "last_prompt_config" not in metadata:
            report["added"].append("metadata.last_prompt_config")

    return metadata


def _normalize_chat_entry(
    chat_entry: dict[str, Any],
    *,
    chat_id: str,
    report: dict[str, list[str]],
) -> dict[str, Any]:
    messages = chat_entry.get("messages", [])
    metadata = chat_entry.get("metadata", {})
    metadata = _ensure_metadata_fields(
        metadata, chat_id=chat_id, messages=messages, report=report
    )
    chat_entry["chat_id"] = chat_id
    chat_entry["metadata"] = metadata
    chat_entry["messages"] = messages
    return chat_entry


def _migrate_v1_to_v2(
    data: dict[str, Any], report: dict[str, list[str]]
) -> dict[str, Any]:
    chat_id = data.get("metadata", {}).get("chat_id") or make_chat_id()
    messages = data.get("messages", [])
    metadata = data.get("metadata", {})
    chat_entry = _normalize_chat_entry(
        {"chat_id": chat_id, "metadata": metadata, "messages": messages},
        chat_id=chat_id,
        report=report,
    )
    report["changes"].append("schema v1 -> v2 container")
    report["added"].extend(
        ["schema_version", "chat_order", "active_chat_id", "chats"]
    )
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "chat_order": [chat_id],
        "active_chat_id": chat_id,
        "chats": {chat_id: chat_entry},
    }


def migrate_chat_data(
    data: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, list[str]]]:
    report = _new_report()
    schema_version = data.get("schema_version")
    if not schema_version:
        migrated = _migrate_v1_to_v2(data, report)
        return migrated, migrated["active_chat_id"], report

    if schema_version != CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Unsupported schema_version=%s; attempting best-effort normalization",
            schema_version,
        )

    chats = data.get("chats", {})
    if not isinstance(chats, dict):
        raise ValueError("Chat file format invalid: 'chats' must be an object")

    chat_order = data.get("chat_order")
    if not isinstance(chat_order, list) or not chat_order:
        chat_order = list(chats.keys())

    active_chat_id = data.get("active_chat_id")
    if not active_chat_id or active_chat_id not in chats:
        active_chat_id = chat_order[0] if chat_order else None
    if not active_chat_id:
        raise ValueError("Chat file contains no chats to load")

    normalized_chats: dict[str, Any] = {}
    for chat_id, chat_entry in chats.items():
        chat_report = _new_report()
        normalized_chats[chat_id] = _normalize_chat_entry(
            chat_entry, chat_id=chat_id, report=chat_report
        )
        for key in report:
            report[key].extend(chat_report[key])

    data["schema_version"] = CURRENT_SCHEMA_VERSION
    data["chat_order"] = chat_order
    data["active_chat_id"] = active_chat_id
    data["chats"] = normalized_chats
    return data, active_chat_id, report


def validate_chat_container(
    container: dict[str, Any], active_chat_id: str
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    chat_entry = container.get("chats", {}).get(active_chat_id)
    if not chat_entry:
        return container, [f"Active chat id '{active_chat_id}' not found"]

    metadata = chat_entry.get("metadata", {})
    messages = chat_entry.get("messages", [])

    last_timestamp = None
    for msg in messages:
        ts = msg.get("timestamp")
        if isinstance(ts, dict) and isinstance(ts.get("unix"), (int, float)):
            if last_timestamp is None or ts["unix"] > last_timestamp["unix"]:
                last_timestamp = ts

    updated_at = metadata.get("updated_at")
    if not updated_at or not isinstance(updated_at, dict):
        warnings.append("metadata.updated_at missing or invalid; resetting to last message timestamp")
        if last_timestamp:
            metadata["updated_at"] = last_timestamp
        else:
            metadata["updated_at"] = make_timestamp()
    elif last_timestamp and updated_at.get("unix", 0) < last_timestamp.get("unix", 0):
        warnings.append("metadata.updated_at older than last message timestamp; updating")
        metadata["updated_at"] = last_timestamp

    last_hyperparameters = find_last_hyperparameters(messages)
    if last_hyperparameters and metadata.get("last_hyperparameters") != last_hyperparameters:
        warnings.append("metadata.last_hyperparameters out of sync; updating")
        metadata["last_hyperparameters"] = last_hyperparameters

    chat_entry["metadata"] = metadata
    container["chats"][active_chat_id] = chat_entry
    return container, warnings


def build_chat_container(
    *,
    chat_id: str,
    metadata: dict[str, Any],
    messages: list[dict[str, Any]],
    existing_container: dict[str, Any] | None,
) -> dict[str, Any]:
    report = _new_report()
    metadata = _ensure_metadata_fields(
        metadata, chat_id=chat_id, messages=messages, report=report
    )
    chat_entry = {
        "chat_id": chat_id,
        "metadata": metadata,
        "messages": messages,
    }
    if not existing_container:
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "chat_order": [chat_id],
            "active_chat_id": chat_id,
            "chats": {chat_id: chat_entry},
        }

    existing_container["schema_version"] = CURRENT_SCHEMA_VERSION
    chat_order = existing_container.get("chat_order") or []
    if chat_id not in chat_order:
        chat_order.append(chat_id)
    existing_container["chat_order"] = chat_order
    existing_container["active_chat_id"] = chat_id
    existing_container.setdefault("chats", {})
    existing_container["chats"][chat_id] = chat_entry
    return existing_container


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate mlx-harmony chat logs to the latest schema.")
    parser.add_argument("chat_file", type=str, help="Path to a chat JSON file.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with the migrated schema.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write migrated output to a new file path.",
    )
    args = parser.parse_args()

    input_path = Path(args.chat_file)
    if not input_path.exists():
        raise SystemExit(f"Chat file not found: {input_path}")

    if not args.in_place and not args.output:
        raise SystemExit("Specify --in-place or --output for migration output.")

    with open(input_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    container, active_chat_id, report = migrate_chat_data(raw)
    container, warnings = validate_chat_container(container, active_chat_id)
    for warning in warnings:
        print(f"[WARN] {warning}")
    if report["changes"] or report["renames"] or report["added"]:
        print("[INFO] Migration changes:")
        for item in report["changes"]:
            print(f"  change: {item}")
        for item in report["renames"]:
            print(f"  rename: {item}")
        for item in report["added"]:
            print(f"  added: {item}")

    output_path = input_path if args.in_place else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(container, handle, indent=2, ensure_ascii=False)

    print(f"[INFO] Migrated chat saved to: {output_path}")
    return 0
