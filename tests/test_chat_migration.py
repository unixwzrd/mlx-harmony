from __future__ import annotations

import json
from pathlib import Path

from mlx_harmony.conversation.conversation_io import load_conversation
from mlx_harmony.conversation.conversation_migration import migrate_chat_data, validate_chat_container


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_migrate_v1_to_v2_schema(test_data_dir: Path) -> None:
    data = _load_fixture(test_data_dir / "chat_schema_v1.json")
    migrated, active_chat_id, report = migrate_chat_data(data)

    assert migrated["schema_version"] == 2
    assert active_chat_id in migrated["chats"]
    metadata = migrated["chats"][active_chat_id]["metadata"]
    assert metadata["model_path"] == "models/old-model"
    assert metadata["prompt_config_path"] == "configs/old-config.json"
    assert report["changes"]


def test_validate_chat_container_updates_metadata(test_data_dir: Path) -> None:
    data = _load_fixture(test_data_dir / "chat_schema_v2.json")
    container, active_chat_id, _ = migrate_chat_data(data)
    # Force updated_at to be stale
    container["chats"][active_chat_id]["metadata"]["updated_at"]["unix"] = 0.0
    updated, warnings = validate_chat_container(container, active_chat_id)

    assert warnings
    updated_at = updated["chats"][active_chat_id]["metadata"]["updated_at"]["unix"]
    assert updated_at > 0.0


def test_load_conversation_accepts_v1_format(test_data_dir: Path, temp_dir: Path) -> None:
    source = test_data_dir / "chat_schema_v1.json"
    target = temp_dir / "legacy_chat.json"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    messages, metadata = load_conversation(target)
    assert len(messages) == 2
    assert metadata["model_path"] == "models/old-model"
    assert "chat_id" in metadata
