from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_harmony.conversation.ids import make_timestamp
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def normalize_chat_name(chat_input: str) -> str:
    """Normalize chat name by stripping directory and .json suffixes."""
    chat_path = Path(chat_input)
    chat_name = chat_path.name
    while chat_name.endswith(".json"):
        chat_name = chat_name[:-5]
    return chat_name


def normalize_dir_path(path_str: str) -> Path:
    """Normalize directory path to prevent nested logs/log/ structures."""
    if not path_str:
        return Path("logs")

    path = Path(path_str)
    if path.is_absolute():
        return path

    parts = [p for p in path.parts if p and p != "."]

    cleaned_parts = []
    prev_was_logs = False
    for part in parts:
        if part == "logs":
            if not prev_was_logs:
                cleaned_parts.append(part)
                prev_was_logs = True
        else:
            cleaned_parts.append(part)
            prev_was_logs = False

    if len(cleaned_parts) >= 2 and cleaned_parts[-2] == "logs" and cleaned_parts[-1] == "log":
        cleaned_parts = cleaned_parts[:-1]

    if not cleaned_parts or (len(cleaned_parts) == 1 and cleaned_parts[0] == "log"):
        cleaned_parts = ["logs"]

    return Path(*cleaned_parts)


def resolve_chat_paths(
    chat_arg: str | None,
    chats_dir: Path,
) -> tuple[Path | None, Path | None, str | None, Path | None]:
    """Resolve chat file paths for load/save based on a chat name or path."""
    chat_file_path: Path | None = None
    load_file_path: Path | None = None
    chat_name: str | None = None
    chat_input_path: Path | None = None

    if not chat_arg:
        return chat_file_path, load_file_path, chat_name, chat_input_path

    chat_input_path = Path(chat_arg)
    chat_input_filename = chat_input_path.name
    chat_name = normalize_chat_name(chat_arg)
    chat_file_path = chats_dir / f"{chat_name}.json"
    load_file_path = chat_file_path

    if not chat_file_path.exists():
        search_dirs = [
            chats_dir,
            Path("logs"),
            Path("logs/log"),
            Path("."),
        ]
        if chat_input_path.parent != Path(".") and str(chat_input_path.parent) != ".":
            search_dirs.append(chat_input_path.parent)

        seen = set()
        unique_search_dirs = []
        for d in search_dirs:
            d_str = str(d)
            if d_str not in seen and d_str not in ("", "."):
                seen.add(d_str)
                unique_search_dirs.append(d)

        for search_dir in unique_search_dirs:
            if not search_dir.exists():
                continue
            candidate = search_dir / f"{chat_name}.json"
            if candidate.exists():
                load_file_path = candidate
                break
            candidate = search_dir / f"{chat_name}.json.json"
            if candidate.exists():
                load_file_path = candidate
                break
            if chat_input_filename != f"{chat_name}.json":
                candidate = search_dir / chat_input_filename
                if candidate.exists():
                    load_file_path = candidate
                    break

    return chat_file_path, load_file_path, chat_name, chat_input_path


def resolve_debug_path(debug_file: str | None, logs_dir: Path) -> Path:
    """Resolve the debug log path, creating parent directories as needed."""
    if debug_file:
        debug_file_path = Path(debug_file)
        if debug_file_path.is_absolute():
            debug_path = debug_file_path
        else:
            debug_path = logs_dir / debug_file_path
    else:
        debug_path = logs_dir / "prompt-debug.log"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    return debug_path


def resolve_dirs_from_config(prompt_config: Any | None) -> tuple[Path, Path]:
    """Resolve chats/logs directories from prompt config or defaults."""
    chats_dir_str = prompt_config.chats_dir if prompt_config and prompt_config.chats_dir else "logs"
    logs_dir_str = prompt_config.logs_dir if prompt_config and prompt_config.logs_dir else "logs"

    chats_dir = normalize_dir_path(chats_dir_str)
    logs_dir = normalize_dir_path(logs_dir_str)

    chats_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return chats_dir, logs_dir
