from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


def normalize_chat_name(chat_input: str) -> str:
    """
    Normalize chat name by stripping directory and .json suffixes.
    """
    chat_path = Path(chat_input)
    chat_name = chat_path.name
    while chat_name.endswith(".json"):
        chat_name = chat_name[:-5]
    return chat_name


def normalize_dir_path(path_str: str) -> Path:
    """
    Normalize directory path to prevent nested logs/log/ structures.
    """
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


def make_timestamp() -> dict[str, str | float]:
    dt = datetime.utcnow()
    return {
        "unix": dt.timestamp(),
        "iso": dt.isoformat() + "Z",
    }


def normalize_timestamp(ts: str | dict[str, str | float] | None) -> dict[str, str | float]:
    if ts is None:
        return make_timestamp()
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return {
                "unix": dt.timestamp(),
                "iso": ts,
            }
        except Exception:
            return make_timestamp()
    if isinstance(ts, dict) and "unix" in ts and "iso" in ts:
        return ts
    return make_timestamp()


def resolve_chat_paths(
    chat_arg: str | None,
    chats_dir: Path,
) -> tuple[Path | None, Path | None, str | None, Path | None]:
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
    chats_dir_str = prompt_config.chats_dir if prompt_config and prompt_config.chats_dir else "logs"
    logs_dir_str = prompt_config.logs_dir if prompt_config and prompt_config.logs_dir else "logs"

    chats_dir = normalize_dir_path(chats_dir_str)
    logs_dir = normalize_dir_path(logs_dir_str)

    chats_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return chats_dir, logs_dir


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
    original_chat_file_path = chat_file_path

    if not prompt_config_path and loaded_metadata.get("prompt_config_path"):
        prompt_config_path = loaded_metadata["prompt_config_path"]
        prompt_config = (
            load_prompt_config(prompt_config_path) if prompt_config_path else None
        )
        print(f"[INFO] Using prompt config from chat: {prompt_config_path}")

    chats_dir: Path | None = None
    logs_dir: Path | None = None
    if prompt_config_path:
        chats_dir, logs_dir = resolve_dirs(prompt_config)
        if chat_arg:
            updated_chat_name = normalize_chat_name(chat_arg)
            new_chat_file_path = chats_dir / f"{updated_chat_name}.json"
            if original_chat_file_path and new_chat_file_path != original_chat_file_path:
                print(f"[INFO] Chat will be saved to: {new_chat_file_path} (per updated config)")
            chat_file_path = new_chat_file_path

    if loaded_metadata.get("model_path") and not model_path:
        model_path = loaded_metadata["model_path"]
        print(f"[INFO] Using model from chat: {model_path}")

    loaded_hyperparameters = loaded_metadata.get("hyperparameters", {})
    if loaded_hyperparameters:
        print(f"[INFO] Loaded hyperparameters from chat: {loaded_hyperparameters}")

    return (
        model_path,
        prompt_config_path,
        prompt_config,
        chats_dir,
        logs_dir,
        chat_file_path,
        loaded_hyperparameters,
    )


def write_debug_prompt(
    *,
    debug_path: Path,
    raw_prompt: str,
    show_console: bool,
) -> None:
    if show_console:
        print("\n[DEBUG] Raw prompt sent to LLM:")
        print("-" * 80)
        print(raw_prompt)
        print("-" * 80)
    with open(debug_path, "a", encoding="utf-8") as df:
        df.write("\n[DEBUG] Raw prompt sent to LLM:\n")
        df.write("-" * 80 + "\n")
        df.write(raw_prompt + "\n")
        df.write("-" * 80 + "\n")


def write_debug_response(
    *,
    debug_path: Path,
    raw_response: str,
    cleaned_response: str,
    show_console: bool,
) -> None:
    if show_console:
        print("\n[DEBUG] Raw response from LLM:")
        print("-" * 80)
        print(cleaned_response)
        print("-" * 80)
    with open(debug_path, "a", encoding="utf-8") as df:
        df.write("\n[DEBUG] Raw response from LLM:\n")
        df.write("-" * 80 + "\n")
        df.write(raw_response + "\n")
        df.write("-" * 80 + "\n")


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
]:
    conversation: list[dict[str, Any]] = []
    loaded_metadata: dict[str, Any] = {}
    loaded_hyperparameters: dict[str, Any] = {}

    updated_chats_dir: Path | None = None
    updated_logs_dir: Path | None = None

    if load_file_path and load_file_path.exists():
        try:
            conversation, loaded_metadata = load_conversation(load_file_path)
            print(f"[INFO] Loaded existing chat from: {load_file_path}")
            if chat_file_path and load_file_path != chat_file_path:
                print(f"[INFO] Chat will be saved to: {chat_file_path}")
            print(f"[INFO] Found {len(conversation)} previous messages (turns)")

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
        except Exception as e:
            print(f"[ERROR] Failed to load chat: {e}")
            raise SystemExit(1) from e
    elif chat_file_path:
        print(f"[INFO] Creating new chat: {chat_file_path}")

    return (
        conversation,
        model_path,
        prompt_config_path,
        prompt_config,
        updated_chats_dir,
        updated_logs_dir,
        chat_file_path,
        loaded_hyperparameters,
    )


def write_debug_tokens(
    *,
    debug_path: Path,
    token_ids: list[int],
    decode_tokens: Callable[[list[int]], str] | None = None,
    label: str = "response",
    enabled: bool = True,
) -> None:
    if not enabled or not token_ids:
        return
    with open(debug_path, "a", encoding="utf-8") as df:
        df.write(f"\n[DEBUG] {label} tokens ({len(token_ids)} IDs):\n")
        df.write("-" * 80 + "\n")
        df.write(str(token_ids) + "\n")
        df.write("-" * 80 + "\n")
        if decode_tokens is not None:
            decoded_all = decode_tokens(token_ids)
            df.write(f"\n[DEBUG] {label} tokens decoded (raw):\n")
            df.write("-" * 80 + "\n")
            df.write(decoded_all + "\n")
            df.write("-" * 80 + "\n")


def find_last_assistant_message(
    conversation: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            return msg
    return None


def display_resume_message(
    conversation: list[dict[str, Any]],
    assistant_name: str,
    thinking_limit: int,
    response_limit: int,
    render_markdown: bool,
    display_assistant: Callable[[str, str, bool], None],
    display_thinking: Callable[[str, bool], None],
    truncate_text: Callable[[str, int], str],
) -> bool:
    if not conversation:
        return False

    print("[INFO] Resuming prior chat...")

    last_assistant_msg = find_last_assistant_message(conversation)
    if not last_assistant_msg:
        print("[INFO] No assistant messages found in conversation history.\n")
        return True

    content = last_assistant_msg.get("content", "")
    if content in ("[Analysis only - no final response]", "[No final response - see thinking above]"):
        analysis = last_assistant_msg.get("analysis", "")
        if analysis:
            display_analysis = truncate_text(analysis, thinking_limit)
            if display_analysis.strip():
                display_thinking(display_analysis, render_markdown)
                print("\n[WARNING] Previous turn only generated analysis. Check max_tokens or repetition_penalty.\n")
        return True

    if content:
        display_content = truncate_text(content, response_limit)
        display_assistant(display_content, assistant_name, render_markdown)
        return True

    analysis = last_assistant_msg.get("analysis", "")
    if analysis:
        display_analysis = truncate_text(analysis, thinking_limit)
        if display_analysis.strip():
            display_thinking(display_analysis, render_markdown)
    return True
