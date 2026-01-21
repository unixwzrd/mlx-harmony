from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def make_timestamp() -> dict[str, str | float]:
    """Return a UTC timestamp dict with unix and iso fields."""
    dt = datetime.utcnow()
    return {
        "unix": dt.timestamp(),
        "iso": dt.isoformat() + "Z",
    }


def make_message_id() -> str:
    """Return a stable message ID for chat logs."""
    return uuid4().hex


def make_chat_id() -> str:
    """Return a stable chat ID for chat logs."""
    return uuid4().hex


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


def _write_debug_block(debug_path: Path, header: str, payload: str) -> None:
    """Write a labeled debug section to the debug log."""
    with open(debug_path, "a", encoding="utf-8") as df:
        separator = "=" * 80
        df.write(f"\n{separator}\n")
        df.write(f"{header} [BEGIN]\n")
        df.write(f"{separator}\n")
        df.write(payload)
        if not payload.endswith("\n"):
            df.write("\n")
        df.write(f"{separator}\n")
        df.write(f"{header} [END]\n")
        df.write(f"{separator}\n")


def write_debug_prompt(
    *,
    debug_path: Path,
    raw_prompt: str,
    show_console: bool,
) -> None:
    """Write a raw prompt block to the debug log and optionally console."""
    if show_console:
        print("\n[DEBUG] Raw prompt sent to LLM:")
        print("-" * 80)
        print(raw_prompt)
        print("-" * 80)
    _write_debug_block(debug_path, "[DEBUG] Raw prompt sent to LLM:", raw_prompt)


def write_debug_response(
    *,
    debug_path: Path,
    raw_response: str,
    cleaned_response: str,
    show_console: bool,
) -> None:
    """Write a raw response block to the debug log and optionally console."""
    if show_console:
        print("\n[DEBUG] Raw response from LLM:")
        print("-" * 80)
        print(cleaned_response)
        print("-" * 80)
    _write_debug_block(debug_path, "[DEBUG] Raw response from LLM:", raw_response)


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
        except Exception as e:
            logger.error("Failed to load chat: %s", e)
            raise SystemExit(1) from e
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


def write_debug_tokens(
    *,
    debug_path: Path,
    token_ids: list[int],
    decode_tokens: Callable[[list[int]], str] | None = None,
    label: str = "response",
    enabled: bool = True,
) -> None:
    """Write token IDs and decoded text to the debug log."""
    if not enabled or not token_ids:
        return
    _write_debug_block(
        debug_path,
        f"[DEBUG] {label} tokens ({len(token_ids)} IDs):",
        str(token_ids),
    )
    if decode_tokens is not None:
        decoded_all = decode_tokens(token_ids)
        _write_debug_block(
            debug_path,
            f"[DEBUG] {label} tokens decoded (raw):",
            decoded_all,
        )


def write_debug_metrics(
    *,
    debug_path: Path,
    metrics: dict[str, Any],
) -> None:
    """Write generation metrics to the debug log."""
    header = _format_metrics_tsv_header(metrics)
    if header is not None:
        _write_debug_block(
            debug_path,
            "[DEBUG] Generation metrics (TSV header):",
            header,
        )
    _write_debug_block(
        debug_path,
        "[DEBUG] Generation metrics (TSV):",
        _format_metrics_tsv(metrics),
    )


def _format_metrics_tsv(metrics: dict[str, Any]) -> str:
    """Format metrics as a TSV line for easy extraction."""
    keys = [
        "prompt_tokens",
        "generated_tokens",
        "elapsed_seconds",
        "tokens_per_second",
        "prompt_start_to_prompt_start_seconds",
        "max_context_tokens",
    ]
    memory_keys = sorted(key for key in metrics if key.startswith("memory_"))
    keys.extend(memory_keys)
    values = [metrics.get(key, "") for key in keys]
    return "TIMING_STATS\t" + "\t".join(str(v) for v in values)


def _format_metrics_tsv_header(metrics: dict[str, Any]) -> str | None:
    """Format a TSV header line (only once) for easy extraction."""
    if getattr(_format_metrics_tsv_header, "_header_written", False):
        return None
    keys = [
        "prompt_tokens",
        "generated_tokens",
        "elapsed_seconds",
        "tokens_per_second",
        "prompt_start_to_prompt_start_seconds",
        "max_context_tokens",
    ]
    memory_keys = sorted(key for key in metrics if key.startswith("memory_"))
    keys.extend(memory_keys)
    _format_metrics_tsv_header._header_written = True
    return "TIMING_STATS_HEADER\t" + "\t".join(keys)


def find_last_assistant_message(
    conversation: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last assistant message in the conversation, if any."""
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
    """Display the last assistant response or analysis when resuming a chat."""
    if not conversation:
        return False

    logger.info("Resuming prior chat...")

    last_assistant_msg = find_last_assistant_message(conversation)
    if not last_assistant_msg:
        logger.info("No assistant messages found in conversation history.")
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
