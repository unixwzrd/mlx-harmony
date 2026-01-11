from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from openai_harmony import Role, StreamableParser
from rich.console import Console
from rich.console import Console as RichConsole
from rich.markdown import Markdown
from unicodefix.transforms import clean_text

from mlx_harmony.config import apply_placeholders, load_profiles, load_prompt_config
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)

# Create a console instance for rich rendering
_console = Console()


# Type definitions for message records
class MessageDict(TypedDict, total=False):
    """Type definition for conversation messages."""
    role: Literal["user", "assistant", "tool", "system", "developer"]
    content: str
    timestamp: str
    name: str  # Optional: for tool messages
    recipient: str  # Optional: for tool messages
    channel: str  # Optional: for Harmony messages (e.g., "commentary", "analysis", "final")
    analysis: str  # Optional: for assistant messages with thinking/analysis
    hyperparameters: dict[str, float | int]  # Optional: hyperparameters used for generation


def get_assistant_name(prompt_config: Any | None) -> str:
    """Get assistant name from prompt config, defaulting to 'Assistant'."""
    if prompt_config and prompt_config.placeholders:
        return prompt_config.placeholders.get("assistant", "Assistant")
    return "Assistant"


def get_truncate_limits(prompt_config: Any | None) -> tuple[int, int]:
    """
    Get truncate limits from prompt config.

    Returns:
        (thinking_limit, response_limit) tuple with defaults (1000, 1000)
    """
    thinking_limit = (
        prompt_config.truncate_thinking
        if prompt_config and prompt_config.truncate_thinking is not None
        else 1000
    )
    response_limit = (
        prompt_config.truncate_response
        if prompt_config and prompt_config.truncate_response is not None
        else 1000
    )
    return (thinking_limit, response_limit)


def truncate_text(text: str, limit: int) -> str:
    """Truncate text to limit, appending '... [truncated]' if needed."""
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


def display_assistant(
    text: str,
    assistant_name: str,
    render_markdown: bool = True,
) -> None:
    """
    Display assistant text with consistent formatting.

    Args:
        text: Assistant response text
        assistant_name: Name to display (e.g., "Assistant", "Mia")
        render_markdown: Whether to render as markdown (default: True)
    """
    if not text:
        return

    if render_markdown:
        prefix = f"{assistant_name}: "
        print(prefix, end="")
        # Create a console with reduced width to account for the prefix
        # Rich needs to know the available width after the prefix
        # Get console width (Rich Console has a size property that returns (width, height))
        if _console and hasattr(_console, "size"):
            console_width = _console.size.width
        elif _console and hasattr(_console, "width"):
            console_width = _console.width
        else:
            console_width = 80  # Default terminal width
        prefix_length = len(prefix)
        available_width = max(console_width - prefix_length, 40)  # Minimum 40 chars
        # Create a temporary console with adjusted width for this output
        temp_console = RichConsole(width=available_width, legacy_windows=False)
        _render_markdown(text, render_markdown=True, console=temp_console)
        print()  # Extra newline after markdown
    else:
        print(f"{assistant_name}: {text}")


def display_thinking(text: str, render_markdown: bool = True) -> None:
    """Display thinking/analysis text with [THINKING - ...] prefix, optionally rendered as markdown."""
    if not text.strip():
        return

    prefix = "[THINKING - "
    print(prefix, end="")

    if render_markdown:
        # Create a console with reduced width to account for the prefix
        if _console and hasattr(_console, "size"):
            console_width = _console.size.width
        elif _console and hasattr(_console, "width"):
            console_width = _console.width
        else:
            console_width = 80  # Default terminal width
        prefix_length = len(prefix)
        available_width = max(console_width - prefix_length, 40)  # Minimum 40 chars
        temp_console = RichConsole(width=available_width, legacy_windows=False)
        _render_markdown(text, render_markdown=True, console=temp_console)
        print("]")  # Close the thinking bracket
    else:
        # Plain text fallback
        print(f"{text}]")

    print()  # Extra newline after thinking


def normalize_chat_name(chat_input: str) -> str:
    """
    Normalize chat name by stripping directory and .json suffixes.

    Args:
        chat_input: User-provided chat name (may include path or .json)

    Returns:
        Normalized chat name without directory or .json extension
    """
    chat_path = Path(chat_input)
    chat_name = chat_path.name  # Extract just filename
    # Strip .json suffix(es)
    while chat_name.endswith(".json"):
        chat_name = chat_name[:-5]
    return chat_name


def normalize_dir_path(path_str: str) -> Path:
    """
    Normalize directory path to prevent nested logs/log/ structures.

    Uses Path.parts to properly handle path components and removes
    redundant segments deterministically.
    """
    if not path_str:
        return Path("logs")

    path = Path(path_str)
    # If absolute, use as-is
    if path.is_absolute():
        return path

    # Convert to parts, normalize separators
    parts = path.parts
    # Remove empty and "." segments
    parts = [p for p in parts if p and p != "."]

    # Collapse repeated "logs" segments
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

    # Handle suffix ("logs", "log") -> ("logs",)
    if len(cleaned_parts) >= 2 and cleaned_parts[-2] == "logs" and cleaned_parts[-1] == "log":
        cleaned_parts = cleaned_parts[:-1]

    # If empty or still problematic, default to "logs"
    if not cleaned_parts or (len(cleaned_parts) == 1 and cleaned_parts[0] == "log"):
        cleaned_parts = ["logs"]

    return Path(*cleaned_parts)


def _render_markdown(text: str, render_markdown: bool = True, console: Any | None = None) -> None:
    """
    Render text as markdown using rich (similar to glow/mdless) if enabled and rich is available.

    Rich's Markdown class will:
    - Format markdown elements (headers, lists, code blocks, etc.) beautifully
    - Handle plain text gracefully (no formatting applied if not markdown)
    - Preserve newlines and structure

    This function also normalizes markdown by adding newlines before markdown elements
    (headers, lists) for proper formatting even if the model didn't emit explicit newlines.

    Args:
        text: Text content to render
        render_markdown: If True, attempt to render as markdown. If False, print as plain text.
        console: Optional Rich Console instance to use (for custom width). If None, uses module-level _console.
    """
    if not text:
        return

    # Use provided console or fall back to module-level console
    render_console = console if console is not None else _console

    # Render with rich if enabled
    # Rich handles both markdown and plain text gracefully
    if render_markdown and render_console is not None:
        try:
            # Normalize markdown: add newlines before markdown elements for proper formatting
            # This handles cases where the model generates markdown syntax but without explicit newlines
            # Only normalize headers and lists - do NOT touch code fences (risky regex)
            normalized_text = text

            # Add newline before headers if not already present
            # Pattern: match "###" or similar not preceded by newline
            normalized_text = re.sub(r"(.)(#{2,6}\s)", r"\1\n\2", normalized_text)

            # Add newline before list items if not already present
            # IMPORTANT: Only normalize if list marker is at start of line (not mid-sentence dashes)
            # Numbered lists: only if at start of line or after whitespace (not mid-sentence)
            normalized_text = re.sub(r"(\S)(\d+\.\s+)", r"\1\n\2", normalized_text)  # Numbered lists after word

            # Bullet lists: Rich's markdown parser only recognizes bullets at start of line
            # So we should NOT normalize mid-sentence dashes like "text - item" (that's a hyphen, not a bullet)
            # Only add newline if pattern is preceded by whitespace AND we're normalizing (mid-text case)
            # Actually, simpler: don't normalize bullets at all - Rich handles them correctly if they're at line start
            # The issue is Rich is interpreting "evening - cook" as a bullet because of our normalization
            # So let's remove bullet normalization entirely - only normalize numbered lists
            # Rich will correctly handle "- item" if it's already at the start of a line

            # Use rich to render markdown beautifully
            # Rich will format markdown elements including code blocks with syntax highlighting
            # Let Rich handle code blocks - don't try to normalize them (risky)
            markdown = Markdown(normalized_text)  # Use Rich default theme instead of hardcoding
            render_console.print(markdown)
        except Exception:
            # Fallback to plain text if rich rendering fails
            print(text, end="")
    else:
        # Plain text fallback (rich not available or markdown disabled)
        print(text, end="")


def save_conversation(
    path: str | Path,
    messages: list[MessageDict | dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None = None,
    tools: list | None = None,
    hyperparameters: dict[str, float | int | None] | None = None,
) -> None:
    """
    Save conversation to a JSON file.

    Each message (turn) includes a timestamp when it was created.
    Assistant messages include the hyperparameters used for that generation,
    allowing tracking of hyperparameter changes during the conversation.
    Metadata.hyperparameters stores the latest hyperparameters for quick restoration.

    Format:
    {
        "metadata": {
            "model_path": "...",
            "prompt_config_path": "...",
            "tools": ["browser", "python"],
            "hyperparameters": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 512,
                ...
            },
            "created_at": "2026-01-06T...",
            "updated_at": "2026-01-06T..."
        },
        "messages": [
            {"role": "user", "content": "...", "timestamp": "2026-01-06T..."},
            {
                "role": "assistant",
                "content": "...",
                "timestamp": "2026-01-06T...",
                "hyperparameters": {"temperature": 0.8, "top_p": 0.9, ...}
            },
            {"role": "tool", "name": "browser", "content": "...", "recipient": "assistant", "timestamp": "2026-01-06T..."}
        ]
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to preserve created_at
    created_at = datetime.utcnow().isoformat() + "Z"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("metadata", {}).get("created_at", created_at)
        except Exception:
            pass  # If we can't read it, use new timestamp

    # Ensure all messages have timestamps (add if missing for backward compatibility)
    # Extract latest hyperparameters from assistant messages for metadata
    now = datetime.utcnow().isoformat() + "Z"
    messages_with_timestamps = []
    latest_hyperparameters = hyperparameters or {}

    # Find the most recent assistant message's hyperparameters as fallback
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "hyperparameters" in msg:
            latest_hyperparameters = msg["hyperparameters"]
            break

    for msg in messages:
        msg_copy = msg.copy()
        if "timestamp" not in msg_copy:
            msg_copy["timestamp"] = now
        messages_with_timestamps.append(msg_copy)

    metadata = {
        "model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "tools": [t.name for t in tools] if tools else [],
        "hyperparameters": latest_hyperparameters,  # Latest hyperparameters from conversation
        "created_at": created_at,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    data = {
        "metadata": metadata,
        "messages": messages_with_timestamps,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_conversation(path: str | Path) -> tuple[list[MessageDict | dict[str, Any]], dict[str, Any]]:
    """
    Load conversation from a JSON file.

    Returns:
        (messages, metadata) tuple where:
        - messages: conversation history (turns) with timestamps
        - metadata: contains model, prompt_config, tools, and hyperparameters
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    metadata = data.get("metadata", {})

    return messages, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with any MLX-LM model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or Hugging Face repo (or set via --profile).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--python",
        dest="use_python",
        action="store_true",
        help="Enable Python tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--apply-patch",
        action="store_true",
        help="Enable apply_patch tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config/default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=None,
        help="Number of previous tokens used for repetition penalty.",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to JSON file with Harmony prompt configuration (GPT-OSS).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Named profile from a profiles JSON (see --profiles-file).",
    )
    parser.add_argument(
        "--profiles-file",
        type=str,
        default="configs/profiles.example.json",
        help="Path to profiles JSON (default: configs/profiles.example.json)",
    )
    parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Chat name (loads from chats_dir/<name>.json if exists, otherwise creates new chat).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: output raw prompts and responses as text.",
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        default=None,
        help="Path to write debug prompts/responses when debug is enabled (default: logs_dir/prompt-debug.log).",
    )
    parser.add_argument(
        "--mlock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal only). "
        "Can also be set in prompt config JSON. Use --mlock to enable or --no-mlock to disable. Default: None (use config or False)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        default=False,
        help="Disable markdown rendering for assistant responses (display as plain text).",
    )
    args = parser.parse_args()

    # Resolve profile/model/prompt_config
    profile_model = None
    profile_prompt_cfg = None
    if args.profile:
        profiles = load_profiles(args.profiles_file)
        if args.profile not in profiles:
            raise SystemExit(
                f"Profile '{args.profile}' not found in {args.profiles_file}"
            )
        profile = profiles[args.profile]
        profile_model = profile.get("model")
        profile_prompt_cfg = profile.get("prompt_config")

    model_path = args.model or profile_model
    if not model_path:
        raise SystemExit("Model must be provided via --model or --profile")

    prompt_config_path = args.prompt_config or profile_prompt_cfg
    prompt_config = (
        load_prompt_config(prompt_config_path) if prompt_config_path else None
    )

    # Resolve directories from config (defaults to "logs" if not specified)
    # Normalize paths to avoid nested directories like logs/log/
    # These will be re-resolved after loading chat if prompt_config changes
    chats_dir_str = prompt_config.chats_dir if prompt_config and prompt_config.chats_dir else "logs"
    logs_dir_str = prompt_config.logs_dir if prompt_config and prompt_config.logs_dir else "logs"

    chats_dir = normalize_dir_path(chats_dir_str)
    logs_dir = normalize_dir_path(logs_dir_str)

    chats_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve chat file path if specified
    # Handle case where user might provide filename with or without .json extension
    # Also handle cases where user includes directory path (extract just filename)
    chat_file_path: Path | None = None
    load_file_path: Path | None = None

    if args.chat:
        # Store original input for search (before normalization)
        chat_input_path = Path(args.chat)
        chat_input_filename = chat_input_path.name

        # Normalize chat name (strip directory and .json suffixes)
        chat_name = normalize_chat_name(args.chat)
        # Always use normalized chats_dir for saving (ignore any directory in user input)
        chat_file_path = chats_dir / f"{chat_name}.json"
        load_file_path = chat_file_path  # Default: load from expected location

        # If file doesn't exist at expected location, try to find it in common locations
        # (e.g., if it was saved with wrong path due to config change)
        # Also search using the original user input path in case they specified a directory
        if not chat_file_path.exists():
            # Try looking for any file matching the name (handles wrong paths from previous saves)
            # Search in common locations: expected location, logs/, nested logs/log/, chats_dir, project root
            # Also try the original user input path if it was different
            search_dirs = [
                chats_dir,
                Path("logs"),
                Path("logs/log"),  # Check nested location in case files were saved there
                Path("."),
            ]

            # If user provided a path with directory, also search there
            if chat_input_path.parent != Path(".") and str(chat_input_path.parent) != ".":
                search_dirs.append(chat_input_path.parent)

            # Remove duplicates while preserving order
            seen = set()
            unique_search_dirs = []
            for d in search_dirs:
                d_str = str(d)
                if d_str not in seen and d_str not in ("", "."):
                    seen.add(d_str)
                    unique_search_dirs.append(d)

            found_file_path = None
            for search_dir in unique_search_dirs:
                if not search_dir.exists():
                    continue

                # Try exact name
                candidate = search_dir / f"{chat_name}.json"
                if candidate.exists():
                    found_file_path = candidate
                    # Always use normalized path for future saves (never use nested paths)
                    expected_path = chats_dir / f"{chat_name}.json"
                    if candidate != expected_path:
                        print(f"[INFO] Found chat file at: {candidate} (will save to: {expected_path})")
                    # Set chat_file_path to normalized path for future saves
                    chat_file_path = expected_path
                    load_file_path = found_file_path  # Load from found location
                    break
                # Try with double .json (handles legacy files with wrong extension)
                candidate = search_dir / f"{chat_name}.json.json"
                if candidate.exists():
                    expected_path = chats_dir / f"{chat_name}.json"
                    print(f"[INFO] Found legacy chat file at: {candidate} (will save to: {expected_path})")
                    found_file_path = candidate
                    # Set chat_file_path to normalized path for future saves
                    chat_file_path = expected_path
                    load_file_path = found_file_path  # Load from found location
                    break
                # Also try the original filename from user input (in case they passed "log/mia-chat.json.json")
                if chat_input_filename != f"{chat_name}.json":
                    candidate = search_dir / chat_input_filename
                    if candidate.exists():
                        expected_path = chats_dir / f"{chat_name}.json"
                        print(f"[INFO] Found chat file at: {candidate} (will save to: {expected_path})")
                        found_file_path = candidate
                        chat_file_path = expected_path
                        load_file_path = found_file_path
                        break

    # Load conversation if chat file exists
    conversation: list[MessageDict | dict[str, Any]] = []
    loaded_metadata = {}
    loaded_hyperparameters = {}
    if load_file_path and load_file_path.exists():
        try:
            conversation, loaded_metadata = load_conversation(load_file_path)
            print(f"[INFO] Loaded existing chat from: {load_file_path}")
            if chat_file_path and load_file_path != chat_file_path:
                print(f"[INFO] Chat will be saved to: {chat_file_path}")
            print(f"[INFO] Found {len(conversation)} previous messages (turns)")

            # Optionally use loaded model/prompt_config if not explicitly set
            if not model_path and loaded_metadata.get("model_path"):
                model_path = loaded_metadata["model_path"]
                print(f"[INFO] Using model from chat: {model_path}")

            # Store original chat file path before potential reload
            original_chat_file_path = chat_file_path

            if not prompt_config_path and loaded_metadata.get("prompt_config_path"):
                prompt_config_path = loaded_metadata["prompt_config_path"]
                prompt_config = (
                    load_prompt_config(prompt_config_path) if prompt_config_path else None
                )
                print(f"[INFO] Using prompt config from chat: {prompt_config_path}")

                # Re-resolve directories after reloading prompt_config in case it changed
                # Normalize paths to avoid nested directories
                chats_dir_str = prompt_config.chats_dir if prompt_config and prompt_config.chats_dir else "logs"
                logs_dir_str = prompt_config.logs_dir if prompt_config and prompt_config.logs_dir else "logs"

                # Always normalize to prevent nested directories (even if config has "logs/log")
                chats_dir = normalize_dir_path(chats_dir_str)
                logs_dir = normalize_dir_path(logs_dir_str)

                chats_dir.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)

                # Re-resolve chat file path with updated directory
                # Always use normalized path for consistency (extract just filename, ignore directory from user input)
                if args.chat:
                    # Normalize chat name (strip directory and .json suffixes)
                    chat_name = normalize_chat_name(args.chat)
                    # Always use normalized chats_dir (ignore any directory in user input)
                    new_chat_file_path = chats_dir / f"{chat_name}.json"
                    # If we loaded from a different location, inform the user
                    if new_chat_file_path != original_chat_file_path:
                        print(f"[INFO] Chat will be saved to: {new_chat_file_path} (per updated config)")
                    chat_file_path = new_chat_file_path

            # Restore hyperparameters from conversation if not explicitly set via CLI
            loaded_hyperparameters = loaded_metadata.get("hyperparameters", {})
            if loaded_hyperparameters:
                print(f"[INFO] Loaded hyperparameters from chat: {loaded_hyperparameters}")
        except Exception as e:
            print(f"[ERROR] Failed to load chat: {e}")
            raise SystemExit(1) from e
    elif chat_file_path:
        print(f"[INFO] Creating new chat: {chat_file_path}")

    # Get mlock from prompt config if not explicitly set via CLI
    mlock = args.mlock
    if mlock is None and prompt_config:
        mlock = prompt_config.mlock

    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        mlock=mlock or False,  # Default to False if None
    )

    tools = []
    if generator.is_gpt_oss:
        tools = get_tools_for_model(
            browser=args.browser,
            python=args.use_python,
            apply_patch=args.apply_patch,
        )

    print(f"[INFO] Starting chat with model: {model_path}")
    if generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected - Harmony format enabled.")
        if tools:
            enabled = ", ".join(t.name for t in tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non-GPT-OSS model - using native chat template.")

    print("[INFO] Type 'q' or `Control-D` to quit.")
    print("[INFO] Type '\\help' to list all out-of-band commands.")
    print("[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7').")
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if chat_file_path:
        print(f"[INFO] Chat will be saved to: {chat_file_path}\n")

    # Get assistant name for display (need this before resume display)
    assistant_name = get_assistant_name(prompt_config)

    # Get truncate limits once (used multiple times)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)

    # Get render_markdown setting once (used multiple times)
    render_markdown = not args.no_markdown if hasattr(args, "no_markdown") else True

    # If resuming a prior chat, display resume message and last assistant output
    # Skip greeting if we have any conversation history
    has_conversation_history = bool(conversation)

    if has_conversation_history:
        print("[INFO] Resuming prior chat...")

        # Find and display the last assistant message
        last_assistant_msg = None
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break

        if last_assistant_msg:
            content = last_assistant_msg.get("content", "")

            # Check if it's a Harmony-formatted message (might be saved as analysis only)
            if content in ("[Analysis only - no final response]", "[No final response - see thinking above]"):
                # Handle case where only analysis was generated
                analysis = last_assistant_msg.get("analysis", "")
                if analysis:
                    # Analysis from JSON already has newlines preserved as \n characters
                    # Don't call clean_text() again - content was already cleaned when saved
                    display_analysis = truncate_text(analysis, thinking_limit)
                    if display_analysis.strip():  # Only print if there's actual analysis
                        display_thinking(display_analysis)
                        print("\n[WARNING] Previous turn only generated analysis. Check max_tokens or repetition_penalty.\n")
            elif content:  # Has content and it's not a special marker
                # Display the last assistant response
                # Content from JSON already has newlines preserved as \n characters
                # JSON.load() converts \n escape sequences to actual newlines
                # Don't call clean_text() again - content was already cleaned when saved
                # This ensures newlines are preserved exactly as saved
                display_content = truncate_text(content, response_limit)
                # Always print if we have content (even if it's short)
                display_assistant(display_content, assistant_name, render_markdown)
            else:
                # Content is empty - check if there's analysis we can show
                analysis = last_assistant_msg.get("analysis", "")
                if analysis:
                    # Analysis from JSON already has newlines preserved as \n characters
                    # Don't call clean_text() again - content was already cleaned when saved
                    display_analysis = truncate_text(analysis, thinking_limit)
                    if display_analysis.strip():
                        display_thinking(display_analysis)
        else:
            print("[INFO] No assistant messages found in conversation history.\n")

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info (or after resume message)
    # Only show greeting if we don't have conversation history
    if not has_conversation_history and prompt_config and prompt_config.assistant_greeting:
        greeting_text = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        # Use markdown rendering for greeting if enabled
        display_assistant(greeting_text, assistant_name, render_markdown)
        conversation.append(
            {
                "role": "assistant",
                "content": greeting_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    max_tool_iterations = 10  # Prevent infinite loops

    # Collect hyperparameters (CLI args take precedence over loaded values, then config, then defaults)
    # For Harmony models, use higher default max_tokens (1024) to allow for
    # both analysis and final channels, otherwise default to 512
    default_max_tokens = 1024 if (generator.is_gpt_oss and generator.use_harmony) else 512
    hyperparameters = {
        "max_tokens": (
            args.max_tokens
            if args.max_tokens is not None
            else (
                loaded_hyperparameters.get("max_tokens")
                or (prompt_config.max_tokens if prompt_config else None)
                or default_max_tokens
            )
        ),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else (
                loaded_hyperparameters.get("temperature")
                or (prompt_config.temperature if prompt_config else None)
            )
        ),
        "top_p": (
            args.top_p
            if args.top_p is not None
            else (
                loaded_hyperparameters.get("top_p")
                or (prompt_config.top_p if prompt_config else None)
            )
        ),
        "min_p": (
            args.min_p
            if args.min_p is not None
            else (
                loaded_hyperparameters.get("min_p")
                or (prompt_config.min_p if prompt_config else None)
            )
        ),
        "top_k": (
            args.top_k
            if args.top_k is not None
            else (
                loaded_hyperparameters.get("top_k")
                or (prompt_config.top_k if prompt_config else None)
            )
        ),
        "repetition_penalty": (
            args.repetition_penalty
            if args.repetition_penalty is not None
            else (
                loaded_hyperparameters.get("repetition_penalty")
                or (prompt_config.repetition_penalty if prompt_config else None)
            )
        ),
        "repetition_context_size": (
            args.repetition_context_size
            if args.repetition_context_size is not None
            else (
                loaded_hyperparameters.get("repetition_context_size")
                or (prompt_config.repetition_context_size if prompt_config else None)
            )
        ),
    }
    # Remove None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Debug file setup: always write debug log by default
    # --debug enables console output, --debug-file overrides the file path
    # Path resolution: absolute paths used as-is, relative paths resolved relative to logs_dir
    # This matches --chat behavior (relative paths resolved relative to chats_dir)
    # Always write to debug file by default (unless explicitly disabled in future)
    if args.debug_file:
        debug_file_path = Path(args.debug_file)
        # If absolute path, use as-is; otherwise resolve relative to logs_dir (consistent with --chat)
        if debug_file_path.is_absolute():
            debug_path = debug_file_path
        else:
            # Relative path: resolve relative to logs_dir (like --chat resolves relative to chats_dir)
            debug_path = logs_dir / debug_file_path
    else:
        # Default to logs_dir/prompt-debug.log (always write debug logs)
        debug_path = logs_dir / "prompt-debug.log"
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            user_input = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if user_input.strip().lower() == "q":
            break

        # Normalize input for command checking
        user_input_stripped = user_input.strip()
        user_input_lower = user_input_stripped.lower()

        # Store help text in constant to avoid duplication
        HELP_TEXT = (
            "\n[INFO] Out-of-band commands:\n"
            "  q, Control-D           - Quit the chat\n"
            "  \\help, /help          - Show this help message\n"
            "  \\list, /list          - List current hyperparameters\n"
            "  \\show, /show          - List current hyperparameters (alias for \\list)\n"
            "  \\set <param>=<value>  - Set a hyperparameter\n"
            "                          Example: \\set temperature=0.7\n"
            "                          Valid parameters: temperature, top_p, min_p, top_k,\n"
            "                          max_tokens, min_tokens_to_keep, repetition_penalty,\n"
            "                          repetition_context_size, xtc_probability, xtc_threshold\n"
        )

        # Handle help command: \help or /help (require prefix)
        if user_input_lower in ("\\help", "/help"):
            print(HELP_TEXT)
            continue

        # Handle hyperparameter listing: \list or \show
        if user_input_lower in ("\\list", "/list", "\\show", "/show"):
            print("\n[INFO] Current hyperparameters:")
            if hyperparameters:
                for param, value in sorted(hyperparameters.items()):
                    print(f"  {param} = {value}")
            else:
                print("  (using defaults)")
            print()  # Extra blank line for readability
            continue

        # Handle hyperparameter changes: \set param=value
        if user_input_stripped.startswith("\\set ") or user_input_stripped.startswith("/set "):
            stripped = user_input.strip().lstrip("\\/")
            # Use removeprefix to correctly remove "set " literal substring
            set_cmd = stripped.removeprefix("set ").strip()
            if "=" in set_cmd:
                param_name, param_value = set_cmd.split("=", 1)
                param_name = param_name.strip().lower()
                param_value = param_value.strip()

                # Try to parse as float first (handles scientific notation, decimals, negatives)
                # Then cast to int if needed for int parameters
                try:
                    parsed_value = float(param_value)  # Handles scientific notation, decimals, negatives
                except ValueError:
                    print(f"[ERROR] Invalid value '{param_value}' for parameter '{param_name}'. Must be a number.")
                    continue

                # Update hyperparameters dict
                float_params = [
                    "temperature",
                    "top_p",
                    "min_p",
                    "repetition_penalty",
                    "xtc_probability",
                    "xtc_threshold",
                ]
                int_params = [
                    "max_tokens",
                    "top_k",
                    "min_tokens_to_keep",
                    "repetition_context_size",
                ]

                if param_name in float_params:
                    hyperparameters[param_name] = parsed_value
                    print(f"[INFO] Set {param_name} = {parsed_value}")
                elif param_name in int_params:
                    hyperparameters[param_name] = int(parsed_value)
                    print(f"[INFO] Set {param_name} = {int(parsed_value)}")
                else:
                    valid_params = ", ".join(float_params + int_params)
                    print(
                        f"[ERROR] Unknown parameter '{param_name}'. "
                        f"Valid parameters: {valid_params}"
                    )
                    continue

                # Save updated hyperparameters immediately if chat file exists
                if chat_file_path and conversation:
                    try:
                        save_conversation(
                            chat_file_path,
                            conversation,
                            model_path,
                            prompt_config_path,
                            tools,
                            hyperparameters,
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed to save updated hyperparameters: {e}")

                continue  # Skip adding this as a user message
            else:
                # \set without = - show usage
                print("\n[ERROR] Invalid \\set command format.")
                print("[INFO] Usage: \\set <param>=<value>")
                print("[INFO] Example: \\set temperature=0.7")
                valid_params = (
                    "temperature, top_p, min_p, top_k, max_tokens, "
                    "min_tokens_to_keep, repetition_penalty, "
                    "repetition_context_size, xtc_probability, xtc_threshold"
                )
                print(f"[INFO] Valid parameters: {valid_params}")
                print()
                continue

        # Handle invalid out-of-band commands (starts with \ or / but not recognized)
        if user_input_stripped.startswith("\\") or user_input_stripped.startswith("/"):
            print("\n[ERROR] Unknown out-of-band command.")
            print(HELP_TEXT)
            continue

        # Add timestamp to user message (turn)
        user_turn = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        conversation.append(user_turn)

        # Main generation loop with tool call handling
        tool_iteration = 0
        while tool_iteration < max_tool_iterations:
            tokens: list[int] = []

            # system_message parameter is for CLI override (if we add --system flag later)
            # The generator's render_prompt() already handles system_model_identity from prompt_config
            # So we pass None here and let the generator handle the fallback
            system_message = None  # Could be overridden by CLI --system flag in the future

            # Debug: always write raw prompt to file, print to console only if --debug is set
            raw_prompt = generator.render_prompt(conversation, system_message)
            if args.debug:  # Only print to console if --debug flag is set
                print("\n[DEBUG] Raw prompt sent to LLM:")
                print("-" * 80)
                print(raw_prompt)
                print("-" * 80)
            # Always write to debug file (debug_path is always set now)
            with open(debug_path, "a", encoding="utf-8") as df:
                df.write("\n[DEBUG] Raw prompt sent to LLM:\n")
                df.write("-" * 80 + "\n")
                df.write(raw_prompt + "\n")
                df.write("-" * 80 + "\n")

            # Use hyperparameters dict (CLI args already merged with loaded values)
            # For Harmony models, we'll parse messages to extract final channel content
            # For Harmony models, use StreamableParser for incremental parsing
            # For non-Harmony models, stream tokens directly
            generation_start_time = time.perf_counter()
            # Initialize these early to avoid scope issues (used outside Harmony branch)
            parsed_messages: Any | None = None  # Will be set for Harmony models
            analysis_text_parts: list[str] = []
            assistant_text = ""
            # Accumulate streamed text for non-Harmony models (avoid decoding twice)
            streamed_text_parts: list[str] = []

            # For Harmony models, reset StreamableParser for this generation
            if generator.is_gpt_oss and generator.use_harmony and generator.encoding:
                # Create a fresh parser for this generation (reset state)
                generator.streamable_parser = StreamableParser(generator.encoding, Role.ASSISTANT, strict=False)

            # Collect all tokens for debugging
            all_generated_tokens = []
            for token_id in generator.generate(
                messages=conversation,
                temperature=hyperparameters.get("temperature"),
                max_tokens=hyperparameters.get("max_tokens"),
                top_p=hyperparameters.get("top_p"),
                min_p=hyperparameters.get("min_p"),
                top_k=hyperparameters.get("top_k"),
                repetition_penalty=hyperparameters.get("repetition_penalty"),
                repetition_context_size=hyperparameters.get("repetition_context_size"),
                system_message=system_message,
            ):
                token_int = int(token_id)
                tokens.append(token_int)
                all_generated_tokens.append(token_int)

                # For Harmony models, use StreamableParser to parse incrementally
                # StreamableParser handles all Harmony token decoding internally - we just display the text deltas
                if generator.is_gpt_oss and generator.use_harmony and generator.streamable_parser:
                    # For Harmony models, don't display during streaming
                    # We'll extract and display all channels (analysis + final) after parsing is complete
                    # This avoids duplicate output and ensures proper formatting
                    try:
                        # Process token through Harmony parser (handles Harmony special tokens correctly)
                        generator.streamable_parser.process(int(token_id))
                    except Exception as e:
                        # If parsing fails, log but continue - will try to parse messages after generation
                        if args.debug:
                            print(f"\n[DEBUG] Streaming parser error: {e}", file=sys.stderr)
                else:
                    # For non-Harmony models, use native tokenizer to decode tokens
                    text = generator.tokenizer.decode([int(token_id)])
                    text = clean_text(text)  # Clean Unicode and remove replacement chars
                    print(text, end="", flush=True)
                    streamed_text_parts.append(text)  # Accumulate for saving

            # After generation, keep model parameters active to prevent swapping
            # This ensures buffers stay wired and don't get swapped out
            # Use generator's keepalive() method (unified behavior)
            generator.keepalive()

            generation_end_time = time.perf_counter()
            generation_elapsed = generation_end_time - generation_start_time
            num_generated_tokens = len(tokens)
            tokens_per_second = num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0

            # Display generation stats
            print(f"\n[INFO] Generated {num_generated_tokens} tokens in {generation_elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

            # Write all generated tokens to debug file (once, not duplicated)
            if all_generated_tokens:
                with open(debug_path, "a", encoding="utf-8") as df:
                    df.write(f"\n[DEBUG] All {len(all_generated_tokens)} tokens generated (token IDs):\n")
                    df.write("-" * 80 + "\n")
                    df.write(str(all_generated_tokens) + "\n")
                    df.write("-" * 80 + "\n")
                    if generator.encoding:
                        decoded_all = generator.encoding.decode(all_generated_tokens)
                        df.write("\n[DEBUG] All tokens decoded (raw response from model):\n")
                        df.write("-" * 80 + "\n")
                        df.write(decoded_all + "\n")
                        df.write("-" * 80 + "\n")

            # For Harmony models, parse messages and extract final channel content
            if generator.is_gpt_oss and generator.use_harmony:
                # Reset for this generation
                final_text_parts = []
                analysis_text_parts = []  # Reset (was initialized above)
                assistant_text = ""  # Reset (was initialized above)

                # Note: assistant_name already computed above, don't recompute
                # For Harmony models, we already streamed text during generation via StreamableParser
                # Now we need to extract the properly parsed messages for saving and analysis channel display
                # Use StreamableParser's messages if available (most reliable - incremental parsing)
                if generator.streamable_parser:
                    # Check parser state before finalizing
                    parser_state = generator.streamable_parser.state
                    if args.debug:
                        print(f"\n[DEBUG] Parser state before process_eos(): {parser_state}")
                        print(f"[DEBUG] Parser current_role: {generator.streamable_parser.current_role}")
                        print(f"[DEBUG] Parser current_channel: {generator.streamable_parser.current_channel}")
                        print(f"[DEBUG] Parser tokens processed: {len(generator.streamable_parser.tokens)}")
                        if generator.streamable_parser.tokens:
                            # Decode first few tokens to see what the model started with
                            first_tokens = generator.streamable_parser.tokens[:20]
                            decoded_start = generator.encoding.decode(first_tokens) if generator.encoding else "N/A"
                            print(f"[DEBUG] First 20 tokens decoded: {decoded_start[:200]}")

                    # Finalize parser (process EOS if needed)
                    try:
                        generator.streamable_parser.process_eos()
                        parsed_messages = generator.streamable_parser.messages
                    except Exception as e:
                        # Fail fast with clear error message and debugging info
                        error_msg = str(e)
                        print(f"\n[ERROR] Harmony parsing failed: {error_msg}")
                        print(f"[ERROR] Parser state: {parser_state}")
                        print(f"[ERROR] Tokens processed: {len(generator.streamable_parser.tokens)}")
                        if generator.streamable_parser.tokens and generator.encoding:
                            first_tokens = generator.streamable_parser.tokens[:50]
                            decoded_start = generator.encoding.decode(first_tokens)
                            print(f"[ERROR] First 50 tokens decoded: {decoded_start[:500]}")
                            print(f"[ERROR] First 20 token IDs: {generator.streamable_parser.tokens[:20]}")
                            # Check what token ID corresponds to <|channel|> (should be 200005)
                            channel_token_id = 200005  # From Harmony format docs
                            if channel_token_id in generator.streamable_parser.tokens[:20]:
                                idx = generator.streamable_parser.tokens[:20].index(channel_token_id)
                                print(f"[ERROR] Found <|channel|> token (200005) at position {idx}")
                            else:
                                print("[ERROR] <|channel|> token (200005) NOT found in first 20 tokens")
                        print("\nThis error indicates the model output is malformed/incomplete:")
                        print("  - The parser was waiting for a message header to complete")
                        print("  - Model output does not conform to Harmony format structure")
                        raise RuntimeError(f"Failed to parse Harmony messages: {error_msg}") from e
                else:
                    # Fallback: parse from tokens using HarmonyEncoding
                    parsed_messages = generator.parse_messages_from_tokens(tokens)

                # Extract text from messages for saving and display
                # NOTE: During streaming, we only displayed final channel content (filtered by current_channel)
                # Here we extract all channels properly from parsed messages
                # For final channel, we want the LAST message (most recent), not all of them
                final_channel_messages = []
                if args.debug:
                    print(f"\n[DEBUG] Parsed {len(parsed_messages)} messages from parser")
                for msg in parsed_messages:
                    channel = getattr(msg, "channel", None)
                    # Extract text content from message
                    msg_text = ""
                    for content in msg.content:
                        if hasattr(content, "text"):
                            msg_text += content.text

                    if args.debug:
                        author = getattr(msg, "author", None)
                        print(f"[DEBUG] Message: channel={channel}, author={author}, text_length={len(msg_text)}")

                    if channel == "final" or channel is None:
                        # Final channel messages - collect all, we'll take the last one
                        final_channel_messages.append(clean_text(msg_text))
                    elif channel in ("analysis", "commentary"):
                        # Analysis/commentary channel - extract for [THINKING] display (was NOT streamed)
                        analysis_text_parts.append(clean_text(msg_text))

                # For final channel, only take the LAST message (most recent completion)
                # Earlier messages might be partial or from previous turns
                if final_channel_messages:
                    final_text_parts = [final_channel_messages[-1]]  # Only the last final channel message

                # Display analysis/thinking if present (this was NOT displayed during streaming)
                if analysis_text_parts:
                    # Join with newlines and preserve structure - only strip spaces/tabs (not newlines)
                    joined_analysis = "\n".join(analysis_text_parts)
                    thinking_text = truncate_text(joined_analysis.lstrip(" \t").rstrip(" \t"), thinking_limit)
                    if thinking_text:
                        display_thinking(thinking_text, render_markdown=not args.no_markdown)

                # Extract final channel text for display and saving
                if final_text_parts:
                    # Join parts and preserve newlines - only strip leading/trailing whitespace (not newlines)
                    joined_text = "".join(final_text_parts)
                    # Strip only leading/trailing spaces/tabs, preserve newlines
                    assistant_text = truncate_text(joined_text.lstrip(" \t").rstrip(" \t"), response_limit)
                    # Display final response (it may not have been displayed during streaming if model only generated analysis)
                    if assistant_text:
                        display_assistant(assistant_text, assistant_name, render_markdown=not args.no_markdown)
                elif analysis_text_parts:
                    # Only analysis channel content was found (no final channel)
                    assistant_text = ""  # No final response
                    if args.debug:
                        print("\n[DEBUG] Only analysis channel found in parsed_messages")
                        print(f"[DEBUG] Total parsed messages: {len(parsed_messages)}")
                        for i, msg in enumerate(parsed_messages):
                            channel = getattr(msg, "channel", None)
                            print(f"[DEBUG] Message {i}: channel={channel}, role={getattr(msg, 'author', None)}")
                elif streamed_text_parts:
                    # No parsed messages but we have streamed content - use it
                    streamed_content = "".join(streamed_text_parts).strip()
                    assistant_text = truncate_text(streamed_content, response_limit)
                    print()  # Newline after streaming
                else:
                    # No parsed messages and no streamed content - fail fast
                    print("\n[ERROR] Failed to parse Harmony messages: no parsed messages and no streamed content")
                    print("This indicates either:")
                    print("  - openai_harmony package is incorrectly installed")
                    print("  - Model output is malformed")
                    print("  - Parsing logic has a bug")
                    if args.debug:
                        raw_text = generator.encoding.decode(tokens) if generator.encoding else "[encoding not available]"
                        print(f"[DEBUG] Raw decoded text: {raw_text[:500]}...")
                    raise RuntimeError("Failed to parse Harmony messages: no parsed messages and no streamed content")
            else:
                # Non-Harmony model: already printed during streaming
                # Use accumulated streamed text (avoid decoding twice)
                print()  # Newline after streaming
                assistant_text = "".join(streamed_text_parts)

            # For GPT-OSS models with tools, check for tool calls
            # (We already parsed messages above for Harmony models, so reuse if available)
            if generator.is_gpt_oss and tools and generator.use_harmony:
                try:
                    # Reuse parsed_messages if we already parsed them above
                    if parsed_messages is None:
                        parsed_messages = generator.parse_messages_from_tokens(tokens)
                    tool_calls = parse_tool_calls_from_messages(
                        parsed_messages, tools
                    )

                    if tool_calls:
                        print(f"\n[TOOL] Detected {len(tool_calls)} tool call(s)")
                        for tool_call in tool_calls:
                            print(
                                f"[TOOL] Executing: {tool_call.tool_name} with args: {tool_call.arguments}"
                            )
                            result = execute_tool_call(tool_call)
                            print(f"[TOOL] Result: {result}")

                            # Add tool result to conversation in Harmony format
                            # Format: <|start|>{tool_name} to=assistant<|channel|>commentary<|message|>{result}<|end|>
                            # Use role="tool" with name field for proper Harmony message construction
                            # Harmony expects tool results in "commentary" channel
                            # Record hyperparameters used for this generation cycle
                            tool_result_msg = {
                                "role": "tool",
                                "name": tool_call.tool_name,  # Tool name goes in Author.name
                                "content": result,
                                "recipient": "assistant",  # Tool results are sent to assistant
                                "channel": "commentary",  # Harmony expects tool results in commentary channel
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "hyperparameters": hyperparameters.copy()
                                if hyperparameters
                                else {},
                            }
                            conversation.append(tool_result_msg)

                        # Continue generation with tool results
                        tool_iteration += 1
                        continue
                except Exception as e:
                    print(f"\n[WARNING] Error parsing tool calls: {e}")

            # No tool calls or non-GPT-OSS model: assistant_text already set above
            # (For Harmony models, it's the final channel content; for others, it's decoded tokens)

            # For Harmony models, raw response already written to debug file above
            # Only write raw response for non-Harmony models to avoid duplication
            if not (generator.is_gpt_oss and generator.use_harmony):
                # Non-Harmony models: decode and write raw response
                raw_response = generator.tokenizer.decode(tokens)
                cleaned_response = clean_text(raw_response)
                if args.debug:  # Only print to console if --debug flag is set
                    print("\n[DEBUG] Raw response from LLM:")
                    print("-" * 80)
                    print(cleaned_response)
                    print("-" * 80)
                # Write to debug file
                with open(debug_path, "a", encoding="utf-8") as df:
                    df.write("\n[DEBUG] Raw response from LLM:\n")
                    df.write("-" * 80 + "\n")
                    df.write(raw_response + "\n")
                    df.write("-" * 80 + "\n")

            # Record hyperparameters used for this generation
            # Check if tool calls exist (for Harmony models) and handle empty assistant_text
            has_tool_calls = (
                generator.is_gpt_oss
                and tools
                and generator.use_harmony
                and parsed_messages is not None
            )

            tool_calls_detected = False
            if has_tool_calls:
                try:
                    tool_calls_check = parse_tool_calls_from_messages(parsed_messages, tools)
                    tool_calls_detected = bool(tool_calls_check)
                except Exception:
                    pass

            # Only save assistant turn if we have actual content (not just analysis or tool calls)
            # If tool calls exist and assistant_text is empty/whitespace, skip saving junk
            if tool_calls_detected and not assistant_text.strip():
                # Tool call issued without meaningful final content - skip saving
                # The tool results will be added in the next iteration (already handled above)
                pass
            elif assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
                # For non-Harmony models, always save; for Harmony models, only if we have final response
                assistant_turn = {
                    "role": "assistant",
                    "content": assistant_text if assistant_text else "[No final response - see thinking above]",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                }
                # For Harmony models, also record analysis channel if present
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    # Join with newlines to preserve structure when saving
                    assistant_turn["analysis"] = "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t")
                conversation.append(assistant_turn)
            else:
                # Harmony model with only analysis channel - save analysis separately
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    # Join with newlines to preserve structure when saving
                    assistant_turn = {
                        "role": "assistant",
                        "content": "[Analysis only - no final response]",
                        "analysis": "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t"),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                    }
                    conversation.append(assistant_turn)

            # Save conversation after each exchange (turn)
            if chat_file_path:
                try:
                    save_conversation(
                        chat_file_path,
                        conversation,
                        model_path,
                        prompt_config_path,
                        tools,
                        hyperparameters,
                    )
                except Exception as e:
                    print(f"\n[WARNING] Failed to save chat: {e}")

            break

    # Final save on exit
    if chat_file_path and conversation:
        try:
            save_conversation(
                chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                tools,
                hyperparameters,
            )
            print(f"\n[INFO] Chat saved to: {chat_file_path}")
        except Exception as e:
            print(f"\n[WARNING] Failed to save chat on exit: {e}")


if __name__ == "__main__":
    main()
