from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PromptConfig:
    """
    Configuration for Harmony prompt fragments and default sampling parameters.
    """

    # Harmony system/developer fields (GPT-OSS only)
    system_model_identity: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "Low" | "Medium" | "High"
    conversation_start_date: Optional[str] = None  # e.g. "<|DATE|>" or explicit
    knowledge_cutoff: Optional[str] = None  # e.g. "2024-06"
    developer_instructions: Optional[str] = None
    assistant_greeting: Optional[str] = None

    # Example dialogues (few-shot examples) - list of conversation turns
    # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    example_dialogues: Optional[List[Dict[str, str]]] = None

    # User-defined placeholders for template expansion
    placeholders: Dict[str, str] = field(default_factory=dict)

    # Sampling defaults (any model)
    max_tokens: Optional[int] = None  # Maximum tokens to generate (default: 1024 for Harmony, 512 otherwise)
    temperature: Optional[float] = None  # Sampling temperature (0.0-2.0, higher = more creative)
    top_p: Optional[float] = None  # Nucleus sampling (0.0-1.0, keep tokens with cumulative probability <= top_p)
    min_p: Optional[float] = None  # Minimum probability threshold (0.0-1.0, filter low-probability tokens)
    min_tokens_to_keep: Optional[int] = None  # Minimum tokens to keep after filtering (default: 1)
    top_k: Optional[int] = None  # Top-k sampling (keep only top k tokens, 0 = disabled)
    xtc_probability: Optional[float] = None  # XTC sampling probability (0.0-1.0, experimental)
    xtc_threshold: Optional[float] = None  # XTC sampling threshold (experimental)
    repetition_penalty: Optional[float] = None  # Repetition penalty (>1.0 penalizes repetition, 1.0 = no penalty)
    repetition_context_size: Optional[int] = None  # Number of previous tokens to consider for repetition penalty

    # Model loading optimizations
    prewarm_cache: Optional[bool] = None  # Pre-warm filesystem cache before loading (default: True)
    mlock: Optional[bool] = None  # Lock model weights in memory using mlock (macOS Metal only, default: False)


_PLACEHOLDER_RE = re.compile(r"<\|([A-Z_]+)\|>")


def _render_placeholders(value: str, user_placeholders: Dict[str, str]) -> str:
    """Expand dynamic placeholders and user-defined symbols.

    Supports both built-in and user-defined placeholders in two formats:
    - Built-in: <|DATE|>, <|TIME|>, <|TIMEZ|>, <|TIMEA|>, <|TIMEU|>, <|DATETIME|>
    - User-defined: <|KEY|> or {key} (both formats supported)
    """

    def repl(match: re.Match) -> str:
        token = match.group(1)
        now = datetime.now()
        now_utc = datetime.now(timezone.utc)

        # Built-in placeholders (checked first)
        if token == "DATE":
            return now.strftime("%Y-%m-%d")
        if token == "DATETIME":
            return now.isoformat(timespec="seconds")
        if token == "TIME":
            # Default: 24-hour local time (HH:MM:SS)
            return now.strftime("%H:%M:%S")
        if token == "TIMEZ":
            # 24-hour local time (HH:MM:SS)
            return now.strftime("%H:%M:%S")
        if token == "TIMEA":
            # 12-hour local time with AM/PM (HH:MM:SS AM/PM)
            return now.strftime("%I:%M:%S %p")
        if token == "TIMEU":
            # UTC time in 24-hour format (HH:MM:SS UTC)
            return now_utc.strftime("%H:%M:%S UTC")

        # User-defined placeholders in <|KEY|> format
        # Normalize to uppercase for case-insensitive matching (consistent with built-ins)
        token_upper = token.upper()
        # Check normalized token against all keys (normalized)
        for key, replacement in user_placeholders.items():
            if key.upper() == token_upper:
                return replacement

        # Not found: return original (could be intentional or typo)
        return match.group(0)

    # First expand <|...|> format (built-in + user-defined)
    value = _PLACEHOLDER_RE.sub(repl, value)
    # Then apply user-defined placeholders in {key} format (case-sensitive for curly braces)
    for key, replacement in user_placeholders.items():
        value = value.replace(f"{{{key}}}", replacement)
    return value


def _maybe_render(value: Optional[str], placeholders: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    return _render_placeholders(value, placeholders)


def apply_placeholders(value: Optional[str], placeholders: Dict[str, str]) -> Optional[str]:
    """Public helper to apply dynamic tokens and user placeholders."""
    return _maybe_render(value, placeholders)


def load_prompt_config(path: str | Path) -> PromptConfig:
    """
    Load a PromptConfig from a JSON file.

    Supported fields:
    {
      "system_model_identity": "You are {assistant} on <|DATE|> at <|TIMEZ|>.",
      "reasoning_effort": "Medium",
      "conversation_start_date": "<|DATE|>",
      "knowledge_cutoff": "2025-01",
      "developer_instructions": "Always answer concisely.",
      "assistant_greeting": "Hello {user}, I'm {assistant}. How can I help you today?",
      "example_dialogues": [
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
      ],
      "placeholders": { "assistant": "Dave", "user": "Morgan" },
      "temperature": 0.8,
      "top_p": 0.9,
      "min_p": 0.0,
      "top_k": 40,
      "min_tokens_to_keep": 1,
      "xtc_probability": 0.0,
      "xtc_threshold": 0.0,
      "repetition_penalty": 1.0,
      "repetition_context_size": 20
    }

    Built-in placeholders:
    - <|DATE|>: Current date (YYYY-MM-DD, local time)
    - <|DATETIME|>: Current datetime (ISO format, local time)
    - <|TIME|>: Current time (HH:MM:SS 24-hour, local time)
    - <|TIMEZ|>: Current time (HH:MM:SS 24-hour, local time)
    - <|TIMEA|>: Current time (HH:MM:SS AM/PM 12-hour, local time)
    - <|TIMEU|>: Current time (HH:MM:SS UTC 24-hour, UTC timezone)

    User-defined placeholders: {key} replaced with placeholders[key]
    """
    raw_path = Path(path)
    data: Dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))

    user_placeholders: Dict[str, str] = {
        str(k): str(v) for k, v in (data.get("placeholders", {}) or {}).items()
    }

    # Load example dialogues (list of conversation turns)
    example_dialogues = data.get("example_dialogues")
    if example_dialogues:
        # Apply placeholders to example dialogues
        processed_examples = []
        for example_turns in example_dialogues:
            processed_turns = []
            for turn in example_turns:
                if "content" in turn:
                    processed_turn = {
                        **turn,
                        "content": _render_placeholders(
                            turn["content"], user_placeholders
                        ),
                    }
                else:
                    processed_turn = turn
                processed_turns.append(processed_turn)
            processed_examples.append(processed_turns)
        example_dialogues = processed_examples

    return PromptConfig(
        system_model_identity=_maybe_render(
            data.get("system_model_identity"), user_placeholders
        ),
        reasoning_effort=data.get("reasoning_effort"),
        conversation_start_date=_maybe_render(
            data.get("conversation_start_date"), user_placeholders
        ),
        knowledge_cutoff=_maybe_render(
            data.get("knowledge_cutoff"), user_placeholders
        ),
        developer_instructions=_maybe_render(
            data.get("developer_instructions"), user_placeholders
        ),
        assistant_greeting=_maybe_render(
            data.get("assistant_greeting"), user_placeholders
        ),
        example_dialogues=example_dialogues,
        placeholders=user_placeholders,
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        min_p=data.get("min_p"),
        min_tokens_to_keep=data.get("min_tokens_to_keep"),
        top_k=data.get("top_k"),
        xtc_probability=data.get("xtc_probability"),
        xtc_threshold=data.get("xtc_threshold"),
        repetition_penalty=data.get("repetition_penalty"),
        repetition_context_size=data.get("repetition_context_size"),
    )


# Dialogue Import/Conversion ------------------------------------------------


def parse_dialogue_text(text: str) -> List[Dict[str, str]]:
    """
    Parse dialogue text in the format:
    ```
    assistant: Hello, how may I help you today?
    user: I'd like to know something about fruit.
    assistant: What would you like to know about fruit?
    user: What is the difference between a fruit and a vegetable?
    ```

    Returns a list of messages in our format:
    [
        {"role": "assistant", "content": "Hello, how may I help you today?"},
        {"role": "user", "content": "I'd like to know something about fruit."},
        ...
    ]

    Supports:
    - Lines starting with "user:" or "assistant:"
    - Case-insensitive role matching
    - Multiline content (content continues until next role line)
    - Blank lines (ignored)
    - Whitespace stripping
    """
    messages: List[Dict[str, str]] = []
    current_role: Optional[str] = None
    current_content: List[str] = []

    # Pattern to match role:content lines
    role_pattern = re.compile(r"^\s*(user|assistant|tool)\s*:\s*(.*)$", re.IGNORECASE)

    for line in text.splitlines():
        line = line.rstrip()  # Remove trailing whitespace

        # Skip empty lines
        if not line.strip():
            if current_role and current_content:
                # Empty line continues current message
                current_content.append("")
            continue

        # Check if this line starts a new role
        match = role_pattern.match(line)
        if match:
            # Save previous message if exists
            if current_role and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    messages.append({"role": current_role.lower(), "content": content})

            # Start new message
            current_role = match.group(1).lower()
            content_start = match.group(2).strip()
            if content_start:
                current_content = [content_start]
            else:
                current_content = []
        else:
            # Continuation of current message
            if current_role:
                current_content.append(line)
            else:
                # Line doesn't start with role: and no current role
                # Skip it or treat as continuation (for backward compatibility)
                continue

    # Save last message
    if current_role and current_content:
        content = "\n".join(current_content).strip()
        if content:
            messages.append({"role": current_role.lower(), "content": content})

    return messages


def parse_dialogue_file(path: str | Path) -> List[Dict[str, str]]:
    """
    Parse a dialogue file and return messages in our format.

    Args:
        path: Path to text file with dialogue format

    Returns:
        List of messages: [{"role": "user", "content": "..."}, ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dialogue file not found: {path}")

    text = path.read_text(encoding="utf-8")
    return parse_dialogue_text(text)


# Profiles --------------------------------------------------------------------


def load_profiles(path: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Load profile map from JSON. Schema:
    {
      "gpt-oss-20b": {
        "model": "/path/to/model",
        "prompt_config": "configs/prompt-config.example.json"
      },
      "gpt-oss-120b": {
        "model": "/path/to/model120",
        "prompt_config": "configs/prompt-config.example.json"
      }
    }
    """
    raw = Path(path)
    return json.loads(raw.read_text(encoding="utf-8"))
