from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptConfig(BaseModel):
    """
    Configuration for Harmony prompt fragments and default sampling parameters.
    """

    @model_validator(mode="after")
    def _normalize_token_limits(self) -> PromptConfig:
        max_tokens = self.max_tokens
        if max_tokens is None or max_tokens <= 0:
            return self

        def _clean(value: int | None) -> int | None:
            if value is None:
                return None
            if value <= 0:
                return None
            return value

        max_user = _clean(self.max_user_tokens)
        max_assistant = _clean(self.max_assistant_tokens)

        if max_user is None and max_assistant is None:
            half = max_tokens // 2
            max_user = half
            max_assistant = max_tokens - half
        elif max_user is None and max_assistant is not None:
            max_user = max(0, max_tokens - max_assistant)
        elif max_assistant is None and max_user is not None:
            max_assistant = max(0, max_tokens - max_user)

        if max_user is None or max_assistant is None:
            return self

        total = max_user + max_assistant
        if total > max_tokens and total > 0:
            max_user = int((max_user / total) * max_tokens)
            max_assistant = max_tokens - max_user

        self.max_user_tokens = max_user
        self.max_assistant_tokens = max_assistant
        return self

    model_config = ConfigDict(extra="ignore")

    # Harmony system/developer fields (GPT-OSS only)
    system_model_identity: str | None = None
    reasoning_effort: str | None = None  # "Low" | "Medium" | "High"
    conversation_start_date: str | None = None  # e.g. "<|DATE|>" or explicit
    knowledge_cutoff: str | None = None  # e.g. "2024-06"
    developer_instructions: str | None = None
    assistant_greeting: str | None = None
    use_harmony: bool | None = Field(
        default=None,
        description="Override Harmony formatting (true/false). Defaults to auto for GPT-OSS.",
    )

    # Example dialogues (few-shot examples) - list of conversations, each with turns
    # Format: [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...]
    example_dialogues: list[list[dict[str, str]]] | None = None

    # User-defined placeholders for template expansion
    placeholders: dict[str, str] = Field(default_factory=dict)

    # Sampling defaults (any model)

    max_user_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum user input tokens before truncation",
    )
    max_assistant_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Maximum assistant response tokens (overrides max_tokens when set)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate (default: 1024 for Harmony, 512 otherwise)",
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum prompt context tokens (truncate history to fit when set)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0, higher = more creative)",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling (0.0-1.0, keep tokens with cumulative probability <= top_p)",
    )
    min_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold (0.0-1.0, filter low-probability tokens)",
    )
    min_tokens_to_keep: int | None = Field(
        default=None,
        ge=1,
        description="Minimum tokens to keep after filtering (default: 1)",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Top-k sampling (keep only top k tokens, 0 = disabled)",
    )
    xtc_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="XTC sampling probability (0.0-1.0, experimental)",
    )
    xtc_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description="XTC sampling threshold (experimental)",
    )
    xtc_special_tokens: list[int] | None = (
        None  # Token IDs to exclude from XTC filtering (e.g., EOS, newline). Auto-detected if None.
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=0.0,
        description="Repetition penalty (>1.0 penalizes repetition, 1.0 = no penalty)",
    )
    repetition_context_size: int | None = Field(
        default=None,
        ge=0,
        description="Number of previous tokens to consider for repetition penalty",
    )

    # Model loading optimizations
    mlock: bool | None = (
        None  # Lock model weights in memory using mlock (macOS Metal only, default: False)
    )
    no_fs_cache: bool | None = (
        None  # Disable filesystem cache when reading weights (macOS only, default: False)
    )
    lazy: bool | None = None  # Lazy-load model weights (default: False)
    seed: int | None = None  # Random seed for generation (-1 = random each run)
    reseed_each_turn: bool | None = None  # Reseed before each generation when seed >= 0

    # Display truncation limits
    truncate_thinking: int | None = Field(
        default=None,
        ge=0,
        description="Truncate thinking/analysis text to N chars (default: 1000)",
    )
    truncate_response: int | None = Field(
        default=None,
        ge=0,
        description="Truncate final response text to N chars (default: 1000)",
    )

    # Directory configuration
    logs_dir: str | None = None  # Directory for debug logs (default: "logs")
    chats_dir: str | None = None  # Directory for chat history files (default: "logs")


_ANGLE_PREFIX = "<|"


class MoshiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    stt_model_path: str | None = Field(default=None)
    stt_config_path: str | None = Field(default=None)
    stt_max_seconds: float = Field(default=8.0)
    stt_vad: bool = Field(default=False)
    stt_vad_threshold: float = Field(default=0.5)
    stt_vad_hits: int = Field(default=2)
    stt_silence: bool = Field(default=True)
    stt_silence_threshold: float = Field(default=0.01)
    stt_silence_ms: int = Field(default=700)
    stt_min_speech_ms: int = Field(default=200)
    stt_block_ms: int = Field(default=80)
    stt_warmup_blocks: int = Field(default=2)
    tts_model_path: str | None = Field(default=None)
    tts_config_path: str | None = Field(default=None)
    tts_voice_path: str | None = Field(default=None)
    quantize: int | None = Field(default=None)
    tts_chunk_chars: int = Field(default=180)
    tts_chunk_sentences: bool = Field(default=True)
    tts_chunk_min_chars: int = Field(default=60)
    tts_stream: bool = Field(default=False)
    use_stt: bool = Field(default=True)
    use_tts: bool = Field(default=True)
    smoke_test: bool = Field(default=False)
    barge_in: bool = Field(default=False)
    barge_in_window_seconds: float = Field(default=2.0)

    def validate_paths(self) -> list[str]:
        missing: list[str] = []
        if self.use_stt:
            if not self.stt_model_path:
                missing.append("stt_model_path")
            elif not Path(self.stt_model_path).exists():
                missing.append(self.stt_model_path)
        if self.use_tts:
            if not self.tts_model_path:
                missing.append("tts_model_path")
            elif not Path(self.tts_model_path).exists():
                missing.append(self.tts_model_path)
            if self.tts_voice_path and not Path(self.tts_voice_path).exists():
                missing.append(self.tts_voice_path)
        return missing


def load_moshi_config(path: str | None) -> MoshiConfig | None:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Moshi config not found: {config_path}")
    with open(config_path, encoding="utf-8") as fobj:
        data = json.load(fobj)
    return MoshiConfig(**data)


_ANGLE_SUFFIX = "|>"


def _build_builtin_placeholders(now: datetime, now_utc: datetime) -> dict[str, str]:
    return {
        "DATE": now.strftime("%Y-%m-%d"),
        "DATETIME": now.isoformat(timespec="seconds"),
        "TIME": now.strftime("%H:%M:%S"),
        "TIMEZ": now.strftime("%H:%M:%S"),
        "TIMEA": now.strftime("%I:%M:%S %p"),
        "TIMEU": now_utc.strftime("%H:%M:%S UTC"),
    }


def _replace_angle_tokens(value: str, replacements: dict[str, str]) -> str:
    """Replace <|TOKEN|> placeholders using the replacements map (case-insensitive)."""
    parts: list[str] = []
    idx = 0
    length = len(value)
    while idx < length:
        start = value.find(_ANGLE_PREFIX, idx)
        if start == -1:
            parts.append(value[idx:])
            break
        end = value.find(_ANGLE_SUFFIX, start + 2)
        if end == -1:
            parts.append(value[idx:])
            break
        parts.append(value[idx:start])
        token = value[start + 2 : end]
        token_upper = token.upper()
        replacement = replacements.get(token_upper)
        if replacement is None:
            parts.append(value[start : end + 2])
        else:
            parts.append(replacement)
        idx = end + 2
    return "".join(parts)


def _render_placeholders(value: str, user_placeholders: dict[str, str]) -> str:
    """Expand dynamic placeholders and user-defined symbols.

    Supports both built-in and user-defined placeholders in two formats:
    - Built-in: <|DATE|>, <|TIME|>, <|TIMEZ|>, <|TIMEA|>, <|TIMEU|>, <|DATETIME|>
    - User-defined: <|KEY|> or {key} (both formats supported)
    """

    if _ANGLE_PREFIX in value:
        now = datetime.now()
        now_utc = datetime.now(UTC)
        replacements = _build_builtin_placeholders(now, now_utc)
        for key, replacement in user_placeholders.items():
            replacements[key.upper()] = replacement
        value = _replace_angle_tokens(value, replacements)

    # Then apply user-defined placeholders in {key} format (case-sensitive for curly braces)
    for key, replacement in user_placeholders.items():
        value = value.replace(f"{{{key}}}", replacement)
    return value


def _maybe_render(value: str | None, placeholders: dict[str, str]) -> str | None:
    if value is None:
        return None
    return _render_placeholders(value, placeholders)


def apply_placeholders(value: str | None, placeholders: dict[str, str]) -> str | None:
    """Public helper to apply dynamic tokens and user placeholders."""
    return _maybe_render(value, placeholders)


def load_prompt_config(path: str | Path) -> PromptConfig | None:
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
      "xtc_special_tokens": null,
      "repetition_penalty": 1.0,
      "repetition_context_size": 20,
      "max_tokens": 1024,
      "max_context_tokens": 4096,
      "mlock": false,
      "truncate_thinking": 1000,
      "truncate_response": 1000,
      "logs_dir": "logs",
      "chats_dir": "logs"
    }

    Built-in placeholders:
    - <|DATE|>: Current date (YYYY-MM-DD, local time)
    - <|DATETIME|>: Current datetime (ISO format, local time)
    - <|TIME|>: Current time (HH:MM:SS 24-hour, local time)
    - <|TIMEZ|>: Current time (HH:MM:SS 24-hour, local time)
    - <|TIMEA|>: Current time (HH:MM:SS AM/PM 12-hour, local time)
    - <|TIMEU|>: Current time (HH:MM:SS UTC 24-hour, UTC timezone)

    User-defined placeholders: {key} replaced with placeholders[key]

    Returns:
        PromptConfig if file exists and is valid, None if file doesn't exist.
        Raises json.JSONDecodeError or ValueError if file exists but is invalid.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        return None

    try:
        data: dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

    user_placeholders: dict[str, str] = {
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
                        "content": _render_placeholders(turn["content"], user_placeholders),
                    }
                else:
                    processed_turn = turn
                processed_turns.append(processed_turn)
            processed_examples.append(processed_turns)
        example_dialogues = processed_examples

    return PromptConfig(
        system_model_identity=_maybe_render(data.get("system_model_identity"), user_placeholders),
        reasoning_effort=data.get("reasoning_effort"),
        conversation_start_date=_maybe_render(
            data.get("conversation_start_date"), user_placeholders
        ),
        knowledge_cutoff=_maybe_render(data.get("knowledge_cutoff"), user_placeholders),
        developer_instructions=_maybe_render(data.get("developer_instructions"), user_placeholders),
        assistant_greeting=_maybe_render(data.get("assistant_greeting"), user_placeholders),
        example_dialogues=example_dialogues,
        placeholders=user_placeholders,
        max_tokens=data.get("max_tokens"),
        max_context_tokens=data.get("max_context_tokens"),
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        min_p=data.get("min_p"),
        min_tokens_to_keep=data.get("min_tokens_to_keep"),
        top_k=data.get("top_k"),
        xtc_probability=data.get("xtc_probability"),
        xtc_threshold=data.get("xtc_threshold"),
        xtc_special_tokens=data.get("xtc_special_tokens")
        if data.get("xtc_special_tokens") is not None
        else None,
        repetition_penalty=data.get("repetition_penalty"),
        repetition_context_size=data.get("repetition_context_size"),
        mlock=data.get("mlock"),
        truncate_thinking=data.get("truncate_thinking"),
        truncate_response=data.get("truncate_response"),
        logs_dir=data.get("logs_dir"),
        chats_dir=data.get("chats_dir"),
    )


# Dialogue Import/Conversion ------------------------------------------------


def parse_dialogue_text(text: str) -> list[dict[str, str]]:
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
    messages: list[dict[str, str]] = []
    current_role: str | None = None
    current_content: list[str] = []

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


def parse_dialogue_file(path: str | Path) -> list[dict[str, str]]:
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


def load_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load profile map from JSON. Schema:
    {
      "gpt-oss-20b": {
        "model": "/path/to/model",
        "prompt_config": "configs/prompt-config.example.json",
        "max_context_tokens": 131072
      },
      "gpt-oss-120b": {
        "model": "/path/to/model120",
        "prompt_config": "configs/prompt-config.example.json",
        "max_context_tokens": 131072
      }
    }
    """
    raw = Path(path)
    return json.loads(raw.read_text(encoding="utf-8"))
