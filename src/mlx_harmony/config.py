from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DETERMINISTIC_TIME_ISO = "2000-01-01T00:00:00Z"
DEFAULT_DETERMINISTIC_SEED = 0
DEFAULT_DETERMINISTIC_RESEED_EACH_TURN = False
DEFAULT_CONFIG_DIR = "configs"


class PromptConfig(BaseModel):
    """
    Configuration for Harmony prompt fragments and default sampling parameters.
    """

    model_config = ConfigDict(extra="ignore")

    # Harmony system/developer fields (GPT-OSS only)
    system_model_identity: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "Low" | "Medium" | "High"
    conversation_start_date: Optional[str] = None  # e.g. "<|DATE|>" or explicit
    knowledge_cutoff: Optional[str] = None  # e.g. "2024-06"
    developer_instructions: Optional[str] = None
    assistant_greeting: Optional[str] = None

    # Example dialogues (few-shot examples) - list of conversations, each with turns
    # Format: [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], ...]
    example_dialogues: Optional[List[List[Dict[str, str]]]] = None

    # User-defined placeholders for template expansion
    placeholders: Dict[str, str] = Field(default_factory=dict)
    deterministic_time_enabled: Optional[bool] = None
    deterministic_time_iso: Optional[str] = None

    # Sampling defaults (any model)
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate (default: 1024 for Harmony, 512 otherwise)",
    )
    max_context_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum prompt context tokens (truncate history to fit when set)",
    )
    max_context_tokens_margin: Optional[int] = Field(
        default=None,
        ge=0,
        description="Safety margin to subtract from max_context_tokens during truncation",
    )
    performance_mode: Optional[bool] = None
    perf_max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Performance mode max tokens override (applies when performance_mode is true)",
    )
    perf_max_context_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Performance mode max context override (applies when performance_mode is true)",
    )
    perf_max_kv_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Performance mode KV window override (applies when performance_mode is true)",
    )
    perf_prompt_token_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Performance mode prompt token budget for early truncation (applies when performance_mode is true)",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0, higher = more creative)",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling (0.0-1.0, keep tokens with cumulative probability <= top_p)",
    )
    min_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold (0.0-1.0, filter low-probability tokens)",
    )
    min_tokens_to_keep: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum tokens to keep after filtering (default: 1)",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Top-k sampling (keep only top k tokens, 0 = disabled)",
    )
    xtc_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="XTC sampling probability (0.0-1.0, experimental)",
    )
    xtc_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.5,
        description="XTC sampling threshold (experimental)",
    )
    xtc_special_tokens: Optional[List[int]] = None  # Token IDs to exclude from XTC filtering (e.g., EOS, newline). Auto-detected if None.
    repetition_penalty: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Repetition penalty (>1.0 penalizes repetition, 1.0 = no penalty)",
    )
    repetition_context_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of previous tokens to consider for repetition penalty",
    )
    loop_detection: Optional[str] = Field(
        default=None,
        description="Loop detection mode: off, cheap, or full (default: cheap)",
    )
    max_kv_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum KV cache size (enables rotating KV cache when set)",
    )

    # Model loading optimizations
    mlock: Optional[bool] = None  # Lock model weights in memory using mlock (macOS Metal only, default: False)
    lazy: Optional[bool] = None  # Lazy-load model weights (default: False)
    seed: Optional[int] = None  # Random seed for generation (-1 = random each run)
    reseed_each_turn: Optional[bool] = None  # Reseed before each generation when seed >= 0
    clear_cache: Optional[bool] = None  # Clear MLX cache during prefill (default: True)
    clear_cache_interval: Optional[int] = None  # Interval for cache clearing (prefill chunks)
    clear_cache_generation: Optional[bool] = None  # Clear MLX cache during generation loop
    log_memory_stats: Optional[bool] = None  # Log MLX memory stats during generation
    log_timing_stats: Optional[bool] = None  # Log generation timing stats during generation
    end_token_strings: Optional[List[str]] = Field(
        default=None,
        description="Optional list of token strings that should stop generation when emitted",
    )

    # Display truncation limits
    truncate_thinking: Optional[int] = Field(
        default=None,
        ge=0,
        description="Truncate thinking/analysis text to N chars (default: 1000)",
    )
    truncate_response: Optional[int] = Field(
        default=None,
        ge=0,
        description="Truncate final response text to N chars (default: 1000)",
    )

    # Directory configuration
    logs_dir: Optional[str] = None  # Directory for debug logs (default: "logs")
    chats_dir: Optional[str] = None  # Directory for chat history files (default: "logs")
    models_dir: Optional[str] = None  # Default directory for model storage (default: "models")


_ANGLE_PREFIX = "<|"
_ANGLE_SUFFIX = "|>"


def _build_builtin_placeholders(
    now: datetime, now_utc: datetime
) -> Dict[str, str]:
    return {
        "DATE": now.strftime("%Y-%m-%d"),
        "DATETIME": now.isoformat(timespec="seconds"),
        "TIME": now.strftime("%H:%M:%S"),
        "TIMEZ": now.strftime("%H:%M:%S"),
        "TIMEA": now.strftime("%I:%M:%S %p"),
        "TIMEU": now_utc.strftime("%H:%M:%S UTC"),
    }


def _replace_angle_tokens(value: str, replacements: Dict[str, str]) -> str:
    """Replace <|TOKEN|> placeholders using the replacements map (case-insensitive)."""
    parts: List[str] = []
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


def _render_placeholders(value: str, user_placeholders: Dict[str, str]) -> str:
    """Expand dynamic placeholders and user-defined symbols.

    Supports both built-in and user-defined placeholders in two formats:
    - Built-in: <|DATE|>, <|TIME|>, <|TIMEZ|>, <|TIMEA|>, <|TIMEU|>, <|DATETIME|>
    - User-defined: <|KEY|> or {key} (both formats supported)
    """

    if _ANGLE_PREFIX in value:
        now = datetime.now()
        now_utc = datetime.now(timezone.utc)
        replacements = _build_builtin_placeholders(now, now_utc)
        for key, replacement in user_placeholders.items():
            replacements[key.upper()] = replacement
        value = _replace_angle_tokens(value, replacements)

    # Then apply user-defined placeholders in {key} format (case-sensitive for curly braces)
    for key, replacement in user_placeholders.items():
        value = value.replace(f"{{{key}}}", replacement)
    return value


def _maybe_render(value: Optional[str], placeholders: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    return _render_placeholders(value, placeholders)


def resolve_config_path(path: str | Path | None, config_dir: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    base_dir = Path(config_dir or os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    fallback = base_dir / candidate
    if fallback.exists():
        return fallback
    return candidate


def apply_placeholders(value: Optional[str], placeholders: Dict[str, str]) -> Optional[str]:
    """Public helper to apply dynamic tokens and user placeholders."""
    return _maybe_render(value, placeholders)


def _parse_deterministic_time(value: str) -> tuple[datetime, datetime]:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Invalid deterministic_time_iso value '{value}'. Expected ISO UTC timestamp."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_local, dt_utc


def load_prompt_config(path: str | Path) -> Optional[PromptConfig]:
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
      "deterministic_time_enabled": false,
      "deterministic_time_iso": "2026-01-27T12:00:00Z",
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
      "chats_dir": "logs",
      "models_dir": "models"
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
        data: Dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

    deterministic_time_enabled = bool(data.get("deterministic_time_enabled", False))
    deterministic_time_iso = data.get("deterministic_time_iso")
    seed = data.get("seed")
    reseed_each_turn = data.get("reseed_each_turn")
    if deterministic_time_enabled and not deterministic_time_iso:
        deterministic_time_iso = DEFAULT_DETERMINISTIC_TIME_ISO
        logger.warning(
            "deterministic_time_enabled is true but deterministic_time_iso is missing; "
            "defaulting deterministic_time_iso=%s",
            deterministic_time_iso,
        )
    if deterministic_time_enabled:
        if seed is None or (isinstance(seed, (int, float)) and int(seed) < 0):
            logger.warning(
                "deterministic_time_enabled is true but seed is missing or random; "
                "defaulting seed=%s and reseed_each_turn=%s",
                DEFAULT_DETERMINISTIC_SEED,
                DEFAULT_DETERMINISTIC_RESEED_EACH_TURN,
            )
            seed = DEFAULT_DETERMINISTIC_SEED
            if reseed_each_turn is None:
                reseed_each_turn = DEFAULT_DETERMINISTIC_RESEED_EACH_TURN
        elif reseed_each_turn is None:
            logger.warning(
                "deterministic_time_enabled is true but reseed_each_turn is missing; "
                "defaulting reseed_each_turn=%s",
                DEFAULT_DETERMINISTIC_RESEED_EACH_TURN,
            )
            reseed_each_turn = DEFAULT_DETERMINISTIC_RESEED_EACH_TURN

    if deterministic_time_enabled:
        now, now_utc = _parse_deterministic_time(deterministic_time_iso)
    else:
        now = datetime.now()
        now_utc = datetime.now(timezone.utc)

    user_placeholders: Dict[str, str] = {
        str(k): str(v) for k, v in (data.get("placeholders", {}) or {}).items()
    }
    placeholder_keys_upper = {key.upper() for key in user_placeholders}
    builtin_placeholders = _build_builtin_placeholders(now, now_utc)
    for key, value in builtin_placeholders.items():
        if key.upper() not in placeholder_keys_upper:
            user_placeholders[key] = value

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
        deterministic_time_enabled=deterministic_time_enabled,
        deterministic_time_iso=deterministic_time_iso,
        max_tokens=data.get("max_tokens"),
        max_context_tokens=data.get("max_context_tokens"),
        perf_prompt_token_budget=data.get("perf_prompt_token_budget"),
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        min_p=data.get("min_p"),
        min_tokens_to_keep=data.get("min_tokens_to_keep"),
        top_k=data.get("top_k"),
        xtc_probability=data.get("xtc_probability"),
        xtc_threshold=data.get("xtc_threshold"),
        xtc_special_tokens=data.get("xtc_special_tokens") if data.get("xtc_special_tokens") is not None else None,
        repetition_penalty=data.get("repetition_penalty"),
        repetition_context_size=data.get("repetition_context_size"),
        mlock=data.get("mlock"),
        seed=seed,
        reseed_each_turn=reseed_each_turn,
        truncate_thinking=data.get("truncate_thinking"),
        truncate_response=data.get("truncate_response"),
        logs_dir=data.get("logs_dir"),
        chats_dir=data.get("chats_dir"),
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


def load_profiles(path: str | Path) -> Dict[str, Dict[str, Any]]:
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


def apply_performance_overrides(
    prompt_config: PromptConfig | None,
    *,
    performance_mode: bool | None = None,
    perf_max_tokens: int | None = None,
    perf_max_context_tokens: int | None = None,
    perf_max_kv_size: int | None = None,
) -> PromptConfig | None:
    updates: Dict[str, Any] = {}
    if performance_mode is not None:
        updates["performance_mode"] = performance_mode
    if perf_max_tokens is not None:
        updates["perf_max_tokens"] = perf_max_tokens
    if perf_max_context_tokens is not None:
        updates["perf_max_context_tokens"] = perf_max_context_tokens
    if perf_max_kv_size is not None:
        updates["perf_max_kv_size"] = perf_max_kv_size
    if not updates:
        return prompt_config
    if prompt_config is None:
        return PromptConfig(**updates)
    return prompt_config.model_copy(update=updates)
