from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


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

    # User-defined placeholders for template expansion
    placeholders: Dict[str, str] = field(default_factory=dict)

    # Sampling defaults (any model)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    min_tokens_to_keep: Optional[int] = None
    top_k: Optional[int] = None
    xtc_probability: Optional[float] = None
    xtc_threshold: Optional[float] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None


_PLACEHOLDER_RE = re.compile(r"<\|([A-Z_]+)\|>")


def _render_placeholders(value: str, user_placeholders: Dict[str, str]) -> str:
    """Expand dynamic placeholders and user-defined symbols."""

    def repl(match: re.Match) -> str:
        token = match.group(1)
        if token == "DATE":
            return datetime.now().strftime("%Y-%m-%d")
        if token == "DATETIME":
            return datetime.now().isoformat(timespec="seconds")
        return match.group(0)

    # First expand built-ins
    value = _PLACEHOLDER_RE.sub(repl, value)
    # Then apply user-defined placeholders: {key} → value
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
      "system_model_identity": "You are {assistant} on <|DATE|> at <|DATETIME|>.",
      "reasoning_effort": "Medium",
      "conversation_start_date": "<|DATE|>",
      "knowledge_cutoff": "2025-01",
      "developer_instructions": "Always answer concisely.",
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
    """
    raw_path = Path(path)
    data: Dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))

    user_placeholders: Dict[str, str] = {
        str(k): str(v) for k, v in (data.get("placeholders", {}) or {}).items()
    }

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

