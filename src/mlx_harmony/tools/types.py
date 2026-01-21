from __future__ import annotations

from typing import Any, Dict

from openai_harmony import Message
from pydantic import BaseModel, ConfigDict


class ToolConfig(BaseModel):
    """Lightweight description of an enabled tool."""

    model_config = ConfigDict(extra="ignore")

    name: str
    enabled: bool


class ToolCall(BaseModel):
    """Represents a parsed tool call from a Harmony message."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    arguments: Dict[str, Any]
    raw_message: Message
