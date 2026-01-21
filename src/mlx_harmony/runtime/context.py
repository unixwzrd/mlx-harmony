from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_harmony.config import PromptConfig
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.tools import ToolConfig


@dataclass(frozen=True)
class RunContext:
    generator: TokenGenerator
    tools: list[ToolConfig]
    prompt_config: PromptConfig | None
    profile_data: dict[str, Any] | None
    chats_dir: str
    logs_dir: str
    chat_file_path: str | None
    chat_input_path: str | None
    debug_path: str
    assistant_name: str
    thinking_limit: int | None
    response_limit: int | None
    render_markdown: bool
