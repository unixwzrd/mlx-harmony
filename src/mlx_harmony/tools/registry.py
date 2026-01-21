from __future__ import annotations

from typing import List

from mlx_harmony.tools.types import ToolConfig


def get_tools_for_model(
    *,
    browser: bool = False,
    python: bool = False,
    apply_patch: bool = False,
) -> List[ToolConfig]:
    """
    Return a list of enabled tools for a GPT‑OSS model.

    Args:
        browser: Enable browser tool.
        python: Enable Python tool.
        apply_patch: Enable apply_patch tool.

    Returns:
        List of ToolConfig objects for enabled tools.
    """
    tools: List[ToolConfig] = []
    if browser:
        tools.append(ToolConfig(name="browser", enabled=True))
    if python:
        tools.append(ToolConfig(name="python", enabled=True))
    if apply_patch:
        tools.append(ToolConfig(name="apply_patch", enabled=True))
    return tools
