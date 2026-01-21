"""
GPT-OSS tool integration.

This module provides tool call parsing, execution, and integration for GPT-OSS models
using the Harmony format. Tools are detected from parsed messages and executed,
with results fed back into the conversation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai_harmony import Message, TextContent

from mlx_harmony.tools.types import ToolCall, ToolConfig


def parse_tool_calls_from_messages(
    messages: List[Message], enabled_tools: List[ToolConfig]
) -> List[ToolCall]:
    """
    Parse tool calls from Harmony messages.

    Tool calls are identified by messages with a recipient field (e.g., `to=browser.navigate`).
    """
    tool_calls: List[ToolCall] = []
    enabled_names = {t.name for t in enabled_tools if t.enabled}

    for msg in messages:
        recipient = getattr(msg, "recipient", None)
        if not recipient:
            continue

        # Parse tool name from recipient (e.g., "browser.navigate" -> "browser")
        tool_name = recipient.split(".")[0] if "." in recipient else recipient

        if tool_name not in enabled_names:
            continue

        # Extract arguments from message content
        content = msg.content
        text = ""
        if isinstance(content, TextContent):
            text = content.text or ""
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, TextContent):
                    text += part.text or ""

        if text:
            try:
                arguments = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                arguments = {"input": text}
        else:
            arguments = {}

        tool_calls.append(
            ToolCall(tool_name=tool_name, arguments=arguments, raw_message=msg)
        )

    return tool_calls


def execute_tool_call(tool_call: ToolCall) -> str:
    """
    Execute a tool call and return the result as a string.

    This is a stub implementation. Real tool executors should be added as separate modules.
    """
    tool_name = tool_call.tool_name
    args = tool_call.arguments

    if tool_name == "browser":
        return _execute_browser_tool(args)
    elif tool_name == "python":
        return _execute_python_tool(args)
    elif tool_name == "apply_patch":
        return _execute_apply_patch_tool(args)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _execute_browser_tool(args: Dict[str, Any]) -> str:
    """Execute browser tool (stub)."""
    # TODO: Implement actual browser tool using aiohttp or similar
    return json.dumps(
        {
            "status": "not_implemented",
            "tool": "browser",
            "message": "Browser tool execution not yet implemented",
        }
    )


def _execute_python_tool(args: Dict[str, Any]) -> str:
    """Execute Python tool (stub)."""
    # TODO: Implement actual Python tool using docker or sandboxed execution
    return json.dumps(
        {
            "status": "not_implemented",
            "tool": "python",
            "message": "Python tool execution not yet implemented",
        }
    )


def _execute_apply_patch_tool(args: Dict[str, Any]) -> str:
    """Execute apply_patch tool (stub)."""
    # TODO: Implement actual apply_patch tool
    return json.dumps(
        {
            "status": "not_implemented",
            "tool": "apply_patch",
            "message": "Apply patch tool execution not yet implemented",
        }
    )


from mlx_harmony.tools.registry import get_tools_for_model
