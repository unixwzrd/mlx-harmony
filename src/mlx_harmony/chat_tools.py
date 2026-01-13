from __future__ import annotations

from typing import Any

from mlx_harmony.chat_history import make_message_id, make_timestamp
from mlx_harmony.logging import get_logger
from mlx_harmony.tools import execute_tool_call, parse_tool_calls_from_messages

logger = get_logger(__name__)


def handle_tool_calls(
    *,
    generator: Any,
    tokens: list[int],
    parsed_messages: list[Any] | None,
    tools: list[Any],
    conversation: list[dict[str, Any]],
    hyperparameters: dict[str, float | int],
) -> tuple[bool, list[Any] | None]:
    """
    Execute tool calls when present, append results to conversation, and signal continuation.

    Returns:
        (should_continue, parsed_messages)
    """
    if not (generator.is_gpt_oss and tools and generator.use_harmony):
        return False, parsed_messages

    try:
        if parsed_messages is None:
            parsed_messages = generator.parse_messages_from_tokens(tokens)
        tool_calls = parse_tool_calls_from_messages(parsed_messages, tools)

        if not tool_calls:
            return False, parsed_messages

        logger.info("Detected %d tool call(s)", len(tool_calls))
        for tool_call in tool_calls:
            logger.info("Executing tool: %s with args: %s", tool_call.tool_name, tool_call.arguments)
            result = execute_tool_call(tool_call)
            logger.info("Tool result: %s", result)

            parent_id = conversation[-1].get("id") if conversation else None
            message_id = make_message_id()
            tool_result_msg = {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": "tool",
                "name": tool_call.tool_name,
                "content": result,
                "recipient": "assistant",
                "channel": "commentary",
                "timestamp": make_timestamp(),
                "hyperparameters": hyperparameters.copy() if hyperparameters else {},
            }
            conversation.append(tool_result_msg)

        return True, parsed_messages
    except Exception as e:
        logger.warning(
            "Error parsing tool calls: %s (check tool call JSON and enabled tools)",
            e,
        )
        return False, parsed_messages


def has_tool_calls(
    *,
    parsed_messages: list[Any] | None,
    tools: list[Any],
) -> bool:
    """Return True if parsed messages include tool calls for enabled tools."""
    if parsed_messages is None or not tools:
        return False
    try:
        return bool(parse_tool_calls_from_messages(parsed_messages, tools))
    except Exception:
        return False
