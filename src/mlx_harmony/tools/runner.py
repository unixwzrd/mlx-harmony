from __future__ import annotations

from typing import Any

from mlx_harmony.chat_history import make_message_id, make_timestamp
from mlx_harmony.logging import get_logger
from mlx_harmony.tools import execute_tool_call, parse_tool_calls_from_messages

logger = get_logger(__name__)


def run_tools_if_requested(
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
            parse_tokens = tokens
            parse_strict = True
            if (
                getattr(generator, "last_finish_reason", None) == "stop"
                and getattr(generator, "last_stop_token_id", None) is not None
                and (not tokens or tokens[-1] != generator.last_stop_token_id)
                and generator.last_stop_token_id not in tokens
            ):
                parse_tokens = tokens + [generator.last_stop_token_id]
            if generator.encoding and 200005 not in parse_tokens[:20]:
                header_tokens = generator.encoding.encode(
                    "<|start|>assistant<|channel|>analysis<|message|>",
                    allowed_special={"<|start|>", "<|channel|>", "<|message|>"},
                )
                parse_tokens = header_tokens + parse_tokens
                parse_strict = False
                logger.warning(
                    "Tool parsing with prepended assistant header (analysis channel) for completion-only parse"
                )
            parsed_messages = generator.parse_messages_from_tokens(parse_tokens, strict=parse_strict)
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
    except Exception as exc:
        logger.warning(
            "Error parsing tool calls: %s (check tool call JSON and enabled tools)",
            exc,
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
