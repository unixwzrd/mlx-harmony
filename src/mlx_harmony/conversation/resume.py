from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def find_last_assistant_message(
    conversation: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last assistant message in the conversation, if any."""
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            return msg
    return None


def display_resume_message(
    conversation: list[dict[str, Any]],
    assistant_name: str,
    thinking_limit: int,
    response_limit: int,
    render_markdown: bool,
    display_assistant: Callable[[str, str, bool], None],
    display_thinking: Callable[[str, bool], None],
    truncate_text: Callable[[str, int], str],
) -> bool:
    """Display the last assistant response or analysis when resuming a chat."""
    if not conversation:
        return False

    logger.info("Resuming prior chat...")

    last_assistant_msg = find_last_assistant_message(conversation)
    if not last_assistant_msg:
        logger.info("No assistant messages found in conversation history.")
        return True

    content = last_assistant_msg.get("content", "")
    if content in ("[Analysis only - no final response]", "[No final response - see thinking above]"):
        analysis = last_assistant_msg.get("analysis", "")
        if analysis:
            display_analysis = truncate_text(analysis, thinking_limit)
            if display_analysis.strip():
                display_thinking(display_analysis, render_markdown)
                print(
                    "\n[WARNING] Previous turn only generated analysis. "
                    "Check max_tokens or repetition_penalty.\n"
                )
        return True

    if content:
        display_content = truncate_text(content, response_limit)
        display_assistant(display_content, assistant_name, render_markdown)
        return True

    analysis = last_assistant_msg.get("analysis", "")
    if analysis:
        display_analysis = truncate_text(analysis, thinking_limit)
        if display_analysis.strip():
            display_thinking(display_analysis, render_markdown)
    return True
