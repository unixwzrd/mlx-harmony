from __future__ import annotations

"""Shared backend helpers for server-side chat execution."""

from dataclasses import dataclass
from typing import Any, Iterable

from mlx_harmony.chat_turn import run_chat_turn
from mlx_harmony.config import apply_placeholders


@dataclass(frozen=True)
class BackendChatResult:
    assistant_text: str
    analysis_text: str | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    last_prompt_start_time: float | None
    generation_index: int = 0


def build_conversation_from_messages(
    *,
    messages: Iterable[dict[str, str]],
    prompt_config: Any,
    make_message_id: Any,
    make_timestamp: Any,
) -> list[dict[str, object]]:
    """Build a conversation list from messages, applying prompt-config defaults.

    Args:
        messages: Raw messages with role/content pairs.
        prompt_config: Prompt config used for placeholders and greetings.
        make_message_id: Callable to generate message IDs.
        make_timestamp: Callable to generate timestamps.

    Returns:
        Normalized conversation list with ids, parent links, and timestamps.
    """
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if not role:
            continue
        content = msg.get("content")
        if content is None:
            continue
        if prompt_config and role == "user":
            content = apply_placeholders(content, getattr(prompt_config, "placeholders", {}))
        normalized.append({"role": role, "content": content})

    has_assistant = any(msg.get("role") == "assistant" for msg in normalized)
    if (
        prompt_config
        and getattr(prompt_config, "assistant_greeting", None)
        and not has_assistant
    ):
        greeting_text = apply_placeholders(
            getattr(prompt_config, "assistant_greeting", ""),
            getattr(prompt_config, "placeholders", {}),
        )
        normalized.insert(0, {"role": "assistant", "content": greeting_text})

    conversation: list[dict[str, object]] = []
    parent_id: str | None = None
    for msg in normalized:
        message_id = make_message_id()
        conversation.append(
            {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": make_timestamp(),
            }
        )
        parent_id = message_id
    return conversation


def run_backend_chat(
    *,
    generator: Any,
    conversation: list[dict[str, object]],
    hyperparameters: dict[str, float | int | bool | str],
    assistant_name: str,
    thinking_limit: int | None,
    response_limit: int | None,
    render_markdown: bool,
    debug_path: Any,
    debug_tokens: str | None,
    enable_artifacts: bool,
    max_context_tokens: int | None,
    max_tool_iterations: int,
    max_resume_attempts: int,
    tools: list[Any],
    last_user_text: str | None,
    make_message_id: Any,
    make_timestamp: Any,
    collect_memory_stats: Any,
    write_debug_metrics: Any,
    write_debug_response: Any,
    write_debug_info: Any,
    write_debug_token_texts: Any,
    write_debug_tokens: Any,
    last_prompt_start_time: float | None,
    generation_index: int,
) -> BackendChatResult:
    """Run a chat turn using the shared turn pipeline.

    Args:
        generator: TokenGenerator instance.
        conversation: Mutable conversation list to append assistant turns.
        hyperparameters: Merged hyperparameters dict.
        assistant_name: Assistant display name.
        thinking_limit: Max thinking tokens to render.
        response_limit: Max response tokens to render.
        render_markdown: Whether to render markdown output.
        debug_path: Debug file path for artifacts.
        debug_tokens: Debug token mode.
        enable_artifacts: Whether to emit prompt/response artifacts.
        max_context_tokens: Max context length for truncation.
        max_tool_iterations: Max tool call iterations per turn.
        max_resume_attempts: Max retry attempts when resuming.
        tools: Tool configs available to the model.
        last_user_text: Last user message for retry context.
        make_message_id: Callable to generate message IDs.
        make_timestamp: Callable to generate timestamps.
        collect_memory_stats: Callable to collect memory metrics.
        write_debug_metrics: Callable to write debug metrics.
        write_debug_response: Callable to write debug responses.
        write_debug_info: Callable to write debug info.
        write_debug_token_texts: Callable to write decoded tokens.
        write_debug_tokens: Callable to write token ids.
        last_prompt_start_time: Prior prompt timestamp for metrics.
        generation_index: Generation counter for seeding.

    Returns:
        BackendChatResult with assistant text, analysis, and token counts.
    """
    result = run_chat_turn(
        generator=generator,
        conversation=conversation,
        hyperparameters=hyperparameters,
        last_saved_hyperparameters={},
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
        debug=False,
        debug_path=debug_path,
        debug_tokens=debug_tokens,
        enable_artifacts=enable_artifacts,
        max_context_tokens=max_context_tokens,
        max_tool_iterations=max_tool_iterations,
        max_resume_attempts=max_resume_attempts,
        tools=tools,
        last_user_text=last_user_text or "",
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        display_assistant=lambda *_args, **_kwargs: None,
        display_thinking=lambda *_args, **_kwargs: None,
        truncate_text=lambda text, _limit: text,
        collect_memory_stats=collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        last_prompt_start_time=last_prompt_start_time,
        generation_index=generation_index,
    )

    assistant_text = ""
    analysis_text: str | None = None
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            assistant_text = str(msg.get("content") or "")
            if msg.get("analysis"):
                analysis_text = str(msg.get("analysis"))
            break

    prompt_tokens = result.prompt_tokens or 0
    completion_tokens = result.completion_tokens or 0
    finish_reason = generator.last_finish_reason
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "stop"

    return BackendChatResult(
        assistant_text=assistant_text,
        analysis_text=analysis_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        last_prompt_start_time=result.last_prompt_start_time,
        generation_index=result.generation_index,
    )
