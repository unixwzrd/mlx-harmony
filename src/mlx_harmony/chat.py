from __future__ import annotations

import sys
import time
from typing import Any, Literal, TypedDict

from mlx_harmony.chat_bootstrap import bootstrap_chat
from mlx_harmony.chat_history import (
    display_resume_message,
    find_last_hyperparameters,
    make_message_id,
    make_timestamp,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.chat_io import (
    load_conversation,
    read_user_input,
    save_conversation,
    try_save_conversation,
)
from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.chat_turn import run_chat_turn
from mlx_harmony.chat_utils import (
    build_hyperparameters,
    parse_command,
    resolve_max_context_tokens,
    truncate_text,
)
from mlx_harmony.config import apply_placeholders
from mlx_harmony.logging import get_logger


def _collect_memory_stats() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception:
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats


# Type definitions for message records
class MessageDict(TypedDict, total=False):
    """Type definition for conversation messages."""
    id: str
    parent_id: str | None
    cache_key: str | None
    role: Literal["user", "assistant", "tool", "system", "developer"]
    content: str
    timestamp: str
    name: str  # Optional: for tool messages
    recipient: str  # Optional: for tool messages
    channel: str  # Optional: for Harmony messages (e.g., "commentary", "analysis", "final")
    analysis: str  # Optional: for assistant messages with thinking/analysis
    hyperparameters: dict[str, float | int | bool | str]  # Optional: hyperparameters used for generation


logger = get_logger(__name__)


def main() -> None:
    bootstrap = bootstrap_chat()
    args = bootstrap.args
    conversation = bootstrap.conversation
    model_path = bootstrap.model_path
    prompt_config_path = bootstrap.prompt_config_path
    context = bootstrap.context
    loaded_hyperparameters = bootstrap.loaded_hyperparameters
    loaded_max_context_tokens = bootstrap.loaded_max_context_tokens
    loaded_model_path = bootstrap.loaded_model_path
    loaded_chat_id = bootstrap.loaded_chat_id

    print(f"[INFO] Starting chat with model: {model_path}")
    if context.generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected - Harmony format enabled.")
        if context.tools:
            enabled = ", ".join(t.name for t in context.tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non-GPT-OSS model - using native chat template.")

    print("[INFO] Type 'q' or `Control-D` to quit.")
    print("[INFO] Type '\\help' to list all out-of-band commands.")
    print("[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7').")
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if context.chat_file_path:
        print(f"[INFO] Chat will be saved to: {context.chat_file_path}\n")

    has_conversation_history = display_resume_message(
        conversation,
        context.assistant_name,
        context.thinking_limit,
        context.response_limit,
        context.render_markdown,
        display_assistant,
        display_thinking,
        truncate_text,
    )

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info (or after resume message)
    # Only show greeting if we don't have conversation history
    if (
        not has_conversation_history
        and context.prompt_config
        and context.prompt_config.assistant_greeting
    ):
        greeting_text = apply_placeholders(
            context.prompt_config.assistant_greeting,
            context.prompt_config.placeholders,
        )
        # Use markdown rendering for greeting if enabled
        display_assistant(greeting_text, context.assistant_name, context.render_markdown)
        parent_id = conversation[-1].get("id") if conversation else None
        message_id = make_message_id()
        conversation.append(
            {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": "assistant",
                "content": greeting_text,
                "timestamp": make_timestamp(),
            }
        )

    max_tool_iterations = 10  # Prevent infinite loops
    max_resume_attempts = 2
    resume_base_hyperparameters: dict[str, float | int | bool | str] | None = None

    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters,
        context.prompt_config,
        context.generator.is_gpt_oss and context.generator.use_harmony,
    )
    last_saved_hyperparameters = find_last_hyperparameters(conversation) or loaded_hyperparameters.copy()
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=context.prompt_config,
        profile_data=context.profile_data,
        model_path=model_path,
    )

    # Debug file info is logged above before resuming chat.
    last_prompt_start_time: float | None = None
    generation_index = 0

    while True:
        try:
            user_input = read_user_input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if user_input.strip().lower() == "q":
            break

        handled, should_apply, message, updates = parse_command(user_input, hyperparameters)
        if handled:
            if message:
                print(message)
            if should_apply and updates:
                hyperparameters.update(updates)
                if context.chat_file_path and conversation:
                    error = try_save_conversation(
                        context.chat_file_path,
                        conversation,
                        model_path,
                        prompt_config_path,
                        context.tools,
                        hyperparameters,
                    )
                    if error:
                        logger.warning(
                            "Failed to save updated hyperparameters: %s (check file path permissions)",
                            error,
                        )
            continue

        # Add timestamp to user message (turn)
        last_user_text = user_input
        user_content = (
            apply_placeholders(user_input, context.prompt_config.placeholders)
            if context.prompt_config
            else user_input
        )
        parent_id = conversation[-1].get("id") if conversation else None
        message_id = make_message_id()
        user_turn = {
            "id": message_id,
            "parent_id": parent_id,
            "cache_key": message_id,
            "role": "user",
            "content": user_content,
            "timestamp": make_timestamp(),
        }
        conversation.append(user_turn)

        result = run_chat_turn(
            generator=context.generator,
            conversation=conversation,
            hyperparameters=hyperparameters,
            last_saved_hyperparameters=last_saved_hyperparameters,
            assistant_name=context.assistant_name,
            thinking_limit=context.thinking_limit,
            response_limit=context.response_limit,
            render_markdown=context.render_markdown,
            debug=args.debug,
            debug_path=context.debug_path,
            debug_tokens=args.debug_tokens,
            enable_artifacts=bool(args.debug or args.debug_file),
            max_context_tokens=max_context_tokens,
            max_tool_iterations=max_tool_iterations,
            max_resume_attempts=max_resume_attempts,
            tools=context.tools,
            last_user_text=last_user_text,
            make_message_id=make_message_id,
            make_timestamp=make_timestamp,
            display_assistant=display_assistant,
            display_thinking=display_thinking,
            truncate_text=truncate_text,
            collect_memory_stats=_collect_memory_stats,
            write_debug_metrics=write_debug_metrics,
            write_debug_response=write_debug_response,
            write_debug_token_texts=write_debug_token_texts,
            write_debug_tokens=write_debug_tokens,
            last_prompt_start_time=last_prompt_start_time,
            generation_index=generation_index,
        )
        hyperparameters = result.hyperparameters
        last_saved_hyperparameters = result.last_saved_hyperparameters
        generation_index = result.generation_index
        last_prompt_start_time = result.last_prompt_start_time

        if context.chat_file_path:
            error = try_save_conversation(
                context.chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                context.prompt_config.model_dump() if context.prompt_config else None,
                context.tools,
                hyperparameters,
                max_context_tokens,
                chat_id,
            )
            if error:
                logger.warning(
                    "Failed to save chat: %s (check file path permissions)",
                    error,
                )

    # Final save on exit
    if context.chat_file_path and conversation:
        error = try_save_conversation(
            context.chat_file_path,
            conversation,
            model_path,
            prompt_config_path,
            context.prompt_config.model_dump() if context.prompt_config else None,
            context.tools,
            hyperparameters,
            max_context_tokens,
            chat_id,
        )
        if error:
            logger.warning(
                "Failed to save chat on exit: %s (check file path permissions)",
                error,
            )
        else:
            print(f"\n[INFO] Chat saved to: {context.chat_file_path}")


if __name__ == "__main__":
    main()
