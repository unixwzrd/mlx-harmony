from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from mlx_harmony.cli.chat_commands import (
    build_hyperparameters,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
    resolve_profile_and_prompt_config,
    truncate_text,
)
from mlx_harmony.cli.chat_input import handle_user_command, read_chat_input
from mlx_harmony.cli.chat_turn import run_generation_loop
from mlx_harmony.cli.chat_voice import (
    init_moshi_components,
)
from mlx_harmony.cli.cli_args import build_parser
from mlx_harmony.config import (
    MoshiConfig,
    apply_placeholders,
    load_profiles,
    load_prompt_config,
)
from mlx_harmony.conversation.conversation_history import (
    display_resume_message,
    find_last_hyperparameters,
    load_chat_session,
    make_message_id,
    make_timestamp,
    resolve_chat_paths,
    resolve_debug_path,
    resolve_dirs_from_config,
)
from mlx_harmony.conversation.conversation_io import (
    load_conversation,
    save_conversation,
    try_save_conversation,
)
from mlx_harmony.generation.generator import TokenGenerator
from mlx_harmony.generation.prompt_cache import PromptTokenCache
from mlx_harmony.logging import get_logger
from mlx_harmony.render_output import display_assistant, display_thinking
from mlx_harmony.tools import (
    get_tools_for_model,
)


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
    hyperparameters: dict[str, float | int | bool]  # Optional: hyperparameters used for generation


logger = get_logger(__name__)

__all__ = ["load_conversation", "save_conversation", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path, prompt_config_path, prompt_config, profile_data = resolve_profile_and_prompt_config(
        args,
        load_profiles,
        load_prompt_config,
    )

    # Resolve directories from config (defaults to "logs" if not specified)
    # These will be re-resolved after loading chat if prompt_config changes
    chats_dir, logs_dir = resolve_dirs_from_config(prompt_config)

    chat_file_path, load_file_path, chat_name, chat_input_path = resolve_chat_paths(
        args.chat, chats_dir
    )
    if load_file_path and chat_file_path and load_file_path != chat_file_path:
        print(f"[INFO] Found chat file at: {load_file_path} (will save to: {chat_file_path})")

    (
        conversation,
        model_path,
        prompt_config_path,
        prompt_config,
        updated_chats_dir,
        updated_logs_dir,
        chat_file_path,
        loaded_hyperparameters,
        loaded_max_context_tokens,
        loaded_model_path,
        loaded_chat_id,
    ) = load_chat_session(
        load_file_path=load_file_path,
        chat_file_path=chat_file_path,
        chat_arg=args.chat,
        model_path=model_path,
        prompt_config_path=prompt_config_path,
        prompt_config=prompt_config,
        load_conversation=load_conversation,
        load_prompt_config=load_prompt_config,
        resolve_dirs=resolve_dirs_from_config,
    )
    if updated_chats_dir is not None and updated_logs_dir is not None:
        chats_dir = updated_chats_dir
        logs_dir = updated_logs_dir

    # Get mlock from prompt config if not explicitly set via CLI
    mlock = args.mlock
    if mlock is None and prompt_config:
        mlock = prompt_config.mlock
    no_fs_cache = bool(args.no_fs_cache)
    if not no_fs_cache and prompt_config and prompt_config.no_fs_cache is not None:
        no_fs_cache = bool(prompt_config.no_fs_cache)
    lazy = args.lazy if args.lazy is not None else False

    use_harmony = None
    if prompt_config and prompt_config.use_harmony is not None:
        use_harmony = prompt_config.use_harmony

    generator = TokenGenerator(
        model_path,
        use_harmony=use_harmony,
        prompt_config=prompt_config,
        lazy=lazy,
        mlock=mlock or False,  # Default to False if None
        no_fs_cache=no_fs_cache,
    )
    if generator.use_harmony and generator.encoding:
        generator.prompt_token_cache = PromptTokenCache()

    tools = []
    if generator.is_gpt_oss:
        tools = get_tools_for_model(
            browser=args.browser,
            python=args.use_python,
            apply_patch=args.apply_patch,
        )

    print(f"[INFO] Starting chat with model: {model_path}")
    if generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected - Harmony format enabled.")
        if tools:
            enabled = ", ".join(t.name for t in tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non-GPT-OSS model - using native chat template.")

    print("[INFO] Type 'q' or `Control-D` to quit.")
    print("[INFO] Type '\\help' to list all out-of-band commands.")
    print(
        "[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7')."
    )
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if chat_file_path:
        print(f"[INFO] Chat will be saved to: {chat_file_path}\n")

    moshi_config, moshi_stt, moshi_tts, smoke_ran = init_moshi_components(args)
    if smoke_ran:
        return

    # Debug file setup: always write debug log by default
    # --debug enables console output, --debug-file overrides the file path
    # Path resolution: absolute paths used as-is, relative paths resolved relative to logs_dir
    # This matches --chat behavior (relative paths resolved relative to chats_dir)
    # Always write to debug file by default (unless explicitly disabled in future)
    debug_path = resolve_debug_path(args.debug_file, logs_dir)
    debug_tokens_mode = args.debug_tokens or "off"
    logger.info("Debug log: %s (tokens: %s)", debug_path, debug_tokens_mode)

    # Get assistant name for display (need this before resume display)
    assistant_name = get_assistant_name(prompt_config)

    # Get truncate limits once (used multiple times)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)

    # Get render_markdown setting once (used multiple times)
    render_markdown = not args.no_markdown if hasattr(args, "no_markdown") else True

    has_conversation_history = display_resume_message(
        conversation,
        assistant_name,
        thinking_limit,
        response_limit,
        render_markdown,
        display_assistant,
        display_thinking,
        truncate_text,
    )

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info (or after resume message)
    # Only show greeting if we don't have conversation history
    if not has_conversation_history and prompt_config and prompt_config.assistant_greeting:
        greeting_text = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        # Use markdown rendering for greeting if enabled
        display_assistant(greeting_text, assistant_name, render_markdown)
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

    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters,
        prompt_config,
        generator.is_gpt_oss and generator.use_harmony,
    )
    last_saved_hyperparameters = (
        find_last_hyperparameters(conversation) or loaded_hyperparameters.copy()
    )
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=prompt_config,
        profile_data=profile_data,
        model_path=model_path,
    )

    # Debug file info is logged above before resuming chat.
    last_prompt_start_time: float | None = None
    generation_index = 0

    def _apply_user_token_limit(text: str) -> str | None:
        if not prompt_config:
            return text
        max_user_tokens = prompt_config.max_user_tokens
        if max_user_tokens is None:
            max_user_tokens = prompt_config.max_tokens
        if max_user_tokens is None:
            return text
        if not text.strip():
            return text
        tokens = generator.tokenizer.encode(text)
        token_count = len(tokens)
        limit = max_user_tokens
        if token_count <= limit:
            return text
        if sys.stdin.isatty():
            print(
                f"[WARN] Input is {token_count} tokens (limit {limit})."
                " Proceed with truncated input? [y/N]: ",
                end="",
                flush=True,
            )
            response = read_user_input("").strip().lower()
            if not response.startswith("y"):
                print("[INFO] Input discarded. Please try again.")
                return None
        else:
            logger.warning(
                "Input is %s tokens (limit %s). Truncating for non-interactive input.",
                token_count,
                limit,
            )
        truncated = generator.tokenizer.decode(tokens[:limit])
        print("[INFO] Using truncated input.")
        return truncated

    while True:
        try:
            user_input = read_chat_input(
                moshi_stt=moshi_stt,
                moshi_config=moshi_config,
            )
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if moshi_stt is not None and not user_input.strip():
            continue
        if user_input.strip().lower() == "q":
            break

        if handle_user_command(
            user_input=user_input,
            hyperparameters=hyperparameters,
            chat_file_path=chat_file_path,
            conversation=conversation,
            model_path=model_path,
            prompt_config_path=prompt_config_path,
            tools=tools,
        ):
            continue

        # Add timestamp to user message (turn)
        user_content = (
            apply_placeholders(user_input, prompt_config.placeholders)
            if prompt_config and prompt_config.placeholders
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
        user_content_limited = _apply_user_token_limit(user_content)
        if user_content_limited is None:
            continue
        user_content = user_content_limited

        last_prompt_start_time, generation_index, last_saved_hyperparameters = run_generation_loop(
            generator=generator,
            conversation=conversation,
            prompt_config=prompt_config,
            prompt_config_path=prompt_config_path,
            tools=tools,
            hyperparameters=hyperparameters,
            assistant_name=assistant_name,
            thinking_limit=thinking_limit,
            response_limit=response_limit,
            render_markdown=render_markdown,
            debug_path=debug_path,
            args=args,
            max_context_tokens=max_context_tokens,
            moshi_stt=moshi_stt,
            moshi_tts=moshi_tts,
            moshi_config=moshi_config,
            last_saved_hyperparameters=last_saved_hyperparameters,
            last_prompt_start_time=last_prompt_start_time,
            generation_index=generation_index,
            chat_file_path=chat_file_path,
            model_path=model_path,
            chat_id=chat_id,
            max_tool_iterations=max_tool_iterations,
        )

    # Final save on exit
    if chat_file_path and conversation:
        error = try_save_conversation(
            chat_file_path,
            conversation,
            model_path,
            prompt_config_path,
            prompt_config.model_dump() if prompt_config else None,
            tools,
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
            print(f"\n[INFO] Chat saved to: {chat_file_path}")


if __name__ == "__main__":
    main()
