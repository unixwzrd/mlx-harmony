from __future__ import annotations

from mlx_harmony.chat.input import (
    apply_user_token_limit,
    handle_user_command,
    read_chat_input,
)
from mlx_harmony.chat.session import initialize_chat_session
from mlx_harmony.chat.turn import run_generation_loop
from mlx_harmony.cli.cli_args import build_parser
from mlx_harmony.config import apply_placeholders
from mlx_harmony.conversation.conversation_history import (
    make_message_id,
    make_timestamp,
)
from mlx_harmony.conversation.conversation_io import try_save_conversation
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)

__all__ = ["main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    state = initialize_chat_session(args)
    if state.smoke_ran:
        return

    last_prompt_start_time: float | None = None
    generation_index = 0

    while True:
        try:
            user_input = read_chat_input(
                moshi_stt=state.moshi_stt,
                moshi_config=state.moshi_config,
            )
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if state.moshi_stt is not None and not user_input.strip():
            continue
        if user_input.strip().lower() == "q":
            break

        if handle_user_command(
            user_input=user_input,
            hyperparameters=state.hyperparameters,
            chat_file_path=state.chat_file_path,
            conversation=state.conversation,
            model_path=state.model_path,
            prompt_config_path=state.prompt_config_path,
            tools=state.tools,
        ):
            continue

        # Add timestamp to user message (turn)
        user_content = (
            apply_placeholders(user_input, state.prompt_config.placeholders)
            if state.prompt_config and state.prompt_config.placeholders
            else user_input
        )
        parent_id = state.conversation[-1].get("id") if state.conversation else None
        message_id = make_message_id()
        user_turn = {
            "id": message_id,
            "parent_id": parent_id,
            "cache_key": message_id,
            "role": "user",
            "content": user_content,
            "timestamp": make_timestamp(),
        }
        state.conversation.append(user_turn)
        user_content_limited = apply_user_token_limit(
            text=user_content,
            prompt_config=state.prompt_config,
            generator=state.generator,
        )
        if user_content_limited is None:
            continue
        user_content = user_content_limited

        last_prompt_start_time, generation_index, state.last_saved_hyperparameters = run_generation_loop(
            generator=state.generator,
            conversation=state.conversation,
            prompt_config=state.prompt_config,
            prompt_config_path=state.prompt_config_path,
            tools=state.tools,
            hyperparameters=state.hyperparameters,
            assistant_name=state.assistant_name,
            thinking_limit=state.thinking_limit,
            response_limit=state.response_limit,
            render_markdown=state.render_markdown,
            debug_path=state.debug_path,
            args=args,
            max_context_tokens=state.max_context_tokens,
            moshi_stt=state.moshi_stt,
            moshi_tts=state.moshi_tts,
            moshi_config=state.moshi_config,
            last_saved_hyperparameters=state.last_saved_hyperparameters,
            last_prompt_start_time=last_prompt_start_time,
            generation_index=generation_index,
            chat_file_path=state.chat_file_path,
            model_path=state.model_path,
            chat_id=state.chat_id,
            max_tool_iterations=10,
        )

    # Final save on exit
    if state.chat_file_path and state.conversation:
        error = try_save_conversation(
            state.chat_file_path,
            state.conversation,
            state.model_path,
            state.prompt_config_path,
            state.prompt_config.model_dump() if state.prompt_config else None,
            state.tools,
            state.hyperparameters,
            state.max_context_tokens,
            state.chat_id,
        )
        if error:
            logger.warning(
                "Failed to save chat on exit: %s (check file path permissions)",
                error,
            )
        else:
            print(f"\n[INFO] Chat saved to: {state.chat_file_path}")


if __name__ == "__main__":
    main()
