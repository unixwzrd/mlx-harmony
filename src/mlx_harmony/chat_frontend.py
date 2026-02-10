from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_harmony.backend_contract import BackendGenerationRequest, FrontendBackend
from mlx_harmony.chat_backend import LocalBackend
from mlx_harmony.chat_history import (
    display_resume_message,
    find_last_hyperparameters,
    make_message_id,
    make_timestamp,
    write_debug_info,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.chat_io import read_user_input, try_save_conversation
from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.chat_utils import (
    build_hyperparameters,
    parse_command,
    resolve_max_context_tokens,
    truncate_text,
)
from mlx_harmony.config import apply_placeholders


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
from mlx_harmony.logging import get_logger
from mlx_harmony.runtime.context import RunContext

logger = get_logger(__name__)


@dataclass
class _FrontendRuntimeState:
    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    generation_index: int
    last_prompt_start_time: float | None


def _use_harmony_prompt_defaults(context: RunContext) -> bool:
    generator = getattr(context, "generator", None)
    if generator is None:
        return False
    return bool(getattr(generator, "is_gpt_oss", False) and getattr(generator, "use_harmony", False))


def _read_multiline_continuation(first_line: str) -> str:
    lines: list[str] = []
    pending = first_line
    while True:
        if pending.rstrip().endswith("\\"):
            lines.append(pending.rstrip()[:-1])
        else:
            lines.append(pending)
            break
        try:
            pending = read_user_input("... ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
    return "\n".join(lines)


def _process_user_input(
    *,
    user_input: str,
    args: Any,
    context: RunContext,
    conversation: list[dict[str, Any]],
    state: _FrontendRuntimeState,
    backend_impl: FrontendBackend,
    models_dir: str | None,
    max_context_tokens: int | None,
    max_tool_iterations: int,
    max_resume_attempts: int,
) -> _FrontendRuntimeState:
    handled, should_apply, message, updates = parse_command(
        user_input,
        state.hyperparameters,
        models_dir=models_dir,
        profiles_file=getattr(args, "profiles_file", None),
    )
    if handled:
        if message:
            print(message)
        if should_apply and updates:
            state.hyperparameters.update(updates)
        return state

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

    backend_request = BackendGenerationRequest(
        conversation=conversation,
        hyperparameters=state.hyperparameters,
        last_saved_hyperparameters=state.last_saved_hyperparameters,
        last_user_text=last_user_text,
        max_context_tokens=max_context_tokens,
        last_prompt_start_time=state.last_prompt_start_time,
        generation_index=state.generation_index,
        max_tool_iterations=max_tool_iterations,
        max_resume_attempts=max_resume_attempts,
        generator=context.generator,
        tools=context.tools,
        assistant_name=context.assistant_name,
        thinking_limit=context.thinking_limit,
        response_limit=context.response_limit,
        render_markdown=context.render_markdown,
        debug=args.debug,
        debug_path=context.debug_path,
        debug_tokens=args.debug_tokens,
        enable_artifacts=bool(args.debug or args.debug_file),
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        display_assistant=display_assistant,
        display_thinking=display_thinking,
        truncate_text=truncate_text,
        collect_memory_stats=_collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
    )
    backend_result = backend_impl.generate(request=backend_request)
    state.hyperparameters = backend_result.hyperparameters
    state.last_saved_hyperparameters = backend_result.last_saved_hyperparameters
    state.generation_index = backend_result.generation_index
    state.last_prompt_start_time = backend_result.last_prompt_start_time

    if not backend_result.handled_conversation:
        if backend_result.analysis_text:
            display_thinking(backend_result.analysis_text, context.render_markdown)
        if backend_result.assistant_text:
            display_assistant(
                backend_result.assistant_text,
                context.assistant_name,
                context.render_markdown,
            )
        parent_id = conversation[-1].get("id") if conversation else None
        message_id = make_message_id()
        assistant_turn = {
            "id": message_id,
            "parent_id": parent_id,
            "cache_key": message_id,
            "role": "assistant",
            "content": backend_result.assistant_text or "",
            "timestamp": make_timestamp(),
        }
        if backend_result.analysis_text:
            assistant_turn["analysis"] = backend_result.analysis_text
        conversation.append(assistant_turn)

    return state


def run_cli_frontend(
    *,
    args: Any,
    context: RunContext,
    conversation: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None,
    loaded_hyperparameters: dict[str, Any],
    loaded_max_context_tokens: int | None,
    loaded_model_path: str | None,
    loaded_chat_id: str | None,
    backend: FrontendBackend | None = None,
) -> None:
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

    if (
        not has_conversation_history
        and context.prompt_config
        and context.prompt_config.assistant_greeting
    ):
        greeting_text = apply_placeholders(
            context.prompt_config.assistant_greeting,
            context.prompt_config.placeholders,
        )
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

    max_tool_iterations = 10
    max_resume_attempts = 2

    state = _FrontendRuntimeState(
        hyperparameters=build_hyperparameters(
        args,
        loaded_hyperparameters,
        context.prompt_config,
        _use_harmony_prompt_defaults(context),
    ),
        last_saved_hyperparameters=find_last_hyperparameters(conversation) or loaded_hyperparameters.copy(),
        generation_index=0,
        last_prompt_start_time=None,
    )
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=context.prompt_config,
        profile_data=context.profile_data,
        model_path=model_path,
    )

    backend_impl = backend or LocalBackend()

    models_dir = None
    if context.prompt_config is not None:
        models_dir = getattr(context.prompt_config, "models_dir", None)

    while True:
        try:
            user_input = read_user_input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.rstrip().endswith("\\"):
            user_input = _read_multiline_continuation(user_input)
        if user_input.strip().lower() == "q":
            break

        previous_hyperparameters = state.hyperparameters.copy()
        state = _process_user_input(
            user_input=user_input,
            args=args,
            context=context,
            conversation=conversation,
            state=state,
            backend_impl=backend_impl,
            models_dir=models_dir,
            max_context_tokens=max_context_tokens,
            max_tool_iterations=max_tool_iterations,
            max_resume_attempts=max_resume_attempts,
        )
        if (
            context.chat_file_path
            and previous_hyperparameters != state.hyperparameters
            and conversation
        ):
            error = try_save_conversation(
                context.chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                context.prompt_config.model_dump() if context.prompt_config else None,
                context.tools,
                state.hyperparameters,
            )
            if error:
                logger.warning(
                    "Failed to save updated hyperparameters: %s (check file path permissions)",
                    error,
                )

        if context.chat_file_path:
            error = try_save_conversation(
                context.chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                context.prompt_config.model_dump() if context.prompt_config else None,
                context.tools,
                state.hyperparameters,
                max_context_tokens,
                chat_id,
            )
            if error:
                logger.warning(
                    "Failed to save chat: %s (check file path permissions)",
                    error,
                )

    if context.chat_file_path and conversation:
        error = try_save_conversation(
            context.chat_file_path,
            conversation,
            model_path,
            prompt_config_path,
            context.prompt_config.model_dump() if context.prompt_config else None,
            context.tools,
            state.hyperparameters,
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


def run_prompt_frontend(
    *,
    prompts: list[str],
    args: Any,
    context: RunContext,
    conversation: list[dict[str, Any]],
    model_path: str,
    prompt_config_path: str | None,
    loaded_hyperparameters: dict[str, Any],
    loaded_max_context_tokens: int | None,
    loaded_model_path: str | None,
    loaded_chat_id: str | None,
    backend: FrontendBackend,
) -> None:
    backend_impl = backend
    models_dir = None
    if context.prompt_config is not None:
        models_dir = getattr(context.prompt_config, "models_dir", None)
    max_tool_iterations = 10
    max_resume_attempts = 2

    state = _FrontendRuntimeState(
        hyperparameters=build_hyperparameters(
            args,
            loaded_hyperparameters,
            context.prompt_config,
            _use_harmony_prompt_defaults(context),
        ),
        last_saved_hyperparameters=find_last_hyperparameters(conversation) or loaded_hyperparameters.copy(),
        generation_index=0,
        last_prompt_start_time=None,
    )
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=context.prompt_config,
        profile_data=context.profile_data,
        model_path=model_path,
    )

    for user_input in prompts:
        if not user_input.strip():
            continue
        state = _process_user_input(
            user_input=user_input,
            args=args,
            context=context,
            conversation=conversation,
            state=state,
            backend_impl=backend_impl,
            models_dir=models_dir,
            max_context_tokens=max_context_tokens,
            max_tool_iterations=max_tool_iterations,
            max_resume_attempts=max_resume_attempts,
        )

        if context.chat_file_path:
            error = try_save_conversation(
                context.chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                context.prompt_config.model_dump() if context.prompt_config else None,
                context.tools,
                state.hyperparameters,
                max_context_tokens,
                chat_id,
            )
            if error:
                logger.warning(
                    "Failed to save chat: %s (check file path permissions)",
                    error,
                )
