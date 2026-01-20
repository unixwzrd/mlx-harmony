from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_harmony.chat.voice import init_moshi_components
from mlx_harmony.cli.chat_commands import (
    build_hyperparameters,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
    resolve_profile_and_prompt_config,
    truncate_text,
)
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
from mlx_harmony.conversation.conversation_io import load_conversation
from mlx_harmony.generation.generator import TokenGenerator
from mlx_harmony.generation.prompt_cache import PromptTokenCache
from mlx_harmony.logging import get_logger
from mlx_harmony.render_output import display_assistant, display_thinking
from mlx_harmony.tools import get_tools_for_model

logger = get_logger(__name__)


@dataclass(slots=True)
class ChatSessionState:
    conversation: list[dict[str, Any]]
    model_path: str
    prompt_config_path: str | None
    prompt_config: Any | None
    chat_file_path: str | None
    generator: TokenGenerator
    tools: list[Any]
    assistant_name: str
    thinking_limit: int
    response_limit: int
    render_markdown: bool
    debug_path: str | None
    hyperparameters: dict[str, float | int | bool]
    last_saved_hyperparameters: dict[str, float | int | bool]
    max_context_tokens: int | None
    chat_id: str | None
    moshi_config: MoshiConfig | None
    moshi_stt: Any | None
    moshi_tts: Any | None
    smoke_ran: bool


def initialize_chat_session(args: Any) -> ChatSessionState:
    model_path, prompt_config_path, prompt_config, profile_data = resolve_profile_and_prompt_config(
        args,
        load_profiles,
        load_prompt_config,
    )

    chats_dir, logs_dir = resolve_dirs_from_config(prompt_config)

    chat_file_path, load_file_path, _chat_name, _chat_input_path = resolve_chat_paths(
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
        mlock=mlock or False,
        no_fs_cache=no_fs_cache,
    )
    if generator.use_harmony and generator.encoding:
        generator.prompt_token_cache = PromptTokenCache()

    tools: list[Any] = []
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

    debug_path = resolve_debug_path(args.debug_file, logs_dir)
    debug_tokens_mode = args.debug_tokens or "off"
    logger.info("Debug log: %s (tokens: %s)", debug_path, debug_tokens_mode)

    assistant_name = get_assistant_name(prompt_config)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
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

    if not has_conversation_history and prompt_config and prompt_config.assistant_greeting:
        greeting_text = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
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

    return ChatSessionState(
        conversation=conversation,
        model_path=model_path,
        prompt_config_path=prompt_config_path,
        prompt_config=prompt_config,
        chat_file_path=chat_file_path,
        generator=generator,
        tools=tools,
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
        debug_path=debug_path,
        hyperparameters=hyperparameters,
        last_saved_hyperparameters=last_saved_hyperparameters,
        max_context_tokens=max_context_tokens,
        chat_id=chat_id,
        moshi_config=moshi_config,
        moshi_stt=moshi_stt,
        moshi_tts=moshi_tts,
        smoke_ran=smoke_ran,
    )
