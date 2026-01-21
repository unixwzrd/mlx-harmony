from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from mlx_harmony.chat_cli import build_parser
from mlx_harmony.chat_history import (
    load_chat_session,
    resolve_chat_paths,
    resolve_debug_path,
    resolve_dirs_from_config,
)
from mlx_harmony.chat_io import load_conversation
from mlx_harmony.chat_utils import (
    get_assistant_name,
    get_truncate_limits,
    resolve_profile_and_prompt_config,
)
from mlx_harmony.config import load_profiles, load_prompt_config
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.logging import get_logger
from mlx_harmony.prompt_cache import PromptTokenCache
from mlx_harmony.runtime.context import RunContext
from mlx_harmony.tools import get_tools_for_model

logger = get_logger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    args: argparse.Namespace
    conversation: list[dict[str, Any]]
    model_path: str
    prompt_config_path: str | None
    context: RunContext
    loaded_hyperparameters: dict[str, Any]
    loaded_max_context_tokens: int | None
    loaded_model_path: str | None
    loaded_chat_id: str | None


def bootstrap_chat() -> BootstrapResult:
    parser = build_parser()
    args = parser.parse_args()

    model_path, prompt_config_path, prompt_config, profile_data = resolve_profile_and_prompt_config(
        args,
        load_profiles,
        load_prompt_config,
    )

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

    mlock = args.mlock
    if mlock is None and prompt_config:
        mlock = prompt_config.mlock
    lazy = args.lazy if args.lazy is not None else False

    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        lazy=lazy,
        mlock=mlock or False,
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

    debug_path = resolve_debug_path(args.debug_file, logs_dir)
    debug_tokens_mode = args.debug_tokens or "off"
    logger.info("Debug log: %s (tokens: %s)", debug_path, debug_tokens_mode)

    assistant_name = get_assistant_name(prompt_config)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
    render_markdown = not args.no_markdown if hasattr(args, "no_markdown") else True

    context = RunContext(
        generator=generator,
        tools=tools,
        prompt_config=prompt_config,
        profile_data=profile_data,
        chats_dir=chats_dir,
        logs_dir=logs_dir,
        chat_file_path=chat_file_path,
        chat_input_path=chat_input_path,
        debug_path=debug_path,
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
    )

    return BootstrapResult(
        args=args,
        conversation=conversation,
        model_path=model_path,
        prompt_config_path=prompt_config_path,
        context=context,
        loaded_hyperparameters=loaded_hyperparameters,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        loaded_chat_id=loaded_chat_id,
    )
