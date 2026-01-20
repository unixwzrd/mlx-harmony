from __future__ import annotations

from typing import Any

from mlx_harmony.harmony.prompt_builder import (
    build_prompt_token_ids,
    prepare_prompt,
    truncate_conversation_for_context,
)


def prepare_generation_context(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
    max_context_tokens: int | None,
    debug_path: str | None,
    debug: bool,
    debug_tokens: str | None,
) -> tuple[list[dict[str, Any]], int, list[int] | None]:
    prompt_conversation, prompt_token_count = truncate_conversation_for_context(
        generator=generator,
        conversation=conversation,
        system_message=system_message,
        max_context_tokens=max_context_tokens,
    )
    prompt_token_ids = build_prompt_token_ids(
        generator=generator,
        conversation=prompt_conversation,
        system_message=system_message,
    )
    prepare_prompt(
        generator=generator,
        conversation=prompt_conversation,
        system_message=system_message,
        debug_path=debug_path,
        debug=debug,
        debug_tokens=debug_tokens,
        prompt_token_ids=prompt_token_ids,
    )
    return prompt_conversation, prompt_token_count, prompt_token_ids
