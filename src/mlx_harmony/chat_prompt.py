from __future__ import annotations

from typing import Any

from mlx_harmony.chat_history import write_debug_prompt, write_debug_tokens
from mlx_harmony.logging import get_logger
from mlx_harmony.prompt_cache import PromptTokenCache

logger = get_logger(__name__)


def prepare_prompt(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
    debug_path: Any,
    debug: bool,
    debug_tokens: str | None,
    prompt_token_ids: list[int] | None = None,
) -> str:
    """Render the prompt and emit debug logs/tokens as configured."""
    raw_prompt = generator.render_prompt(conversation, system_message)
    write_debug_prompt(
        debug_path=debug_path,
        raw_prompt=raw_prompt,
        show_console=debug,
    )
    if debug_tokens in ("in", "both"):
        token_ids = (
            prompt_token_ids
            if prompt_token_ids is not None
            else generator.render_prompt_tokens(conversation, system_message)
        )
        write_debug_tokens(
            debug_path=debug_path,
            token_ids=token_ids,
            decode_tokens=generator.encoding.decode if generator.encoding else None,
            label="prompt",
            enabled=True,
        )
    return raw_prompt


def build_prompt_token_ids(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
) -> list[int]:
    """Build prompt token IDs using cached per-message tokens when available."""
    cache = getattr(generator, "prompt_token_cache", None)
    if (
        getattr(generator, "use_harmony", False)
        and getattr(generator, "encoding", None) is not None
        and isinstance(cache, PromptTokenCache)
    ):
        return cache.build_prompt_token_ids(
            conversation=conversation,
            generator=generator,
            system_message=system_message,
        )
    return generator.render_prompt_tokens(conversation, system_message)


def truncate_conversation_for_context(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
    max_context_tokens: int | None,
    max_context_tokens_margin: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return a truncated conversation and prompt token count."""
    effective_max_context = max_context_tokens
    if max_context_tokens and max_context_tokens_margin is not None:
        if max_context_tokens_margin >= max_context_tokens:
            logger.warning(
                "max_context_tokens_margin=%s >= max_context_tokens=%s; ignoring margin",
                max_context_tokens_margin,
                max_context_tokens,
            )
        else:
            effective_max_context = max_context_tokens - max_context_tokens_margin

    if not effective_max_context or effective_max_context <= 0:
        prompt_tokens = len(generator.render_prompt_tokens(conversation, system_message))
        return conversation, prompt_tokens
    if not conversation:
        prompt_tokens = len(generator.render_prompt_tokens(conversation, system_message))
        return conversation, prompt_tokens

    cache = getattr(generator, "prompt_token_cache", None)
    if isinstance(cache, PromptTokenCache):
        total_tokens = cache.update_with_conversation(
            conversation=conversation,
            generator=generator,
            system_message=system_message,
        )
        if total_tokens <= effective_max_context:
            return conversation, total_tokens
        trimmed = list(conversation)
        while trimmed and total_tokens > effective_max_context:
            drop_index: int | None = None
            for idx, msg in enumerate(trimmed):
                if msg.get("role") not in ("system", "developer"):
                    drop_index = idx
                    break
            if drop_index is None:
                break
            cache_key = trimmed[drop_index].get("cache_key")
            delta = cache.message_tokens.get(cache_key) if cache_key else None
            if delta is None:
                total_tokens = cache.rebuild(
                    conversation=trimmed,
                    generator=generator,
                    system_message=system_message,
                )
                if total_tokens <= effective_max_context:
                    return trimmed, total_tokens
                delta = cache.message_tokens.get(cache_key) if cache_key else None
                if delta is None:
                    break
            trimmed = trimmed[:drop_index] + trimmed[drop_index + 1 :]
            total_tokens -= delta

        if total_tokens > effective_max_context:
            logger.warning(
                "Prompt exceeds effective max_context_tokens=%s even after truncation",
                effective_max_context,
            )
        return trimmed, total_tokens

    trimmed = list(conversation)
    while True:
        token_ids = generator.render_prompt_tokens(trimmed, system_message)
        if len(token_ids) <= effective_max_context:
            return trimmed, len(token_ids)
        if len(trimmed) <= 1:
            logger.warning(
                "Prompt exceeds effective max_context_tokens=%s even after truncation",
                effective_max_context,
            )
            return trimmed, len(token_ids)
        # Drop the oldest non-system/developer message first.
        dropped = False
        for idx, msg in enumerate(trimmed):
            if msg.get("role") not in ("system", "developer"):
                trimmed = trimmed[idx + 1 :]
                dropped = True
                break
        if not dropped:
            return trimmed, len(token_ids)
