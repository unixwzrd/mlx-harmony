from __future__ import annotations

from typing import Any

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


class PromptTokenCache:
    """In-memory token count cache for prompt truncation."""

    def __init__(self) -> None:
        self.message_tokens: dict[str, int] = {}
        self.message_token_ids: dict[str, list[int]] = {}
        self.total_tokens: int = 0
        self.base_tokens: int = 0
        self.message_count: int = 0
        self.system_message: str | None = None
        self.base_prompt_tokens: list[int] = []
        self.base_prefix_tokens: list[int] = []
        self.assistant_start_tokens: list[int] = []

    def reset(self) -> None:
        self.message_tokens.clear()
        self.message_token_ids.clear()
        self.total_tokens = 0
        self.base_tokens = 0
        self.message_count = 0
        self.system_message = None
        self.base_prompt_tokens = []
        self.base_prefix_tokens = []
        self.assistant_start_tokens = []

    def _common_suffix(self, first: list[int], second: list[int]) -> list[int]:
        match_count = 0
        while (
            match_count < len(first)
            and match_count < len(second)
            and first[-(match_count + 1)] == second[-(match_count + 1)]
        ):
            match_count += 1
        if match_count == 0:
            return []
        return first[len(first) - match_count :]

    def _ensure_base_tokens(
        self, *, generator: Any, system_message: str | None
    ) -> tuple[list[int], list[int]]:
        base_prompt_tokens = generator.render_prompt_tokens([], system_message)
        if base_prompt_tokens != self.base_prompt_tokens:
            dummy_message = {"role": "user", "content": "x"}
            dummy_tokens = generator.render_prompt_tokens([dummy_message], system_message)
            assistant_start_tokens = self._common_suffix(
                base_prompt_tokens, dummy_tokens
            )
            if not assistant_start_tokens:
                logger.error(
                    "Unable to derive assistant start tokens for prompt caching."
                )
                raise RuntimeError(
                    "Unable to derive assistant start tokens for prompt caching."
                )
            base_prefix_tokens = base_prompt_tokens[
                : len(base_prompt_tokens) - len(assistant_start_tokens)
            ]
            self.base_prompt_tokens = list(base_prompt_tokens)
            self.base_prefix_tokens = list(base_prefix_tokens)
            self.assistant_start_tokens = list(assistant_start_tokens)
        return self.base_prefix_tokens, self.assistant_start_tokens

    def _message_token_ids(
        self,
        *,
        message: dict[str, Any],
        generator: Any,
        system_message: str | None,
    ) -> list[int]:
        base_prefix_tokens, assistant_start_tokens = self._ensure_base_tokens(
            generator=generator, system_message=system_message
        )
        tokens_with_message = generator.render_prompt_tokens([message], system_message)
        prefix_len = len(base_prefix_tokens)
        suffix_len = len(assistant_start_tokens)
        if len(tokens_with_message) < prefix_len + suffix_len:
            logger.error("Prompt tokenization returned an unexpected token sequence.")
            raise RuntimeError("Prompt tokenization returned an unexpected token sequence.")
        if tokens_with_message[:prefix_len] != base_prefix_tokens:
            logger.error("Prompt token prefix mismatch while caching prompt tokens.")
            raise RuntimeError("Prompt token prefix mismatch while caching prompt tokens.")
        if suffix_len and tokens_with_message[-suffix_len:] != assistant_start_tokens:
            logger.error("Prompt token suffix mismatch while caching prompt tokens.")
            raise RuntimeError("Prompt token suffix mismatch while caching prompt tokens.")
        return tokens_with_message[prefix_len : len(tokens_with_message) - suffix_len]

    def rebuild(
        self,
        *,
        conversation: list[dict[str, Any]],
        generator: Any,
        system_message: str | None,
        base_tokens: int | None = None,
    ) -> int:
        """Recompute cached token counts for the full conversation."""
        base_prefix_tokens, assistant_start_tokens = self._ensure_base_tokens(
            generator=generator, system_message=system_message
        )
        base_tokens = (
            base_tokens
            if base_tokens is not None
            else len(base_prefix_tokens) + len(assistant_start_tokens)
        )

        self.message_tokens.clear()
        self.message_token_ids.clear()
        self.base_tokens = base_tokens
        self.system_message = system_message

        total_tokens = base_tokens
        for msg in conversation:
            cache_key = msg.get("cache_key")
            token_ids = self._message_token_ids(
                message=msg, generator=generator, system_message=system_message
            )
            if cache_key:
                self.message_token_ids[cache_key] = token_ids
                self.message_tokens[cache_key] = len(token_ids)
            total_tokens += len(token_ids)

        self.total_tokens = total_tokens
        self.message_count = len(conversation)
        return total_tokens

    def update_with_conversation(
        self,
        *,
        conversation: list[dict[str, Any]],
        generator: Any,
        system_message: str | None,
    ) -> int:
        """Update cached totals for the current conversation and return total tokens."""
        base_prefix_tokens, assistant_start_tokens = self._ensure_base_tokens(
            generator=generator, system_message=system_message
        )
        base_tokens = len(base_prefix_tokens) + len(assistant_start_tokens)
        if (
            self.system_message != system_message
            or base_tokens != self.base_tokens
            or len(conversation) < self.message_count
        ):
            return self.rebuild(
                conversation=conversation,
                generator=generator,
                system_message=system_message,
                base_tokens=base_tokens,
            )

        if len(conversation) == self.message_count:
            return self.total_tokens

        total_prev = self.total_tokens
        for idx in range(self.message_count, len(conversation)):
            cache_key = conversation[idx].get("cache_key")
            token_ids = self._message_token_ids(
                message=conversation[idx],
                generator=generator,
                system_message=system_message,
            )
            delta = len(token_ids)
            if cache_key:
                self.message_token_ids[cache_key] = token_ids
                self.message_tokens[cache_key] = max(delta, 0)
            total_prev += delta

        self.total_tokens = total_prev
        self.message_count = len(conversation)
        return total_prev

    def build_prompt_token_ids(
        self,
        *,
        conversation: list[dict[str, Any]],
        generator: Any,
        system_message: str | None,
    ) -> list[int]:
        """Build prompt token IDs from cached per-message tokens."""
        base_prefix_tokens, assistant_start_tokens = self._ensure_base_tokens(
            generator=generator, system_message=system_message
        )
        prompt_tokens: list[int] = list(base_prefix_tokens)
        for msg in conversation:
            cache_key = msg.get("cache_key")
            token_ids = self.message_token_ids.get(cache_key) if cache_key else None
            if token_ids is None:
                token_ids = self._message_token_ids(
                    message=msg,
                    generator=generator,
                    system_message=system_message,
                )
                if cache_key:
                    self.message_token_ids[cache_key] = token_ids
                    self.message_tokens[cache_key] = len(token_ids)
            prompt_tokens.extend(token_ids)
        prompt_tokens.extend(assistant_start_tokens)
        return prompt_tokens
