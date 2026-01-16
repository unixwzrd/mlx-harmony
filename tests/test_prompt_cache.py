from __future__ import annotations

from typing import Any

from mlx_harmony.generation.prompt_cache import PromptTokenCache


class DummyGenerator:
    def render_prompt_tokens(
        self, messages: list[dict[str, Any]], system_message: str | None
    ) -> list[int]:
        base_tokens = [1, 2]
        assistant_start = [99]
        tokens: list[int] = list(base_tokens)
        for message in messages:
            role = message.get("role", "user")
            role_token = 10 if role == "user" else 20
            content_len = len(message.get("content", ""))
            tokens.extend([role_token, content_len])
        tokens.extend(assistant_start)
        return tokens


def test_prompt_cache_builds_prompt_tokens() -> None:
    generator = DummyGenerator()
    cache = PromptTokenCache()
    conversation = [
        {"role": "user", "content": "hi", "cache_key": "m1"},
        {"role": "assistant", "content": "ok", "cache_key": "m2"},
    ]

    total = cache.update_with_conversation(
        conversation=conversation,
        generator=generator,
        system_message=None,
    )
    expected_tokens = generator.render_prompt_tokens(conversation, None)

    assert total == len(expected_tokens)
    assert cache.build_prompt_token_ids(
        conversation=conversation,
        generator=generator,
        system_message=None,
    ) == expected_tokens
