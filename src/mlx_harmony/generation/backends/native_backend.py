from __future__ import annotations

from typing import Optional

from mlx_harmony.generation.backend import ModelBackend
from mlx_harmony.prompts.native import NativePromptRenderer
from mlx_harmony.runtime.tokenizer import TokenizerProtocol


class NativeBackend(ModelBackend):
    def __init__(self, *, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._renderer = NativePromptRenderer(tokenizer=tokenizer)

    def prepare_prompt(
        self,
        *,
        prompt_tokens: list[int] | None,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        system_message: str | None,
    ) -> tuple[str | list[int], list[int] | None]:
        if prompt_tokens is not None:
            return prompt_tokens, prompt_tokens
        if messages:
            prompt_text = self._renderer.render_prompt_text(messages, system_message)
            return prompt_text, None
        if prompt is None:
            raise ValueError("prompt, prompt_tokens, or messages must be provided")
        return prompt, None

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def get_tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    def get_stop_tokens(self) -> list[int]:
        return []
