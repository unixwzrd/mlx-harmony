from __future__ import annotations

from mlx_harmony.generation.backend import ModelBackend
from mlx_harmony.prompts.harmony import HarmonyPromptRenderer
from mlx_harmony.runtime.tokenizer import TokenizerProtocol


class GPTOSSBackend(ModelBackend):
    def __init__(self, *, encoding, tokenizer, prompt_config) -> None:
        self._encoding = encoding
        self._tokenizer = tokenizer
        self._renderer = HarmonyPromptRenderer(
            encoding=encoding,
            prompt_config=prompt_config,
        )

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
            prompt_token_ids = self._renderer.render_prompt_tokens(
                messages,
                system_message,
            )
            return prompt_token_ids, prompt_token_ids
        if prompt is None:
            raise ValueError("prompt, prompt_tokens, or messages must be provided")
        return prompt, None

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)

    def get_tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    def get_stop_tokens(self) -> list[int]:
        return list(self._encoding.stop_tokens())
