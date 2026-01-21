from __future__ import annotations

from typing import Optional


class NativePromptRenderer:
    """Render prompts using the tokenizer's native chat template."""

    def __init__(self, *, tokenizer) -> None:
        self.tokenizer = tokenizer

    def render_prompt_text(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> str:
        if system_message:
            messages = [{"role": "system", "content": system_message}, *messages]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def render_prompt_tokens(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> list[int]:
        prompt_text = self.render_prompt_text(messages, system_message)
        return self.tokenizer.encode(prompt_text)
