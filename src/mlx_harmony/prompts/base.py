from __future__ import annotations

from typing import Optional, Protocol


class PromptRenderer(Protocol):
    """Interface for rendering prompts for different model formats."""

    def render_prompt_text(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> str:
        """Render a prompt as text (for tokenizers that encode text)."""
        ...

    def render_prompt_tokens(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> list[int]:
        """Render a prompt directly as token IDs."""
        ...
