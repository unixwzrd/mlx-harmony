from __future__ import annotations

"""Shared model initialization helpers for CLI and server backends."""

from mlx_harmony.config import PromptConfig
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.prompt_cache import PromptTokenCache


def initialize_generator(
    *,
    model_path: str,
    prompt_config: PromptConfig | None,
    prompt_config_path: str | None,
    lazy: bool,
    mlock: bool,
) -> TokenGenerator:
    """Create and configure a TokenGenerator with shared defaults.

    This mirrors the CLI bootstrap behavior so the server and CLI share the
    same model initialization path (including Harmony prompt cache setup).

    Args:
        model_path: Path to the model directory.
        prompt_config: Prompt config used to set defaults and placeholders.
        prompt_config_path: Prompt config path used for parity logging.
        lazy: Whether to defer model weight loading.
        mlock: Whether to lock model weights in memory.

    Returns:
        Configured TokenGenerator instance.
    """
    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        lazy=lazy,
        mlock=mlock,
    )
    generator.prompt_config_path = prompt_config_path
    if generator.use_harmony and generator.encoding:
        generator.prompt_token_cache = PromptTokenCache()
    return generator
