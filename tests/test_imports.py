"""
Basic import tests to verify all modules can be imported.
"""


def test_import_token_generator():
    """Basic import smoke test; does not load any real model."""
    from mlx_harmony import TokenGenerator  # noqa: F401


def test_import_config():
    """Test importing config module."""
    from mlx_harmony.config import (  # noqa: F401
        PromptConfig,
        apply_placeholders,
        load_prompt_config,
    )


def test_import_chat():
    """Test importing chat module."""
    from mlx_harmony.chat import (  # noqa: F401
        load_conversation,
        save_conversation,
    )


def test_import_tools():
    """Test importing tools module."""
    from mlx_harmony.tools import (  # noqa: F401
        get_tools_for_model,
        parse_tool_calls_from_messages,
    )
