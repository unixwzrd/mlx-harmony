from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.console import Console as RichConsole
from rich.markdown import Markdown

# Create a console instance for rich rendering
_console = Console()


def display_assistant(
    text: str,
    assistant_name: str,
    render_markdown: bool = True,
) -> None:
    """
    Display assistant text with consistent formatting.

    Args:
        text: Assistant response text
        assistant_name: Name to display (e.g., "Assistant", "Mia")
        render_markdown: Whether to render as markdown (default: True)
    """
    if not text:
        return

    if render_markdown:
        prefix = f"{assistant_name}: "
        print(prefix, end="")
        if _console and hasattr(_console, "size"):
            console_width = _console.size.width
        elif _console and hasattr(_console, "width"):
            console_width = _console.width
        else:
            console_width = 80
        prefix_length = len(prefix)
        available_width = max(console_width - prefix_length, 40)
        temp_console = RichConsole(width=available_width, legacy_windows=False)
        _render_markdown(text, render_markdown=True, console=temp_console)
        print()
    else:
        print(f"{assistant_name}: {text}")


def display_thinking(text: str, render_markdown: bool = True) -> None:
    """Display thinking/analysis text with [THINKING - ...] prefix, optionally rendered as markdown."""
    if not text.strip():
        return

    prefix = "[THINKING - "
    print(prefix, end="")

    if render_markdown:
        if _console and hasattr(_console, "size"):
            console_width = _console.size.width
        elif _console and hasattr(_console, "width"):
            console_width = _console.width
        else:
            console_width = 80
        prefix_length = len(prefix)
        available_width = max(console_width - prefix_length, 40)
        temp_console = RichConsole(width=available_width, legacy_windows=False)
        _render_markdown(text, render_markdown=True, console=temp_console)
        print("]")
    else:
        print(f"{text}]")

    print()


def _render_markdown(text: str, render_markdown: bool = True, console: Any | None = None) -> None:
    """
    Render text as markdown using rich (similar to glow/mdless) if enabled and rich is available.
    """
    if not text:
        return

    render_console = console if console is not None else _console

    if render_markdown and render_console is not None:
        try:
            normalized_text = text
            normalized_text = re.sub(r"(.)(#{2,6}\s)", r"\1\n\2", normalized_text)
            normalized_text = re.sub(r"(\S)(\d+\.\s+)", r"\1\n\2", normalized_text)
            markdown = Markdown(normalized_text)
            render_console.print(markdown)
        except Exception:
            print(text, end="")
    else:
        print(text, end="")
