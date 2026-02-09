"""Tests for thin HTTP client rendering and response parsing."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.client import _extract_message_fields, _print_response


def _capture_output(func) -> tuple[str, str]:
    """Capture stdout/stderr from a callable."""
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        func()
    return stdout.getvalue(), stderr.getvalue()


def _non_empty_lines(text: str) -> list[str]:
    """Normalize captured output for semantic comparisons."""
    return [line for line in (line.strip() for line in text.splitlines()) if line]


def test_extract_message_fields_returns_analysis_and_content() -> None:
    """Extract analysis/content from OpenAI-style response payload."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "analysis": "thinking text",
                    "content": "final text",
                }
            }
        ]
    }
    analysis, content = _extract_message_fields(body)
    assert analysis == "thinking text"
    assert content == "final text"


def test_extract_message_fields_handles_missing_or_invalid_message() -> None:
    """Return empty strings when choices/message shape is invalid."""
    assert _extract_message_fields({}) == ("", "")
    assert _extract_message_fields({"choices": [{"message": "invalid"}]}) == ("", "")


def test_print_response_renders_analysis_and_content() -> None:
    """Render analysis and assistant content with expected labels."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "analysis": "analysis output",
                    "content": "assistant output",
                }
            }
        ]
    }
    stdout, stderr = _capture_output(lambda: _print_response(body))
    assert stderr == ""
    lines = _non_empty_lines(stdout)
    assert lines == ["[THINKING - analysis output]", "Assistant: assistant output"]


def test_client_output_equivalence_with_cli_plain_rendering() -> None:
    """Match client and CLI output labels/order for analysis+final text."""
    analysis = "analysis text"
    content = "assistant text"
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "analysis": analysis,
                    "content": content,
                }
            }
        ]
    }
    client_stdout, _ = _capture_output(lambda: _print_response(body))

    def _render_cli_plain() -> None:
        display_thinking(analysis, render_markdown=False)
        display_assistant(content, "Assistant", render_markdown=False)
        print()

    cli_stdout, _ = _capture_output(_render_cli_plain)
    assert _non_empty_lines(client_stdout) == _non_empty_lines(cli_stdout)


def test_client_output_equivalence_final_only_with_cli_plain_rendering() -> None:
    """Match client and CLI output labels/order for final-only responses."""
    content = "assistant final only"
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }
    client_stdout, client_stderr = _capture_output(lambda: _print_response(body))
    assert client_stderr == ""

    def _render_cli_plain() -> None:
        display_assistant(content, "Assistant", render_markdown=False)
        print()

    cli_stdout, cli_stderr = _capture_output(_render_cli_plain)
    assert cli_stderr == ""
    assert _non_empty_lines(client_stdout) == _non_empty_lines(cli_stdout)


def test_client_output_equivalence_analysis_only_with_cli_plain_rendering() -> None:
    """Match client and CLI output labels/order for analysis-only responses."""
    analysis = "analysis-only text"
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "analysis": analysis,
                    "content": "",
                }
            }
        ]
    }
    client_stdout, client_stderr = _capture_output(lambda: _print_response(body))
    assert client_stderr == ""

    def _render_cli_plain() -> None:
        display_thinking(analysis, render_markdown=False)

    cli_stdout, cli_stderr = _capture_output(_render_cli_plain)
    assert cli_stderr == ""
    assert _non_empty_lines(client_stdout) == _non_empty_lines(cli_stdout)


def test_print_response_emits_error_when_choices_missing() -> None:
    """Emit an explicit stderr error when choices are missing."""
    stdout, stderr = _capture_output(lambda: _print_response({}))
    assert stdout == ""
    assert "Invalid response: missing choices" in stderr
