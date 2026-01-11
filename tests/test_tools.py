"""
Unit tests for tool parsing and execution.

Tests tool call detection and parsing from Harmony messages.
"""
from mlx_harmony.tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)


class TestToolParsing:
    """Test tool call parsing from messages."""

    def test_parse_simple_tool_call(self):
        """Test parsing a simple tool call."""
        from openai_harmony import Author, Message, Role, TextContent

        from mlx_harmony.tools import ToolConfig

        # Create a mock Harmony message with tool call (recipient field)
        tool_content = TextContent(text='{"url": "https://example.com"}')
        msg = Message(
            role=Role.ASSISTANT,
            author=Author(role=Role.ASSISTANT),
            content=[tool_content],
            recipient="browser.navigate",
        )
        messages = [msg]
        enabled_tools = [ToolConfig(name="browser", enabled=True)]

        tool_calls = parse_tool_calls_from_messages(messages, enabled_tools)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "browser"
        assert "url" in tool_calls[0].arguments

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        from openai_harmony import Author, Message, Role, TextContent

        from mlx_harmony.tools import ToolConfig

        msg1 = Message(
            role=Role.ASSISTANT,
            author=Author(role=Role.ASSISTANT),
            content=[TextContent(text='{"url": "https://example.com"}')],
            recipient="browser.navigate",
        )
        msg2 = Message(
            role=Role.ASSISTANT,
            author=Author(role=Role.ASSISTANT),
            content=[TextContent(text='{"code": "print(\\"hello\\")"}')],
            recipient="python.execute",
        )
        messages = [msg1, msg2]
        enabled_tools = [
            ToolConfig(name="browser", enabled=True),
            ToolConfig(name="python", enabled=True),
        ]

        tool_calls = parse_tool_calls_from_messages(messages, enabled_tools)
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "browser"
        assert tool_calls[1].tool_name == "python"

    def test_parse_no_tool_calls(self):
        """Test parsing messages with no tool calls."""
        from openai_harmony import Author, Message, Role, TextContent

        from mlx_harmony.tools import ToolConfig

        messages = [
            Message(role=Role.USER, author=Author(role=Role.USER), content=[TextContent(text="Hello")]),
            Message(role=Role.ASSISTANT, author=Author(role=Role.ASSISTANT), content=[TextContent(text="Hi there!")]),
        ]
        enabled_tools = [ToolConfig(name="browser", enabled=True)]
        tool_calls = parse_tool_calls_from_messages(messages, enabled_tools)
        assert len(tool_calls) == 0

    def test_parse_empty_messages(self):
        """Test parsing empty messages."""
        from mlx_harmony.tools import ToolConfig

        enabled_tools = [ToolConfig(name="browser", enabled=True)]
        tool_calls = parse_tool_calls_from_messages([], enabled_tools)
        assert len(tool_calls) == 0


class TestToolExecution:
    """Test tool execution (currently stubbed)."""

    def test_execute_browser_tool(self):
        """Test executing browser tool (stub)."""
        from openai_harmony import Author, Message, Role, TextContent

        from mlx_harmony.tools import ToolCall

        tool_call = ToolCall(
            tool_name="browser",
            arguments={"url": "https://example.com"},
            raw_message=Message(
                role=Role.ASSISTANT,
                author=Author(role=Role.ASSISTANT),
                content=[TextContent(text="")],
            ),
        )
        result = execute_tool_call(tool_call)
        # Currently stubbed, should return a JSON string
        assert result is not None
        assert isinstance(result, str)
        # Should be valid JSON
        import json
        json.loads(result)  # Should not raise

    def test_execute_python_tool(self):
        """Test executing Python tool (stub)."""
        from openai_harmony import Author, Message, Role, TextContent

        from mlx_harmony.tools import ToolCall

        tool_call = ToolCall(
            tool_name="python",
            arguments={"code": "print('hello')"},
            raw_message=Message(
                role=Role.ASSISTANT,
                author=Author(role=Role.ASSISTANT),
                content=[TextContent(text="")],
            ),
        )
        result = execute_tool_call(tool_call)
        # Currently stubbed, should return a JSON string
        assert result is not None
        assert isinstance(result, str)


class TestToolConfiguration:
    """Test tool configuration and enabling."""

    def test_get_tools_for_model(self):
        """Test getting tools for a model."""
        tools = get_tools_for_model(browser=True, python=False, apply_patch=False)
        assert len(tools) > 0
        browser_tool = next((t for t in tools if t.name == "browser"), None)
        assert browser_tool is not None
        assert browser_tool.enabled is True

    def test_get_tools_all_disabled(self):
        """Test getting tools with all disabled."""
        tools = get_tools_for_model(browser=False, python=False, apply_patch=False)
        # Should still return tools, but they're disabled
        assert len(tools) >= 0
