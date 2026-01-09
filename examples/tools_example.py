#!/usr/bin/env python3
"""
Example demonstrating tool call parsing and execution.

Note: This example shows the tool infrastructure, but the actual tool
executors (browser, python, apply_patch) are currently stubs.
Real implementations would need to be added separately.
"""

from mlx_harmony import TokenGenerator
from mlx_harmony.tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)


def main():
    # This example demonstrates the tool infrastructure
    # For actual tool execution, you would need a GPT-OSS model
    # and real tool executor implementations

    print("=== Tool Infrastructure Example ===\n")

    # Get available tools for a model
    tools = get_tools_for_model(browser=True, python=True, apply_patch=False)
    print("Available tools:")
    for tool in tools:
        status = "enabled" if tool.enabled else "disabled"
        print(f"  - {tool.name}: {status}")

    # Example: Parse tool calls from messages (mock)
    print("\n=== Tool Call Parsing ===")
    print("(This would normally come from GPT-OSS model output)")

    # In a real scenario, you would parse tool calls from Harmony messages
    # like this (from the chat loop):
    """
    # After generating tokens with a GPT-OSS model:
    messages = parse_messages_from_tokens(tokens)
    tool_calls = parse_tool_calls_from_messages(messages, tools)
    
    for tool_call in tool_calls:
        result = execute_tool_call(tool_call)
        # Feed result back into conversation
        conversation.append({
            "role": "tool",
            "name": tool_call.tool_name,
            "content": result
        })
    """

    print("\nNote: Tool executors are currently stubs.")
    print("To use tools:")
    print("  1. Use a GPT-OSS model (openai/gpt-oss-*)")
    print("  2. Enable tools with --browser, --python, --apply-patch flags")
    print("  3. Implement actual tool executors in src/mlx_harmony/tools/")
    print("\nSee the tool implementation files for details:")
    print("  - src/mlx_harmony/tools/__init__.py")


if __name__ == "__main__":
    main()
