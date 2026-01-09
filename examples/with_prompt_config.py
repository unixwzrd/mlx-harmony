#!/usr/bin/env python3
"""
Example using prompt configs for customization.

This example demonstrates how to use a prompt config JSON file to customize
system instructions, placeholders, and sampling parameters.
"""

import json
from pathlib import Path

from mlx_harmony import TokenGenerator
from mlx_harmony.config import PromptConfig, load_prompt_config


def main():
    # Use a small test model
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    # Create a simple prompt config programmatically
    # (You can also load from a JSON file using load_prompt_config)
    prompt_config = PromptConfig(
        system_model_identity="You are a helpful coding assistant.",
        assistant_greeting="Hello! I'm ready to help with your coding questions.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
        placeholders={
            "assistant": "CodeBot",
            "user": "Developer",
        },
    )

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        lazy=False,
    )

    # Show assistant greeting if configured
    if prompt_config.assistant_greeting:
        from mlx_harmony.config import apply_placeholders

        greeting = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        print(f"\n{greeting}\n")

    # Chat with the configured assistant
    print("Starting chat. Type 'quit' to end.\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\nGoodbye!")
            break

        conversation.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        tokens = []
        for token_id in generator.generate(
            messages=conversation,
            # Parameters from prompt_config are used as defaults
            # You can override them here if needed
            max_tokens=prompt_config.max_tokens or 200,
            temperature=prompt_config.temperature or 0.7,
            top_p=prompt_config.top_p or 0.9,
        ):
            token_text = generator.tokenizer.decode([int(token_id)])
            print(token_text, end="", flush=True)
            tokens.append(token_id)

        print()

        assistant_text = generator.tokenizer.decode([int(t) for t in tokens])
        conversation.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
