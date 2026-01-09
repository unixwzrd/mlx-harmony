#!/usr/bin/env python3
"""
Example demonstrating custom placeholders in prompts.

This example shows how to use built-in placeholders (<|DATE|>, <|TIME|>, etc.)
and custom placeholders ({key}) in prompt configurations.
"""

from mlx_harmony import TokenGenerator
from mlx_harmony.config import PromptConfig, apply_placeholders


def main():
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    # Create a prompt config with placeholders
    prompt_config = PromptConfig(
        system_model_identity="You are {assistant_name}, a helpful assistant created on <|DATE|> at <|TIMEZ|>.",
        assistant_greeting="Hello {user_name}! Today is <|DATE|> and the current time is <|TIMEZ|> (UTC: <|TIMEU|>). How can I help you?",
        placeholders={
            "assistant_name": "HarmonyBot",
            "user_name": "User",
        },
        temperature=0.7,
        max_tokens=150,
    )

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(model_path, prompt_config=prompt_config, lazy=False)

    # Show how placeholders are expanded
    print("\n=== Placeholder Expansion Examples ===\n")

    examples = [
        prompt_config.system_model_identity,
        prompt_config.assistant_greeting,
        "Current date: <|DATE|>",
        "Current time (24h): <|TIMEZ|>",
        "Current time (12h): <|TIMEA|>",
        "Current time (UTC): <|TIMEU|>",
        "User: {user_name}, Assistant: {assistant_name}",
    ]

    for example in examples:
        if example:
            expanded = apply_placeholders(example, prompt_config.placeholders)
            print(f"Original: {example}")
            print(f"Expanded: {expanded}\n")

    # Chat with the configured assistant
    print("=== Chat Example ===\n")

    if prompt_config.assistant_greeting:
        greeting = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        print(f"{greeting}\n")

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
            max_tokens=prompt_config.max_tokens or 150,
            temperature=prompt_config.temperature or 0.7,
        ):
            token_text = generator.tokenizer.decode([int(token_id)])
            print(token_text, end="", flush=True)
            tokens.append(token_id)

        print()

        assistant_text = generator.tokenizer.decode([int(t) for t in tokens])
        conversation.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
