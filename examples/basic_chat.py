#!/usr/bin/env python3
"""
Basic chat example using mlx-harmony.

This example demonstrates how to use the TokenGenerator for simple chat interactions.
"""

from mlx_harmony import TokenGenerator


def main():
    # Use a small test model for quick responses
    # For GPT-OSS models, use: "openai/gpt-oss-20b"
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(model_path, lazy=False)

    print("\nStarting chat. Type 'quit' or 'exit' to end.\n")

    conversation = []

    while True:
        # Get user input
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})

        # Generate response
        print("Assistant: ", end="", flush=True)
        tokens = []
        for token_id in generator.generate(
            messages=conversation,
            max_tokens=100,
            temperature=0.7,
        ):
            token_text = generator.tokenizer.decode([int(token_id)])
            print(token_text, end="", flush=True)
            tokens.append(token_id)

        print()  # New line after response

        # Add assistant response to conversation
        assistant_text = generator.tokenizer.decode([int(t) for t in tokens])
        conversation.append({"role": "assistant", "content": assistant_text})

        # Keep conversation from growing too long (simple truncation)
        if len(conversation) > 10:
            conversation = conversation[-10:]


if __name__ == "__main__":
    main()
