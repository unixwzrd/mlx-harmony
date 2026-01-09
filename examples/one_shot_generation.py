#!/usr/bin/env python3
"""
One-shot text generation example.

This example demonstrates how to generate text from a single prompt without a conversation.
"""

from mlx_harmony import TokenGenerator


def main():
    # Use a small test model
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(model_path, lazy=False)

    # Simple prompt for one-shot generation
    prompt = "Explain what machine learning is in one paragraph:"

    print(f"\nPrompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    # Generate tokens directly from a prompt string
    tokens = []
    for token_id in generator.generate(
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
    ):
        token_text = generator.tokenizer.decode([int(token_id)])
        print(token_text, end="", flush=True)
        tokens.append(token_id)

    print("\n")

    # Show stats
    print(f"\nGenerated {len(tokens)} tokens")


if __name__ == "__main__":
    main()
