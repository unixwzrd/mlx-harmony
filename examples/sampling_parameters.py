#!/usr/bin/env python3
"""
Example demonstrating advanced sampling parameters.

This example shows how different sampling parameters affect text generation.
"""

from mlx_harmony import TokenGenerator


def generate_with_params(generator: TokenGenerator, prompt: str, params: dict):
    """Generate text with specific parameters."""
    print(f"\nParameters: {params}")
    print("Response: ", end="", flush=True)

    tokens = []
    for token_id in generator.generate(
        prompt=prompt,
        max_tokens=100,
        **params,
    ):
        token_text = generator.tokenizer.decode([int(token_id)])
        print(token_text, end="", flush=True)
        tokens.append(token_id)

    print(f"\n(Generated {len(tokens)} tokens)\n")
    return len(tokens)


def main():
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(model_path, lazy=False)

    prompt = "Write a creative story about a robot learning to paint:"

    print(f"\nPrompt: {prompt}\n")
    print("=" * 60)

    # Test different temperature settings
    print("\n=== Temperature (creativity control) ===")
    for temp in [0.0, 0.7, 1.5]:
        generate_with_params(generator, prompt, {"temperature": temp})

    # Test top_p (nucleus sampling)
    print("\n=== Top-p (nucleus sampling) ===")
    for top_p in [0.5, 0.9, 1.0]:
        generate_with_params(
            generator, prompt, {"temperature": 0.7, "top_p": top_p}
        )

    # Test top_k (top-k sampling)
    print("\n=== Top-k (diversity control) ===")
    for top_k in [10, 50, 0]:  # 0 = disabled
        generate_with_params(
            generator, prompt, {"temperature": 0.7, "top_k": top_k}
        )

    # Test repetition penalty
    print("\n=== Repetition Penalty ===")
    for rep_penalty in [1.0, 1.2, 1.5]:
        generate_with_params(
            generator,
            prompt,
            {"temperature": 0.7, "repetition_penalty": rep_penalty},
        )

    print("\n" + "=" * 60)
    print("\nExperiment with these parameters to find the best settings for your use case!")


if __name__ == "__main__":
    main()
