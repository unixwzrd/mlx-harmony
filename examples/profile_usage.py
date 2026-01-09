#!/usr/bin/env python3
"""
Example using profiles to bundle models and configs.

This example demonstrates how to use profiles to easily switch between
different model/config combinations.
"""

from pathlib import Path

from mlx_harmony import TokenGenerator
from mlx_harmony.config import load_profiles, load_prompt_config


def main():
    import json

    # Load profiles from the example file
    # You can create your own profiles.json file
    profiles_file = Path(__file__).parent.parent / "configs" / "profiles.example.json"

    if not profiles_file.exists():
        print(f"Profile file not found: {profiles_file}")
        print("Creating a simple example profile...")
        # Create a simple example
        profiles_data = {
            "test-model": {
                "model": "mlx-community/Qwen1.5-0.5B-Chat-4bit",
                "prompt_config": None,
            }
        }
        print(json.dumps(profiles_data, indent=2))
        return

    profiles = load_profiles(str(profiles_file))
    print(f"Available profiles: {list(profiles.keys())}\n")

    # Use the first available profile (or a specific one)
    profile_name = list(profiles.keys())[0] if profiles else None

    if not profile_name:
        print("No profiles found in profiles file.")
        return

    profile = profiles[profile_name]
    model_path = profile.get("model")
    prompt_config_path = profile.get("prompt_config")

    print(f"Using profile: {profile_name}")
    print(f"Model: {model_path}")

    # Load prompt config if specified
    prompt_config = None
    if prompt_config_path:
        prompt_config_path = Path(__file__).parent.parent / prompt_config_path
        if prompt_config_path.exists():
            prompt_config = load_prompt_config(str(prompt_config_path))
            print(f"Prompt config: {prompt_config_path}")

    print("\nLoading model...")
    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        lazy=False,
    )

    # Simple chat loop
    print("\nStarting chat. Type 'quit' to end.\n")

    conversation = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        conversation.append({"role": "user", "content": user_input})

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

        print()

        assistant_text = generator.tokenizer.decode([int(t) for t in tokens])
        conversation.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
