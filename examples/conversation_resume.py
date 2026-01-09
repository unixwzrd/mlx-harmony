#!/usr/bin/env python3
"""
Example demonstrating conversation save and resume.

This example shows how to save conversations to JSON files and resume them later.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from mlx_harmony import TokenGenerator


def save_conversation(chat_file: Path, conversation: list, model_path: str):
    """Save a conversation to a JSON file."""
    chat_data = {
        "metadata": {
            "model_path": model_path,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "messages": conversation,
    }

    chat_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

    print(f"\nConversation saved to: {chat_file}")


def load_conversation(chat_file: Path) -> tuple[list, dict]:
    """Load a conversation from a JSON file."""
    if not chat_file.exists():
        return [], {}

    with open(chat_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    metadata = data.get("metadata", {})

    return messages, metadata


def main():
    model_path = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
    chat_file = Path(__file__).parent.parent / "logs" / "example_chat.json"

    print(f"Loading model: {model_path}")
    generator = TokenGenerator(model_path, lazy=False)

    # Try to load existing conversation
    conversation, metadata = load_conversation(chat_file)
    if conversation:
        print(f"\nResuming conversation from: {chat_file}")
        print(f"Previous model: {metadata.get('model_path', 'unknown')}")
        print(f"Previous messages: {len(conversation)} turns\n")

        # Display recent conversation history
        print("Recent conversation history:")
        for msg in conversation[-4:]:  # Show last 2 exchanges
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]  # First 100 chars
            print(f"  {role}: {content}...")
        print()
    else:
        print("\nStarting new conversation.\n")

    print("Type 'quit' to end and save, 'clear' to start fresh.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            save_conversation(chat_file, conversation, model_path)
            print("\nGoodbye!")
            break

        if user_input.lower() == "clear":
            conversation = []
            print("\nConversation cleared.\n")
            continue

        # Add user message
        conversation.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

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

        print()

        # Add assistant message
        assistant_text = generator.tokenizer.decode([int(t) for t in tokens])
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "timestamp": datetime.now(UTC).isoformat(),
                "hyperparameters": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
            }
        )

        # Auto-save periodically (every 5 exchanges)
        if len(conversation) % 10 == 0:
            save_conversation(chat_file, conversation, model_path)


if __name__ == "__main__":
    main()
