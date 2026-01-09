#!/usr/bin/env python3
"""
Example client for the mlx-harmony HTTP API server.

This example demonstrates how to interact with the HTTP API server.
Start the server first:
    mlx-harmony-server --model mlx-community/Qwen1.5-0.5B-Chat-4bit

Then run this script to test the API.
"""

import json
import sys

try:
    import httpx
except ImportError:
    print("This example requires 'httpx'. Install it with:")
    print("  pip install httpx")
    sys.exit(1)


def chat_completion(
    client: httpx.Client,
    messages: list,
    model: str = "test-model",
    stream: bool = False,
    **kwargs,
) -> dict:
    """Send a chat completion request to the server."""
    url = "http://localhost:8000/v1/chat/completions"

    data = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }

    response = client.post(url, json=data)
    response.raise_for_status()

    if stream:
        # Handle streaming response
        full_content = ""
        for line in response.iter_lines():
            if line.startswith("data: "):
                content = line[6:]  # Remove "data: " prefix
                if content == "[DONE]":
                    break
                try:
                    chunk = json.loads(content)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        print(text, end="", flush=True)
                        full_content += text
                except json.JSONDecodeError:
                    continue
        print()  # New line after stream
        return {"content": full_content}
    else:
        # Handle non-streaming response
        return response.json()


def main():
    base_url = "http://localhost:8000"

    # Test server connection
    print("Testing server connection...")
    try:
        with httpx.Client(timeout=30.0) as client:
            # Try a simple request
            messages = [{"role": "user", "content": "Hello!"}]
            response = chat_completion(
                client,
                messages,
                model="test-model",
                max_tokens=50,
                temperature=0.7,
            )

            print("\n=== Non-streaming Response ===")
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            print(f"Role: {message.get('role')}")
            print(f"Content: {message.get('content')}")

            print("\n=== Streaming Response ===")
            messages = [{"role": "user", "content": "Count to 5:"}]
            print("Assistant: ", end="", flush=True)
            chat_completion(
                client,
                messages,
                model="test-model",
                stream=True,
                max_tokens=50,
                temperature=0.0,  # Deterministic
            )

            print("\n=== Multi-turn Conversation ===")
            conversation = [
                {"role": "user", "content": "What is 2+2?"},
            ]
            response1 = chat_completion(
                client,
                conversation,
                model="test-model",
                max_tokens=30,
            )
            assistant_msg1 = response1["choices"][0]["message"]
            print(f"User: {conversation[0]['content']}")
            print(f"Assistant: {assistant_msg1['content']}")

            conversation.append(assistant_msg1)
            conversation.append({"role": "user", "content": "What about 3+3?"})

            response2 = chat_completion(
                client,
                conversation,
                model="test-model",
                max_tokens=30,
            )
            assistant_msg2 = response2["choices"][0]["message"]
            print(f"User: {conversation[-1]['content']}")
            print(f"Assistant: {assistant_msg2['content']}")

    except httpx.ConnectError:
        print(f"\nError: Could not connect to server at {base_url}")
        print("\nMake sure the server is running:")
        print("  mlx-harmony-server --model mlx-community/Qwen1.5-0.5B-Chat-4bit")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
