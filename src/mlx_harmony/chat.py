from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_prompt_config
from .generator import TokenGenerator
from .tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)


def save_conversation(
    path: str | Path,
    messages: List[Dict[str, str]],
    model_path: str,
    prompt_config_path: Optional[str] = None,
    tools: Optional[List] = None,
) -> None:
    """
    Save conversation to a JSON file.

    Format:
    {
        "metadata": {
            "model_path": "...",
            "prompt_config_path": "...",
            "tools": ["browser", "python"],
            "created_at": "2026-01-06T...",
            "updated_at": "2026-01-06T..."
        },
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "tool", "name": "browser", "content": "...", "recipient": "assistant"}
        ]
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to preserve created_at
    created_at = datetime.utcnow().isoformat() + "Z"
    if path.exists():
        try:
            with open(path, "r") as f:
                existing = json.load(f)
                created_at = existing.get("metadata", {}).get("created_at", created_at)
        except Exception:
            pass  # If we can't read it, use new timestamp

    metadata = {
        "model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "tools": [t.name for t in tools] if tools else [],
        "created_at": created_at,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    data = {
        "metadata": metadata,
        "messages": messages,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_conversation(path: str | Path) -> tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Load conversation from a JSON file.

    Returns:
        (messages, metadata) tuple where messages is the conversation history
        and metadata contains model/prompt_config info.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    metadata = data.get("metadata", {})

    return messages, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with any MLX-LM model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or Hugging Face repo (or set via --profile).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--python",
        dest="use_python",
        action="store_true",
        help="Enable Python tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--apply-patch",
        action="store_true",
        help="Enable apply_patch tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config/default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=None,
        help="Number of previous tokens used for repetition penalty.",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to JSON file with Harmony prompt configuration (GPT-OSS).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Named profile from a profiles JSON (see --profiles-file).",
    )
    parser.add_argument(
        "--profiles-file",
        type=str,
        default="configs/profiles.example.json",
        help="Path to profiles JSON (default: configs/profiles.example.json)",
    )
    parser.add_argument(
        "--save-conversation",
        type=str,
        default=None,
        help="Path to save conversation JSON file (auto-saves after each exchange).",
    )
    parser.add_argument(
        "--load-conversation",
        type=str,
        default=None,
        help="Path to load previous conversation JSON file (resumes chat).",
    )
    args = parser.parse_args()

    # Resolve profile/model/prompt_config
    profile_model = None
    profile_prompt_cfg = None
    if args.profile:
        from .config import load_profiles

        profiles = load_profiles(args.profiles_file)
        if args.profile not in profiles:
            raise SystemExit(
                f"Profile '{args.profile}' not found in {args.profiles_file}"
            )
        profile = profiles[args.profile]
        profile_model = profile.get("model")
        profile_prompt_cfg = profile.get("prompt_config")

    model_path = args.model or profile_model
    if not model_path:
        raise SystemExit("Model must be provided via --model or --profile")

    prompt_config_path = args.prompt_config or profile_prompt_cfg
    prompt_config = (
        load_prompt_config(prompt_config_path) if prompt_config_path else None
    )

    # Load conversation if specified
    conversation: List[Dict[str, str]] = []
    loaded_metadata = {}
    if args.load_conversation:
        try:
            conversation, loaded_metadata = load_conversation(args.load_conversation)
            print(f"[INFO] Loaded conversation from: {args.load_conversation}")
            print(f"[INFO] Found {len(conversation)} previous messages")
            
            # Optionally use loaded model/prompt_config if not explicitly set
            if not model_path and loaded_metadata.get("model_path"):
                model_path = loaded_metadata["model_path"]
                print(f"[INFO] Using model from conversation: {model_path}")
            if not prompt_config_path and loaded_metadata.get("prompt_config_path"):
                prompt_config_path = loaded_metadata["prompt_config_path"]
                prompt_config = (
                    load_prompt_config(prompt_config_path) if prompt_config_path else None
                )
                print(f"[INFO] Using prompt config from conversation: {prompt_config_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load conversation: {e}")
            raise SystemExit(1)

    generator = TokenGenerator(model_path, prompt_config=prompt_config)

    tools = []
    if generator.is_gpt_oss:
        tools = get_tools_for_model(
            browser=args.browser,
            python=args.use_python,
            apply_patch=args.apply_patch,
        )

    print(f"[INFO] Starting chat with model: {model_path}")
    if generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected – Harmony format enabled.")
        if tools:
            enabled = ", ".join(t.name for t in tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non–GPT-OSS model – using native chat template.")

    print("[INFO] Type 'q' to quit.")
    if args.save_conversation:
        print(f"[INFO] Conversation will be saved to: {args.save_conversation}")

    max_tool_iterations = 10  # Prevent infinite loops

    while True:
        user_input = input("\n>> ")
        if user_input.strip().lower() == "q":
            break

        conversation.append({"role": "user", "content": user_input})

        # Main generation loop with tool call handling
        tool_iteration = 0
        while tool_iteration < max_tool_iterations:
            tokens: List[int] = []
            for token_id in generator.generate(
                messages=conversation,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                min_p=args.min_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                repetition_context_size=args.repetition_context_size,
            ):
                tokens.append(int(token_id))
                text = generator.tokenizer.decode([int(token_id)])
                print(text, end="", flush=True)
            print()

            # For GPT-OSS models with tools, check for tool calls
            if generator.is_gpt_oss and tools and generator.use_harmony:
                try:
                    parsed_messages = generator.parse_messages_from_tokens(tokens)
                    tool_calls = parse_tool_calls_from_messages(
                        parsed_messages, tools
                    )

                    if tool_calls:
                        print(f"\n[TOOL] Detected {len(tool_calls)} tool call(s)")
                        for tool_call in tool_calls:
                            print(
                                f"[TOOL] Executing: {tool_call.tool_name} with args: {tool_call.arguments}"
                            )
                            result = execute_tool_call(tool_call)
                            print(f"[TOOL] Result: {result}")

                            # Add tool result to conversation in Harmony format
                            # Format: <|start|>{tool_name} to=assistant<|channel|>commentary<|message|>{result}<|end|>
                            # Use role="tool" with name field for proper Harmony message construction
                            tool_result_msg = {
                                "role": "tool",
                                "name": tool_call.tool_name,  # Tool name goes in Author.name
                                "content": result,
                                "recipient": "assistant",  # Tool results are sent to assistant
                            }
                            conversation.append(tool_result_msg)

                        # Continue generation with tool results
                        tool_iteration += 1
                        continue
                except Exception as e:
                    print(f"\n[WARNING] Error parsing tool calls: {e}")

            # No tool calls or non-GPT-OSS model: add assistant response and break
            assistant_text = generator.tokenizer.decode(tokens)
            conversation.append({"role": "assistant", "content": assistant_text})
            
            # Save conversation after each exchange
            if args.save_conversation:
                try:
                    save_conversation(
                        args.save_conversation,
                        conversation,
                        model_path,
                        prompt_config_path,
                        tools,
                    )
                except Exception as e:
                    print(f"\n[WARNING] Failed to save conversation: {e}")
            
            break
    
    # Final save on exit
    if args.save_conversation and conversation:
        try:
            save_conversation(
                args.save_conversation,
                conversation,
                model_path,
                prompt_config_path,
                tools,
            )
            print(f"\n[INFO] Conversation saved to: {args.save_conversation}")
        except Exception as e:
            print(f"\n[WARNING] Failed to save conversation on exit: {e}")


if __name__ == "__main__":
    main()

