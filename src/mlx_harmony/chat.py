from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from unicodefix.transforms import clean_text

from .config import apply_placeholders, load_prompt_config
from .generator import TokenGenerator
from .tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)


def _extract_content_from_raw_harmony(raw_text: str) -> Optional[str]:
    """
    Extract message content from raw Harmony format text when parsing fails.

    Tries to extract:
    - Analysis channel content: <|channel|>analysis<|message|>...<|end|>
    - Final channel content: <|channel|>final<|message|>...<|end|>
    - Messages without channel: <|message|>...<|end|>

    Returns the extracted content or None if extraction fails.
    """
    if not raw_text:
        return None

    # Pattern to match Harmony message format
    # Matches: <|channel|>analysis<|message|>content<|end|>
    # or: <|message|>content<|end|>
    patterns = [
        # Final channel messages (preferred)
        r'<\|channel\|>final<\|message\|>(.*?)<\|end\|>',
        # Messages without channel
        r'<\|start\|>assistant<\|message\|>(.*?)<\|end\|>',
        # Analysis channel (fallback)
        r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, raw_text, re.DOTALL)
        if matches:
            # Return the last match (most recent message)
            content = matches[-1].strip()
            if content:
                return content

    return None


def save_conversation(
    path: str | Path,
    messages: List[Dict[str, str]],
    model_path: str,
    prompt_config_path: Optional[str] = None,
    tools: Optional[List] = None,
    hyperparameters: Optional[Dict[str, Optional[float | int]]] = None,
) -> None:
    """
    Save conversation to a JSON file.

    Each message (turn) includes a timestamp when it was created.
    Assistant messages include the hyperparameters used for that generation,
    allowing tracking of hyperparameter changes during the conversation.
    Metadata.hyperparameters stores the latest hyperparameters for quick restoration.

    Format:
    {
        "metadata": {
            "model_path": "...",
            "prompt_config_path": "...",
            "tools": ["browser", "python"],
            "hyperparameters": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 512,
                ...
            },
            "created_at": "2026-01-06T...",
            "updated_at": "2026-01-06T..."
        },
        "messages": [
            {"role": "user", "content": "...", "timestamp": "2026-01-06T..."},
            {
                "role": "assistant",
                "content": "...",
                "timestamp": "2026-01-06T...",
                "hyperparameters": {"temperature": 0.8, "top_p": 0.9, ...}
            },
            {"role": "tool", "name": "browser", "content": "...", "recipient": "assistant", "timestamp": "2026-01-06T..."}
        ]
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to preserve created_at
    created_at = datetime.utcnow().isoformat() + "Z"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("metadata", {}).get("created_at", created_at)
        except Exception:
            pass  # If we can't read it, use new timestamp

    # Ensure all messages have timestamps (add if missing for backward compatibility)
    # Extract latest hyperparameters from assistant messages for metadata
    now = datetime.utcnow().isoformat() + "Z"
    messages_with_timestamps = []
    latest_hyperparameters = hyperparameters or {}

    # Find the most recent assistant message's hyperparameters as fallback
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "hyperparameters" in msg:
            latest_hyperparameters = msg["hyperparameters"]
            break

    for msg in messages:
        msg_copy = msg.copy()
        if "timestamp" not in msg_copy:
            msg_copy["timestamp"] = now
        messages_with_timestamps.append(msg_copy)

    metadata = {
        "model_path": model_path,
        "prompt_config_path": prompt_config_path,
        "tools": [t.name for t in tools] if tools else [],
        "hyperparameters": latest_hyperparameters,  # Latest hyperparameters from conversation
        "created_at": created_at,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    data = {
        "metadata": metadata,
        "messages": messages_with_timestamps,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_conversation(path: str | Path) -> tuple[List[Dict[str, str]], Dict[str, any]]:
    """
    Load conversation from a JSON file.

    Returns:
        (messages, metadata) tuple where:
        - messages: conversation history (turns) with timestamps
        - metadata: contains model, prompt_config, tools, and hyperparameters
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: output raw prompts and responses as text.",
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        default=None,
        help="Path to write debug prompts/responses when debug is enabled (default: log/prompt-debug.log).",
    )
    parser.add_argument(
        "--prewarm-cache",
        action="store_true",
        default=True,
        help="Pre-warm filesystem cache before loading model (speeds up loading, default: True).",
    )
    parser.add_argument(
        "--no-prewarm-cache",
        dest="prewarm_cache",
        action="store_false",
        help="Disable filesystem cache pre-warming.",
    )
    parser.add_argument(
        "--mlock",
        action="store_true",
        default=None,
        help="Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal only). "
        "Can also be set in prompt config JSON. Default: False",
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
    loaded_hyperparameters = {}
    if args.load_conversation:
        try:
            conversation, loaded_metadata = load_conversation(args.load_conversation)
            print(f"[INFO] Loaded conversation from: {args.load_conversation}")
            print(f"[INFO] Found {len(conversation)} previous messages (turns)")

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

            # Restore hyperparameters from conversation if not explicitly set via CLI
            loaded_hyperparameters = loaded_metadata.get("hyperparameters", {})
            if loaded_hyperparameters:
                print(f"[INFO] Loaded hyperparameters from conversation: {loaded_hyperparameters}")
        except Exception as e:
            print(f"[ERROR] Failed to load conversation: {e}")
            raise SystemExit(1)

    # Get mlock/prewarm_cache from prompt config if not explicitly set via CLI
    mlock = args.mlock
    if mlock is None and prompt_config:
        mlock = prompt_config.mlock

    prewarm_cache = args.prewarm_cache
    if prompt_config and prompt_config.prewarm_cache is not None:
        prewarm_cache = prompt_config.prewarm_cache

    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
        prewarm_cache=prewarm_cache,
        mlock=mlock or False,  # Default to False if None
    )

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

    print("[INFO] Type 'q' or `Control-D` to quit.\n")
    if args.save_conversation:
        print(f"[INFO] Conversation will be saved to: {args.save_conversation}")

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info
    if prompt_config and prompt_config.assistant_greeting and not conversation:
        greeting_text = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        # Get assistant name for greeting display
        assistant_name = "Assistant"
        if prompt_config.placeholders:
            assistant_name = prompt_config.placeholders.get("assistant", "Assistant")
        print(f"{assistant_name}: {greeting_text}")
        conversation.append(
            {
                "role": "assistant",
                "content": greeting_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    max_tool_iterations = 10  # Prevent infinite loops

    # Collect hyperparameters (CLI args take precedence over loaded values, then config, then defaults)
    # For Harmony models, use higher default max_tokens (1024) to allow for
    # both analysis and final channels, otherwise default to 512
    default_max_tokens = 1024 if (generator.is_gpt_oss and generator.use_harmony) else 512
    hyperparameters = {
        "max_tokens": (
            args.max_tokens
            if args.max_tokens is not None
            else (
                loaded_hyperparameters.get("max_tokens")
                or (prompt_config.max_tokens if prompt_config else None)
                or default_max_tokens
            )
        ),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else (
                loaded_hyperparameters.get("temperature")
                or (prompt_config.temperature if prompt_config else None)
            )
        ),
        "top_p": (
            args.top_p
            if args.top_p is not None
            else (
                loaded_hyperparameters.get("top_p")
                or (prompt_config.top_p if prompt_config else None)
            )
        ),
        "min_p": (
            args.min_p
            if args.min_p is not None
            else (
                loaded_hyperparameters.get("min_p")
                or (prompt_config.min_p if prompt_config else None)
            )
        ),
        "top_k": (
            args.top_k
            if args.top_k is not None
            else (
                loaded_hyperparameters.get("top_k")
                or (prompt_config.top_k if prompt_config else None)
            )
        ),
        "repetition_penalty": (
            args.repetition_penalty
            if args.repetition_penalty is not None
            else (
                loaded_hyperparameters.get("repetition_penalty")
                or (prompt_config.repetition_penalty if prompt_config else None)
            )
        ),
        "repetition_context_size": (
            args.repetition_context_size
            if args.repetition_context_size is not None
            else (
                loaded_hyperparameters.get("repetition_context_size")
                or (prompt_config.repetition_context_size if prompt_config else None)
            )
        ),
    }
    # Remove None values
    hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

    # Debug file setup: if --debug-file is provided, enable debug automatically
    debug_enabled = bool(args.debug or args.debug_file)
    debug_path = None
    if debug_enabled:
        debug_path = Path(args.debug_file or "log/prompt-debug.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            user_input = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if user_input.strip().lower() == "q":
            break

        # Add timestamp to user message (turn)
        user_turn = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        conversation.append(user_turn)

        # Main generation loop with tool call handling
        tool_iteration = 0
        while tool_iteration < max_tool_iterations:
            tokens: List[int] = []

            # Debug: output raw prompt (to file only, unless --debug is explicitly set)
            if debug_enabled:
                raw_prompt = generator._messages_to_prompt(conversation, None)
                if args.debug:  # Only print to console if --debug flag is set
                    print("\n[DEBUG] Raw prompt sent to LLM:")
                    print("-" * 80)
                    print(raw_prompt)
                    print("-" * 80)
                if debug_path:
                    with open(debug_path, "a", encoding="utf-8") as df:
                        df.write("\n[DEBUG] Raw prompt sent to LLM:\n")
                        df.write("-" * 80 + "\n")
                        df.write(raw_prompt + "\n")
                        df.write("-" * 80 + "\n")

            # Use hyperparameters dict (CLI args already merged with loaded values)
            # For Harmony models, we'll parse messages to extract final channel content
            # For non-Harmony models, stream tokens directly
            parsed_messages = None  # Will be set for Harmony models
            for token_id in generator.generate(
                messages=conversation,
                temperature=hyperparameters.get("temperature"),
                max_tokens=hyperparameters.get("max_tokens"),
                top_p=hyperparameters.get("top_p"),
                min_p=hyperparameters.get("min_p"),
                top_k=hyperparameters.get("top_k"),
                repetition_penalty=hyperparameters.get("repetition_penalty"),
                repetition_context_size=hyperparameters.get("repetition_context_size"),
            ):
                tokens.append(int(token_id))
                # For non-Harmony models, stream tokens directly
                if not generator.is_gpt_oss or not generator.use_harmony:
                    text = generator.tokenizer.decode([int(token_id)])
                    text = clean_text(text)  # Clean Unicode and remove replacement chars
                    print(text, end="", flush=True)

            # For Harmony models, parse messages and extract final channel content
            if generator.is_gpt_oss and generator.use_harmony:
                # Get assistant name from prompt config for display
                assistant_name = "Assistant"
                if prompt_config and prompt_config.placeholders:
                    assistant_name = prompt_config.placeholders.get("assistant", "Assistant")

                try:
                    parsed_messages = generator.parse_messages_from_tokens(tokens)
                    # Extract text from final channel messages (or messages without channel)
                    # Format analysis channel as [THINKING - ...]
                    final_text_parts = []
                    analysis_text_parts = []

                    for msg in parsed_messages:
                        channel = getattr(msg, "channel", None)
                        # Extract text content from message
                        msg_text = ""
                        for content in msg.content:
                            if hasattr(content, "text"):
                                msg_text += content.text

                        if channel == "final" or channel is None:
                            # Prefer final channel or no-channel messages
                            # Clean Unicode to normalize problematic characters
                            final_text_parts.append(clean_text(msg_text))
                        elif channel == "analysis":
                            # Collect analysis channel content
                            # Clean Unicode to normalize problematic characters
                            analysis_text_parts.append(clean_text(msg_text))

                    # Display formatted output
                    # Show analysis/thinking first if present
                    if analysis_text_parts:
                        thinking_text = " ".join(analysis_text_parts).strip()
                        # Truncate if too long to avoid spam
                        if len(thinking_text) > 2500:
                            thinking_text = thinking_text[:2500] + "... [truncated]"
                        if thinking_text:
                            print(f"\n[THINKING - {thinking_text}]\n")

                    # Show final channel message with assistant name
                    if final_text_parts:
                        assistant_text = "".join(final_text_parts).strip()
                        print(f"{assistant_name}: {assistant_text}\n")
                    elif analysis_text_parts:
                        # Only analysis channel content was found (no final channel)
                        # Don't display analysis as if it were the response - just warn
                        assistant_text = ""  # No final response
                        print(
                            "\n[WARNING] Model generated only analysis channel - "
                            "no final response. The thinking process is shown above.\n"
                            "Possible causes:\n"
                            "  - max_tokens too low (try increasing --max-tokens)\n"
                            "  - repetition_penalty may need adjustment (try lowering if too high, raising if stuck)\n"
                            "  - Model may need a nudge to transition to final channel\n"
                        )
                    else:
                        # Fallback: if no messages found, try to decode raw tokens
                        # and strip Harmony tags manually
                        raw_text = generator.tokenizer.decode(tokens)
                        raw_text = clean_text(raw_text)  # Clean Unicode and remove replacement chars
                        # Try to extract content from raw text if parsing failed
                        assistant_text = _extract_content_from_raw_harmony(raw_text)
                        if assistant_text:
                            assistant_text = clean_text(assistant_text)
                            print(f"{assistant_name}: {assistant_text}")
                        else:
                            assistant_text = raw_text  # Last resort (already cleaned)
                            print(raw_text)
                except Exception as e:
                    # Fallback: if parsing fails, decode raw tokens and try to extract content
                    raw_text = generator.tokenizer.decode(tokens)
                    raw_text = clean_text(raw_text)  # Clean Unicode and remove replacement chars
                    # Extract analysis and final channels separately
                    analysis_matches = re.findall(
                        r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>', raw_text, re.DOTALL
                    )
                    final_matches = re.findall(
                        r'<\|channel\|>final<\|message\|>(.*?)<\|end\|>', raw_text, re.DOTALL
                    )

                    # Show analysis/thinking if present
                    if analysis_matches:
                        thinking_text = " ".join(clean_text(m.strip()) for m in analysis_matches).strip()
                        # Truncate if too long
                        if len(thinking_text) > 2500:
                            thinking_text = thinking_text[:2500] + "... [truncated]"
                        if thinking_text:
                            print(f"\n[THINKING - {thinking_text}]\n")

                    # Show final channel message with assistant name
                    if final_matches:
                        assistant_text = clean_text(final_matches[-1].strip())
                        print(f"{assistant_name}: {assistant_text}\n")
                    elif analysis_matches:
                        # Only analysis channel found - don't display as response
                        assistant_text = ""  # No final response
                        thinking_text = " ".join(m.strip() for m in analysis_matches).strip()
                        # Truncate if too long
                        if len(thinking_text) > 2500:
                            thinking_text = thinking_text[:2500] + "... [truncated]"
                        if thinking_text and not analysis_text_parts:
                            # Show thinking if we haven't already
                            print(f"\n[THINKING - {thinking_text}]\n")
                        print(
                            "\n[WARNING] Model generated only analysis channel - "
                            "no final response. The thinking process is shown above.\n"
                            "Possible causes:\n"
                            "  - max_tokens too low (try increasing --max-tokens)\n"
                            "  - repetition_penalty may need adjustment (try lowering if too high, raising if stuck)\n"
                            "  - Model may need a nudge to transition to final channel\n"
                        )
                    else:
                        # Try to extract any assistant message
                        assistant_text = _extract_content_from_raw_harmony(raw_text)
                        if assistant_text:
                            print(f"{assistant_name}: {assistant_text}")
                        else:
                            # Last resort: show raw text (shouldn't happen often)
                            assistant_text = raw_text
                            print(f"\n[WARNING] Failed to parse Harmony messages: {e}")
                            print(raw_text)
            else:
                # Non-Harmony model: already printed during streaming
                print()  # Newline after streaming
                assistant_text = clean_text(generator.tokenizer.decode(tokens))

            # For GPT-OSS models with tools, check for tool calls
            # (We already parsed messages above for Harmony models, so reuse if available)
            if generator.is_gpt_oss and tools and generator.use_harmony:
                try:
                    # Reuse parsed_messages if we already parsed them above
                    if parsed_messages is None:
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
                            # Record hyperparameters used for this generation cycle
                            tool_result_msg = {
                                "role": "tool",
                                "name": tool_call.tool_name,  # Tool name goes in Author.name
                                "content": result,
                                "recipient": "assistant",  # Tool results are sent to assistant
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "hyperparameters": hyperparameters.copy()
                                if hyperparameters
                                else {},
                            }
                            conversation.append(tool_result_msg)

                        # Continue generation with tool results
                        tool_iteration += 1
                        continue
                except Exception as e:
                    print(f"\n[WARNING] Error parsing tool calls: {e}")

            # No tool calls or non-GPT-OSS model: assistant_text already set above
            # (For Harmony models, it's the final channel content; for others, it's decoded tokens)

            # Debug: output raw response (to file only, unless --debug is explicitly set)
            if debug_enabled:
                raw_response = generator.tokenizer.decode(tokens)
                # Clean Unicode for display, but keep original for file
                cleaned_response = clean_text(raw_response)
                if args.debug:  # Only print to console if --debug flag is set
                    print("\n[DEBUG] Raw response from LLM:")
                    print("-" * 80)
                    print(cleaned_response)
                    print("-" * 80)
                if debug_path:
                    with open(debug_path, "a", encoding="utf-8") as df:
                        df.write("\n[DEBUG] Raw response from LLM:\n")
                        df.write("-" * 80 + "\n")
                        df.write(raw_response + "\n")  # Keep original for debugging
                        df.write("-" * 80 + "\n")

            # Record hyperparameters used for this generation
            # Only save assistant turn if we have actual content (not just analysis)
            if assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
                # For non-Harmony models, always save; for Harmony models, only if we have final response
                assistant_turn = {
                    "role": "assistant",
                    "content": assistant_text if assistant_text else "[No final response - see thinking above]",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                }
                # For Harmony models, also record analysis channel if present
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    assistant_turn["analysis"] = " ".join(analysis_text_parts).strip()
                conversation.append(assistant_turn)
            else:
                # Harmony model with only analysis channel - save analysis separately
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    assistant_turn = {
                        "role": "assistant",
                        "content": "[Analysis only - no final response]",
                        "analysis": " ".join(analysis_text_parts).strip(),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                    }
                    conversation.append(assistant_turn)

            # Save conversation after each exchange (turn)
            if args.save_conversation:
                try:
                    save_conversation(
                        args.save_conversation,
                        conversation,
                        model_path,
                        prompt_config_path,
                        tools,
                        hyperparameters,
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
                hyperparameters,
            )
            print(f"\n[INFO] Conversation saved to: {args.save_conversation}")
        except Exception as e:
            print(f"\n[WARNING] Failed to save conversation on exit: {e}")


if __name__ == "__main__":
    main()
