from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Literal, TypedDict

from openai_harmony import Role, StreamableParser
from unicodefix.transforms import clean_text

from mlx_harmony.chat_generation import stream_generation
from mlx_harmony.chat_harmony import parse_harmony_response
from mlx_harmony.chat_history import (
    display_resume_message,
    load_chat_session,
    make_timestamp,
    resolve_chat_paths,
    resolve_debug_path,
    resolve_dirs_from_config,
    write_debug_prompt,
    write_debug_response,
    write_debug_tokens,
)
from mlx_harmony.chat_io import load_conversation, save_conversation
from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.chat_utils import (
    build_hyperparameters,
    get_assistant_name,
    get_truncate_limits,
    parse_command,
    resolve_profile_and_prompt_config,
    truncate_text,
)
from mlx_harmony.config import apply_placeholders, load_profiles, load_prompt_config
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.tools import (
    execute_tool_call,
    get_tools_for_model,
    parse_tool_calls_from_messages,
)


# Type definitions for message records
class MessageDict(TypedDict, total=False):
    """Type definition for conversation messages."""
    role: Literal["user", "assistant", "tool", "system", "developer"]
    content: str
    timestamp: str
    name: str  # Optional: for tool messages
    recipient: str  # Optional: for tool messages
    channel: str  # Optional: for Harmony messages (e.g., "commentary", "analysis", "final")
    analysis: str  # Optional: for assistant messages with thinking/analysis
    hyperparameters: dict[str, float | int]  # Optional: hyperparameters used for generation


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
        "--chat",
        type=str,
        default=None,
        help="Chat name (loads from chats_dir/<name>.json if exists, otherwise creates new chat).",
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
        help="Path to write debug prompts/responses when debug is enabled (default: logs_dir/prompt-debug.log).",
    )
    parser.add_argument(
        "--debug-tokens",
        nargs="?",
        const="out",
        choices=["in", "out", "both"],
        default=None,
        help="Write token IDs and decoded tokens to the debug log. "
        "Use 'in' for prompt tokens, 'out' for response tokens, or 'both'. "
        "If set with no value, defaults to 'out'.",
    )
    parser.add_argument(
        "--mlock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal only). "
        "Can also be set in prompt config JSON. Use --mlock to enable or --no-mlock to disable. Default: None (use config or False)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        default=False,
        help="Disable markdown rendering for assistant responses (display as plain text).",
    )
    args = parser.parse_args()

    model_path, prompt_config_path, prompt_config = resolve_profile_and_prompt_config(
        args,
        load_profiles,
        load_prompt_config,
    )

    # Resolve directories from config (defaults to "logs" if not specified)
    # These will be re-resolved after loading chat if prompt_config changes
    chats_dir, logs_dir = resolve_dirs_from_config(prompt_config)

    chat_file_path, load_file_path, chat_name, chat_input_path = resolve_chat_paths(
        args.chat, chats_dir
    )
    if load_file_path and chat_file_path and load_file_path != chat_file_path:
        print(f"[INFO] Found chat file at: {load_file_path} (will save to: {chat_file_path})")

    (
        conversation,
        model_path,
        prompt_config_path,
        prompt_config,
        updated_chats_dir,
        updated_logs_dir,
        chat_file_path,
        loaded_hyperparameters,
    ) = load_chat_session(
        load_file_path=load_file_path,
        chat_file_path=chat_file_path,
        chat_arg=args.chat,
        model_path=model_path,
        prompt_config_path=prompt_config_path,
        prompt_config=prompt_config,
        load_conversation=load_conversation,
        load_prompt_config=load_prompt_config,
        resolve_dirs=resolve_dirs_from_config,
    )
    if updated_chats_dir is not None and updated_logs_dir is not None:
        chats_dir = updated_chats_dir
        logs_dir = updated_logs_dir

    # Get mlock from prompt config if not explicitly set via CLI
    mlock = args.mlock
    if mlock is None and prompt_config:
        mlock = prompt_config.mlock

    generator = TokenGenerator(
        model_path,
        prompt_config=prompt_config,
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
        print("[INFO] GPT-OSS model detected - Harmony format enabled.")
        if tools:
            enabled = ", ".join(t.name for t in tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non-GPT-OSS model - using native chat template.")

    print("[INFO] Type 'q' or `Control-D` to quit.")
    print("[INFO] Type '\\help' to list all out-of-band commands.")
    print("[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7').")
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if chat_file_path:
        print(f"[INFO] Chat will be saved to: {chat_file_path}\n")

    # Get assistant name for display (need this before resume display)
    assistant_name = get_assistant_name(prompt_config)

    # Get truncate limits once (used multiple times)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)

    # Get render_markdown setting once (used multiple times)
    render_markdown = not args.no_markdown if hasattr(args, "no_markdown") else True

    has_conversation_history = display_resume_message(
        conversation,
        assistant_name,
        thinking_limit,
        response_limit,
        render_markdown,
        display_assistant,
        display_thinking,
        truncate_text,
    )

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info (or after resume message)
    # Only show greeting if we don't have conversation history
    if not has_conversation_history and prompt_config and prompt_config.assistant_greeting:
        greeting_text = apply_placeholders(
            prompt_config.assistant_greeting, prompt_config.placeholders
        )
        # Use markdown rendering for greeting if enabled
        display_assistant(greeting_text, assistant_name, render_markdown)
        conversation.append(
            {
                "role": "assistant",
                "content": greeting_text,
                "timestamp": make_timestamp(),
            }
        )

    max_tool_iterations = 10  # Prevent infinite loops

    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters,
        prompt_config,
        generator.is_gpt_oss and generator.use_harmony,
    )

    # Debug file setup: always write debug log by default
    # --debug enables console output, --debug-file overrides the file path
    # Path resolution: absolute paths used as-is, relative paths resolved relative to logs_dir
    # This matches --chat behavior (relative paths resolved relative to chats_dir)
    # Always write to debug file by default (unless explicitly disabled in future)
    debug_path = resolve_debug_path(args.debug_file, logs_dir)

    while True:
        try:
            user_input = input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if user_input.strip().lower() == "q":
            break

        handled, should_apply, message, updates = parse_command(user_input, hyperparameters)
        if handled:
            if message:
                print(message)
            if should_apply and updates:
                hyperparameters.update(updates)
                if chat_file_path and conversation:
                    try:
                        save_conversation(
                            chat_file_path,
                            conversation,
                            model_path,
                            prompt_config_path,
                            tools,
                            hyperparameters,
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed to save updated hyperparameters: {e}")
            continue

        # Add timestamp to user message (turn)
        user_turn = {
            "role": "user",
            "content": user_input,
            "timestamp": make_timestamp(),
        }
        conversation.append(user_turn)

        # Main generation loop with tool call handling
        tool_iteration = 0
        while tool_iteration < max_tool_iterations:
            tokens: list[int] = []

            # system_message parameter is for CLI override (if we add --system flag later)
            # The generator's render_prompt() already handles system_model_identity from prompt_config
            # So we pass None here and let the generator handle the fallback
            system_message = None  # Could be overridden by CLI --system flag in the future

            # Debug: always write raw prompt to file, print to console only if --debug is set
            raw_prompt = generator.render_prompt(conversation, system_message)
            write_debug_prompt(
                debug_path=debug_path,
                raw_prompt=raw_prompt,
                show_console=args.debug,
            )
            if args.debug_tokens in ("in", "both"):
                prompt_token_ids = generator.render_prompt_tokens(
                    conversation, system_message
                )
                write_debug_tokens(
                    debug_path=debug_path,
                    token_ids=prompt_token_ids,
                    decode_tokens=generator.encoding.decode if generator.encoding else None,
                    label="prompt",
                    enabled=True,
                )

            # Use hyperparameters dict (CLI args already merged with loaded values)
            # For Harmony models, we'll parse messages to extract final channel content
            # For Harmony models, use StreamableParser for incremental parsing
            # For non-Harmony models, stream tokens directly
            generation_start_time = time.perf_counter()
            # Initialize these early to avoid scope issues (used outside Harmony branch)
            parsed_messages: Any | None = None  # Will be set for Harmony models
            analysis_text_parts: list[str] = []
            assistant_text = ""
            # Accumulate streamed text for non-Harmony models (avoid decoding twice)
            streamed_text_parts: list[str] = []

            # For Harmony models, reset StreamableParser for this generation
            if generator.is_gpt_oss and generator.use_harmony and generator.encoding:
                # Create a fresh parser for this generation (reset state)
                generator.streamable_parser = StreamableParser(generator.encoding, Role.ASSISTANT, strict=False)

            tokens, all_generated_tokens, streamed_text_parts = stream_generation(
                generator=generator,
                conversation=conversation,
                system_message=system_message,
                hyperparameters=hyperparameters,
                on_text=lambda text: print(text, end="", flush=True),
            )

            # After generation, keep model parameters active to prevent swapping
            # This ensures buffers stay wired and don't get swapped out
            # Use generator's keepalive() method (unified behavior)
            generator.keepalive()

            generation_end_time = time.perf_counter()
            generation_elapsed = generation_end_time - generation_start_time
            num_generated_tokens = len(tokens)
            tokens_per_second = num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0

            # Display generation stats
            print(f"\n[INFO] Generated {num_generated_tokens} tokens in {generation_elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

            # Write all generated tokens to debug file (once, not duplicated)
            write_debug_tokens(
                debug_path=debug_path,
                token_ids=all_generated_tokens,
                decode_tokens=generator.encoding.decode if generator.encoding else None,
                label="response",
                enabled=args.debug_tokens in ("out", "both"),
            )

            if generator.is_gpt_oss and generator.use_harmony:
                parse_result = parse_harmony_response(
                    generator=generator,
                    tokens=tokens,
                    streamed_text_parts=streamed_text_parts,
                    assistant_name=assistant_name,
                    thinking_limit=thinking_limit,
                    response_limit=response_limit,
                    render_markdown=render_markdown,
                    debug=args.debug,
                    display_assistant=display_assistant,
                    display_thinking=display_thinking,
                    truncate_text=truncate_text,
                )
                assistant_text = parse_result.assistant_text
                analysis_text_parts = parse_result.analysis_text_parts
                parsed_messages = parse_result.parsed_messages
            else:
                # Non-Harmony model: already printed during streaming
                # Use accumulated streamed text (avoid decoding twice)
                print()  # Newline after streaming
                assistant_text = "".join(streamed_text_parts)

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
                            # Harmony expects tool results in "commentary" channel
                            # Record hyperparameters used for this generation cycle
                            tool_result_msg = {
                                "role": "tool",
                                "name": tool_call.tool_name,  # Tool name goes in Author.name
                                "content": result,
                                "recipient": "assistant",  # Tool results are sent to assistant
                                "channel": "commentary",  # Harmony expects tool results in commentary channel
                                "timestamp": make_timestamp(),
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

            # Write raw response to debug file for all models
            if generator.is_gpt_oss and generator.use_harmony and generator.encoding:
                raw_response = generator.encoding.decode(all_generated_tokens)
            else:
                raw_response = generator.tokenizer.decode(all_generated_tokens)
            cleaned_response = clean_text(raw_response)
            write_debug_response(
                debug_path=debug_path,
                raw_response=raw_response,
                cleaned_response=cleaned_response,
                show_console=args.debug,
            )

            # Record hyperparameters used for this generation
            # Check if tool calls exist (for Harmony models) and handle empty assistant_text
            has_tool_calls = (
                generator.is_gpt_oss
                and tools
                and generator.use_harmony
                and parsed_messages is not None
            )

            tool_calls_detected = False
            if has_tool_calls:
                try:
                    tool_calls_check = parse_tool_calls_from_messages(parsed_messages, tools)
                    tool_calls_detected = bool(tool_calls_check)
                except Exception:
                    pass

            # Only save assistant turn if we have actual content (not just analysis or tool calls)
            # If tool calls exist and assistant_text is empty/whitespace, skip saving junk
            if tool_calls_detected and not assistant_text.strip():
                # Tool call issued without meaningful final content - skip saving
                # The tool results will be added in the next iteration (already handled above)
                pass
            elif assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
                # For non-Harmony models, always save; for Harmony models, only if we have final response
                assistant_turn = {
                    "role": "assistant",
                    "content": assistant_text if assistant_text else "[No final response - see thinking above]",
                    "timestamp": make_timestamp(),
                    "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                }
                # For Harmony models, also record analysis channel if present
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    # Join with newlines to preserve structure when saving
                    assistant_turn["analysis"] = "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t")
                conversation.append(assistant_turn)
            else:
                # Harmony model with only analysis channel - save analysis separately
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    # Join with newlines to preserve structure when saving
                    assistant_turn = {
                        "role": "assistant",
                        "content": "[Analysis only - no final response]",
                        "analysis": "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t"),
                        "timestamp": make_timestamp(),
                        "hyperparameters": hyperparameters.copy() if hyperparameters else {},
                    }
                    conversation.append(assistant_turn)

            # Save conversation after each exchange (turn)
            if chat_file_path:
                try:
                    save_conversation(
                        chat_file_path,
                        conversation,
                        model_path,
                        prompt_config_path,
                        tools,
                        hyperparameters,
                    )
                except Exception as e:
                    print(f"\n[WARNING] Failed to save chat: {e}")

            break

    # Final save on exit
    if chat_file_path and conversation:
        try:
            save_conversation(
                chat_file_path,
                conversation,
                model_path,
                prompt_config_path,
                tools,
                hyperparameters,
            )
            print(f"\n[INFO] Chat saved to: {chat_file_path}")
        except Exception as e:
            print(f"\n[WARNING] Failed to save chat on exit: {e}")


if __name__ == "__main__":
    main()
