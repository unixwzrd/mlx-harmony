from __future__ import annotations

import sys
import time
from typing import Any, Literal, TypedDict

from unicodefix.transforms import clean_text

from mlx_harmony.chat_bootstrap import bootstrap_chat
from mlx_harmony.chat_generation import stream_generation
from mlx_harmony.chat_harmony import parse_harmony_response
from mlx_harmony.chat_history import (
    display_resume_message,
    find_last_hyperparameters,
    make_message_id,
    make_timestamp,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.chat_io import (
    load_conversation,
    read_user_input,
    save_conversation,
    try_save_conversation,
)
from mlx_harmony.chat_prompt import (
    build_prompt_token_ids,
    prepare_prompt,
    truncate_conversation_for_context,
)
from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.chat_utils import (
    build_hyperparameters,
    parse_command,
    resolve_max_context_tokens,
    truncate_text,
)
from mlx_harmony.config import apply_placeholders
from mlx_harmony.logging import get_logger
from mlx_harmony.tools.runner import has_tool_calls, run_tools_if_requested


def _collect_memory_stats() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception:
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats


def _decode_raw_response(
    *,
    token_ids: list[int],
    decode_token: callable,
) -> str:
    """Decode tokens one-by-one to preserve special token markers."""
    return "".join(decode_token([token_id]) for token_id in token_ids)


def _decode_prompt_tail(
    *,
    token_ids: list[int],
    decode_token: callable,
    start_token_id: int = 200006,
) -> str:
    """Decode prompt tail from the last <|start|> token to preserve header tags."""
    last_start_idx = -1
    for idx in range(len(token_ids) - 1, -1, -1):
        if token_ids[idx] == start_token_id:
            last_start_idx = idx
            break
    if last_start_idx < 0:
        return ""
    return "".join(decode_token([token_id]) for token_id in token_ids[last_start_idx:])


# Type definitions for message records
class MessageDict(TypedDict, total=False):
    """Type definition for conversation messages."""
    id: str
    parent_id: str | None
    cache_key: str | None
    role: Literal["user", "assistant", "tool", "system", "developer"]
    content: str
    timestamp: str
    name: str  # Optional: for tool messages
    recipient: str  # Optional: for tool messages
    channel: str  # Optional: for Harmony messages (e.g., "commentary", "analysis", "final")
    analysis: str  # Optional: for assistant messages with thinking/analysis
    hyperparameters: dict[str, float | int | bool | str]  # Optional: hyperparameters used for generation


logger = get_logger(__name__)


def main() -> None:
    bootstrap = bootstrap_chat()
    args = bootstrap.args
    conversation = bootstrap.conversation
    model_path = bootstrap.model_path
    prompt_config_path = bootstrap.prompt_config_path
    context = bootstrap.context
    loaded_hyperparameters = bootstrap.loaded_hyperparameters
    loaded_max_context_tokens = bootstrap.loaded_max_context_tokens
    loaded_model_path = bootstrap.loaded_model_path
    loaded_chat_id = bootstrap.loaded_chat_id

    print(f"[INFO] Starting chat with model: {model_path}")
    if context.generator.is_gpt_oss:
        print("[INFO] GPT-OSS model detected - Harmony format enabled.")
        if context.tools:
            enabled = ", ".join(t.name for t in context.tools if t.enabled)
            print(f"[INFO] Tools enabled: {enabled}")
    else:
        print("[INFO] Non-GPT-OSS model - using native chat template.")

    print("[INFO] Type 'q' or `Control-D` to quit.")
    print("[INFO] Type '\\help' to list all out-of-band commands.")
    print("[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7').")
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if context.chat_file_path:
        print(f"[INFO] Chat will be saved to: {context.chat_file_path}\n")

    has_conversation_history = display_resume_message(
        conversation,
        context.assistant_name,
        context.thinking_limit,
        context.response_limit,
        context.render_markdown,
        display_assistant,
        display_thinking,
        truncate_text,
    )

    # Optional assistant greeting from prompt config (only when starting fresh)
    # Print greeting AFTER quit instruction and save info (or after resume message)
    # Only show greeting if we don't have conversation history
    if (
        not has_conversation_history
        and context.prompt_config
        and context.prompt_config.assistant_greeting
    ):
        greeting_text = apply_placeholders(
            context.prompt_config.assistant_greeting,
            context.prompt_config.placeholders,
        )
        # Use markdown rendering for greeting if enabled
        display_assistant(greeting_text, context.assistant_name, context.render_markdown)
        parent_id = conversation[-1].get("id") if conversation else None
        message_id = make_message_id()
        conversation.append(
            {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": "assistant",
                "content": greeting_text,
                "timestamp": make_timestamp(),
            }
        )

    max_tool_iterations = 10  # Prevent infinite loops
    max_resume_attempts = 2
    resume_base_hyperparameters: dict[str, float | int | bool | str] | None = None

    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters,
        context.prompt_config,
        context.generator.is_gpt_oss and context.generator.use_harmony,
    )
    last_saved_hyperparameters = find_last_hyperparameters(conversation) or loaded_hyperparameters.copy()
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=context.prompt_config,
        profile_data=context.profile_data,
        model_path=model_path,
    )

    # Debug file info is logged above before resuming chat.
    last_prompt_start_time: float | None = None
    generation_index = 0

    while True:
        try:
            user_input = read_user_input("\n>> ")
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
                if context.chat_file_path and conversation:
                    error = try_save_conversation(
                        context.chat_file_path,
                        conversation,
                        model_path,
                        prompt_config_path,
                        context.tools,
                        hyperparameters,
                    )
                    if error:
                        logger.warning(
                            "Failed to save updated hyperparameters: %s (check file path permissions)",
                            error,
                        )
            continue

        # Add timestamp to user message (turn)
        last_user_text = user_input
        user_content = (
            apply_placeholders(user_input, context.prompt_config.placeholders)
            if context.prompt_config and context.prompt_config.placeholders
            else user_input
        )
        parent_id = conversation[-1].get("id") if conversation else None
        message_id = make_message_id()
        user_turn = {
            "id": message_id,
            "parent_id": parent_id,
            "cache_key": message_id,
            "role": "user",
            "content": user_content,
            "timestamp": make_timestamp(),
        }
        conversation.append(user_turn)

        # Main generation loop with tool call handling
        tool_iteration = 0
        resume_attempts = 0
        while tool_iteration < max_tool_iterations:
            tokens: list[int] = []
            prompt_start_time = time.perf_counter()
            prompt_start_delta = (
                prompt_start_time - last_prompt_start_time
                if last_prompt_start_time is not None
                else None
            )
            last_prompt_start_time = prompt_start_time

            # system_message parameter is for CLI override (if we add --system flag later)
            # The generator's render_prompt() already handles system_model_identity from prompt_config
            # So we pass None here and let the generator handle the fallback
            system_message = None  # Could be overridden by CLI --system flag in the future

            # Debug: always write raw prompt to file, print to console only if --debug is set
            prompt_conversation, prompt_token_count = truncate_conversation_for_context(
                generator=context.generator,
                conversation=conversation,
                system_message=system_message,
                max_context_tokens=max_context_tokens,
            )
            prompt_token_ids = build_prompt_token_ids(
                generator=context.generator,
                conversation=prompt_conversation,
                system_message=system_message,
            )
            raw_prompt = prepare_prompt(
                generator=context.generator,
                conversation=prompt_conversation,
                system_message=system_message,
                debug_path=context.debug_path,
                debug=args.debug,
                debug_tokens=args.debug_tokens,
                prompt_token_ids=prompt_token_ids,
            )

            # Use hyperparameters dict (CLI args already merged with loaded values)
            # For Harmony models, parse messages after generation (single pass)
            # For non-Harmony models, stream tokens directly
            generation_start_time = time.perf_counter()
            # Initialize these early to avoid scope issues (used outside Harmony branch)
            parsed_messages: Any | None = None  # Will be set for Harmony models
            analysis_text_parts: list[str] = []
            assistant_text = ""
            # Accumulate streamed text for non-Harmony models (avoid decoding twice)
            streamed_text_parts: list[str] = []

            seed_value = hyperparameters.get("seed", -1)
            reseed_each_turn = bool(hyperparameters.get("reseed_each_turn", False))
            effective_seed: int | None = None
            if isinstance(seed_value, (int, float)) and int(seed_value) >= 0:
                effective_seed = int(seed_value)
                if reseed_each_turn:
                    effective_seed += generation_index

            tokens, all_generated_tokens, streamed_text_parts = stream_generation(
                generator=context.generator,
                conversation=prompt_conversation,
                system_message=system_message,
                prompt_token_ids=prompt_token_ids,
                hyperparameters=hyperparameters,
                seed=effective_seed,
                on_text=lambda text: print(text, end="", flush=True),
            )
            generation_index += 1

            # After generation, keep model parameters active to prevent swapping
            # This ensures buffers stay wired and don't get swapped out
            # Use generator's keepalive() method (unified behavior)
            context.generator.keepalive()

            generation_end_time = time.perf_counter()
            generation_elapsed = generation_end_time - generation_start_time
            num_generated_tokens = len(tokens)
            tokens_per_second = num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0

            # Display generation stats
            print(f"\n[INFO] Generated {num_generated_tokens} tokens in {generation_elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

            memory_stats = _collect_memory_stats()
            write_debug_metrics(
                debug_path=context.debug_path,
                metrics={
                    "prompt_tokens": prompt_token_count,
                    "generated_tokens": num_generated_tokens,
                    "elapsed_seconds": generation_elapsed,
                    "tokens_per_second": tokens_per_second,
                    "prompt_start_to_prompt_start_seconds": prompt_start_delta,
                    "max_context_tokens": max_context_tokens,
                    "prefill_start_offset": getattr(
                        context.generator, "_last_prefill_start_offset", None
                    ),
                    **memory_stats,
                },
            )

            # Write all generated tokens to debug file (once, not duplicated)
            write_debug_tokens(
                debug_path=context.debug_path,
                token_ids=all_generated_tokens,
                decode_tokens=(
                    context.generator.encoding.decode
                    if context.generator.encoding
                    else None
                ),
                label="response",
                mode=args.debug_tokens or "off",
            )

            if (
                context.generator.is_gpt_oss
                and context.generator.use_harmony
                and context.generator.encoding
            ):
                raw_prefix = ""
                if prompt_token_ids:
                    raw_prefix = _decode_prompt_tail(
                        token_ids=prompt_token_ids,
                        decode_token=context.generator.encoding.decode,
                    )
                raw_response = _decode_raw_response(
                    token_ids=all_generated_tokens,
                    decode_token=context.generator.encoding.decode,
                )
                if raw_prefix and not raw_response.lstrip().startswith("<|"):
                    raw_response = f"{raw_prefix}{raw_response}"
                if (
                    context.generator.last_finish_reason == "stop"
                    and context.generator.last_stop_token_id is not None
                ):
                    stop_text = context.generator.encoding.decode(
                        [context.generator.last_stop_token_id]
                    )
                    if stop_text and not raw_response.endswith(stop_text):
                        raw_response = f"{raw_response}{stop_text}"
            else:
                raw_response = _decode_raw_response(
                    token_ids=all_generated_tokens,
                    decode_token=context.generator.tokenizer.decode,
                )
                if (
                    context.generator.last_finish_reason == "stop"
                    and context.generator.last_stop_token_id is not None
                ):
                    stop_text = context.generator.tokenizer.decode(
                        [context.generator.last_stop_token_id]
                    )
                    if stop_text and not raw_response.endswith(stop_text):
                        raw_response = f"{raw_response}{stop_text}"
            resume_reason: str | None = None
            if context.generator.last_finish_reason == "length":
                resume_reason = "length"
            elif context.generator.last_stop_reason == "loop_detected":
                resume_reason = "loop_detected"
            should_resume = resume_reason is not None and resume_attempts < max_resume_attempts

            cleaned_response = clean_text(raw_response)
            write_debug_response(
                debug_path=context.debug_path,
                raw_response=raw_response,
                cleaned_response=cleaned_response,
                show_console=args.debug and not should_resume,
            )
            write_debug_token_texts(
                debug_path=context.debug_path,
                token_ids=all_generated_tokens,
                decode_token=(
                    context.generator.encoding.decode
                    if context.generator.encoding
                    else context.generator.tokenizer.decode
                ),
                label="response",
                mode=args.debug_tokens or "off",
            )
            if should_resume:
                resume_base_hyperparameters = hyperparameters.copy()
                if resume_reason == "loop_detected":
                    print("[INFO] Response became repetitive; rethinking to complete the final answer...")
                else:
                    print("[INFO] Response truncated; rethinking to complete the final answer...")
                resume_hyperparameters = hyperparameters.copy()
                current_penalty = float(resume_hyperparameters.get("repetition_penalty") or 1.0)
                penalty_floor = 1.15 if resume_reason == "length" else 1.25
                resume_hyperparameters["repetition_penalty"] = max(current_penalty, penalty_floor)
                current_context = int(resume_hyperparameters.get("repetition_context_size") or 0)
                context_floor = 1024 if resume_reason == "length" else 2048
                resume_hyperparameters["repetition_context_size"] = max(current_context, context_floor)
                resume_hyperparameters["loop_detection"] = "full"
            else:
                resume_hyperparameters = hyperparameters

            if context.generator.is_gpt_oss and context.generator.use_harmony:
                parse_result = parse_harmony_response(
                    generator=context.generator,
                    tokens=tokens,
                    streamed_text_parts=streamed_text_parts,
                    assistant_name=context.assistant_name,
                    thinking_limit=context.thinking_limit,
                    response_limit=context.response_limit,
                    render_markdown=context.render_markdown,
                    debug=args.debug,
                    display_assistant=display_assistant,
                    display_thinking=display_thinking,
                    truncate_text=truncate_text,
                    suppress_display=should_resume,
                )
                assistant_text = parse_result.assistant_text
                analysis_text_parts = parse_result.analysis_text_parts
                parsed_messages = parse_result.parsed_messages
            else:
                # Non-Harmony model: already printed during streaming
                # Use accumulated streamed text (avoid decoding twice)
                print()  # Newline after streaming
                assistant_text = "".join(streamed_text_parts)

            if should_resume:
                parent_id = conversation[-1].get("id") if conversation else None
                message_id = make_message_id()
                assistant_turn = {
                    "id": message_id,
                    "parent_id": parent_id,
                    "cache_key": message_id,
                    "role": "assistant",
                    "content": "[Response truncated - recovery in progress]",
                    "timestamp": make_timestamp(),
                }
                if hyperparameters and hyperparameters != last_saved_hyperparameters:
                    assistant_turn["hyperparameters"] = hyperparameters.copy()
                    last_saved_hyperparameters = hyperparameters.copy()
                conversation.append(assistant_turn)

                list_keywords = (
                    "list",
                    "names",
                    "examples",
                    "top ",
                    "best ",
                    "types",
                    "kinds",
                    "ways",
                    "steps",
                    "ideas",
                )
                list_likely = False
                if last_user_text:
                    lowered = last_user_text.lower()
                    list_likely = any(keyword in lowered for keyword in list_keywords)

                if resume_reason == "loop_detected":
                    resume_prompt = (
                        "Your previous reply became repetitive. Provide a concise final answer from the "
                        "beginning with no repeated phrases. Use a short paragraph (no lists). Do not say "
                        "'continued' or refer to earlier output."
                    )
                else:
                    if list_likely:
                        resume_prompt = (
                            "Your previous reply was truncated. Provide the complete answer "
                            "from the beginning. Do not say 'continued' or refer to earlier output. "
                            "Keep the answer concise; if a list is appropriate, limit it to at most 8 "
                            "items and do not repeat any item."
                        )
                    else:
                        resume_prompt = (
                            "Your previous reply was truncated. Provide the complete answer "
                            "from the beginning. Do not say 'continued' or refer to earlier output. "
                            "Keep the answer concise and avoid repeating phrases. Use a short paragraph."
                        )
                parent_id = conversation[-1].get("id")
                message_id = make_message_id()
                resume_turn = {
                    "id": message_id,
                    "parent_id": parent_id,
                    "cache_key": message_id,
                    "role": "user",
                    "content": resume_prompt,
                    "timestamp": make_timestamp(),
                }
                conversation.append(resume_turn)
                resume_attempts += 1
                hyperparameters = resume_hyperparameters
                continue

            if resume_base_hyperparameters is not None and not should_resume:
                hyperparameters = resume_base_hyperparameters
                resume_base_hyperparameters = None

            should_continue, parsed_messages = run_tools_if_requested(
                generator=context.generator,
                tokens=tokens,
                parsed_messages=parsed_messages,
                tools=context.tools,
                conversation=conversation,
                hyperparameters=hyperparameters,
            )
            if should_continue:
                tool_iteration += 1
                continue

            # No tool calls or non-GPT-OSS model: assistant_text already set above
            # (For Harmony models, it's the final channel content; for others, it's decoded tokens)

            # Write raw response to debug file for all models
            # Record hyperparameters used for this generation
            # Check if tool calls exist (for Harmony models) and handle empty assistant_text
            tool_calls_detected = has_tool_calls(
                parsed_messages=parsed_messages,
                tools=context.tools,
            )

            # Only save assistant turn if we have actual content (not just analysis or tool calls)
            # If tool calls exist and assistant_text is empty/whitespace, skip saving junk
            if tool_calls_detected and not assistant_text.strip():
                # Tool call issued without meaningful final content - skip saving
                # The tool results will be added in the next iteration (already handled above)
                pass
            elif assistant_text or (
                not context.generator.is_gpt_oss or not context.generator.use_harmony
            ):
                # For non-Harmony models, always save; for Harmony models, only if we have final response
                parent_id = conversation[-1].get("id") if conversation else None
                message_id = make_message_id()
                assistant_turn = {
                    "id": message_id,
                    "parent_id": parent_id,
                    "cache_key": message_id,
                    "role": "assistant",
                    "content": assistant_text if assistant_text else "[No final response - see thinking above]",
                    "timestamp": make_timestamp(),
                }
                if hyperparameters and hyperparameters != last_saved_hyperparameters:
                    assistant_turn["hyperparameters"] = hyperparameters.copy()
                    last_saved_hyperparameters = hyperparameters.copy()
                # For Harmony models, also record analysis channel if present
                if (
                    context.generator.is_gpt_oss
                    and context.generator.use_harmony
                    and analysis_text_parts
                ):
                    # Join with newlines to preserve structure when saving
                    assistant_turn["analysis"] = "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t")
                conversation.append(assistant_turn)
            else:
                # Harmony model with only analysis channel - save analysis separately
                if (
                    context.generator.is_gpt_oss
                    and context.generator.use_harmony
                    and analysis_text_parts
                ):
                    # Join with newlines to preserve structure when saving
                    parent_id = conversation[-1].get("id") if conversation else None
                    message_id = make_message_id()
                    assistant_turn = {
                        "id": message_id,
                        "parent_id": parent_id,
                        "cache_key": message_id,
                        "role": "assistant",
                        "content": "[Analysis only - no final response]",
                        "analysis": "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t"),
                        "timestamp": make_timestamp(),
                    }
                    if hyperparameters and hyperparameters != last_saved_hyperparameters:
                        assistant_turn["hyperparameters"] = hyperparameters.copy()
                        last_saved_hyperparameters = hyperparameters.copy()
                    conversation.append(assistant_turn)

            # Save conversation after each exchange (turn)
            if context.chat_file_path:
                error = try_save_conversation(
                    context.chat_file_path,
                    conversation,
                    model_path,
                    prompt_config_path,
                    context.prompt_config.model_dump() if context.prompt_config else None,
                    context.tools,
                    hyperparameters,
                    max_context_tokens,
                    chat_id,
                )
                if error:
                    logger.warning(
                        "Failed to save chat: %s (check file path permissions)",
                        error,
                    )

            break

    # Final save on exit
    if context.chat_file_path and conversation:
        error = try_save_conversation(
            context.chat_file_path,
            conversation,
            model_path,
            prompt_config_path,
            context.prompt_config.model_dump() if context.prompt_config else None,
            context.tools,
            hyperparameters,
            max_context_tokens,
            chat_id,
        )
        if error:
            logger.warning(
                "Failed to save chat on exit: %s (check file path permissions)",
                error,
            )
        else:
            print(f"\n[INFO] Chat saved to: {context.chat_file_path}")


if __name__ == "__main__":
    main()
