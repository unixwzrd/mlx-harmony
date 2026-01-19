from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from openai_harmony import Role, StreamableParser
from unicodefix.transforms import clean_text

from mlx_harmony.cli.chat_commands import (
    build_hyperparameters,
    get_assistant_name,
    get_truncate_limits,
    parse_command,
    resolve_max_context_tokens,
    resolve_profile_and_prompt_config,
    truncate_text,
)
from mlx_harmony.cli.chat_voice import (
    init_moshi_components,
    listen_for_user_input,
    voice_status,
)
from mlx_harmony.cli.cli_args import build_parser
from mlx_harmony.config import (
    MoshiConfig,
    apply_placeholders,
    load_profiles,
    load_prompt_config,
)
from mlx_harmony.conversation.conversation_history import (
    display_resume_message,
    find_last_hyperparameters,
    load_chat_session,
    make_message_id,
    make_timestamp,
    resolve_chat_paths,
    resolve_debug_path,
    resolve_dirs_from_config,
    write_debug_metrics,
    write_debug_response,
    write_debug_tokens,
)
from mlx_harmony.conversation.conversation_io import (
    load_conversation,
    read_user_input,
    read_user_input_from_first_line,
    save_conversation,
    try_save_conversation,
)
from mlx_harmony.generation.generation_stream import stream_generation
from mlx_harmony.generation.generator import TokenGenerator
from mlx_harmony.generation.prompt_cache import PromptTokenCache
from mlx_harmony.harmony.harmony_parser import parse_harmony_response
from mlx_harmony.harmony.prompt_builder import (
    build_prompt_token_ids,
    prepare_prompt,
    truncate_conversation_for_context,
)
from mlx_harmony.harmony.tool_calls import handle_tool_calls, has_tool_calls
from mlx_harmony.logging import get_logger
from mlx_harmony.render_output import display_assistant, display_thinking
from mlx_harmony.speech.moshi.loader import chunk_text
from mlx_harmony.tools import (
    get_tools_for_model,
)


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
    hyperparameters: dict[str, float | int | bool]  # Optional: hyperparameters used for generation


logger = get_logger(__name__)

__all__ = ["load_conversation", "save_conversation", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path, prompt_config_path, prompt_config, profile_data = resolve_profile_and_prompt_config(
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
        loaded_max_context_tokens,
        loaded_model_path,
        loaded_chat_id,
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
    no_fs_cache = bool(args.no_fs_cache)
    if not no_fs_cache and prompt_config and prompt_config.no_fs_cache is not None:
        no_fs_cache = bool(prompt_config.no_fs_cache)
    lazy = args.lazy if args.lazy is not None else False

    use_harmony = None
    if prompt_config and prompt_config.use_harmony is not None:
        use_harmony = prompt_config.use_harmony

    generator = TokenGenerator(
        model_path,
        use_harmony=use_harmony,
        prompt_config=prompt_config,
        lazy=lazy,
        mlock=mlock or False,  # Default to False if None
        no_fs_cache=no_fs_cache,
    )
    if generator.use_harmony and generator.encoding:
        generator.prompt_token_cache = PromptTokenCache()

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
    print(
        "[INFO] Type '\\set <param>=<value>' to change hyperparameters (e.g., '\\set temperature=0.7')."
    )
    print("[INFO] Type '\\list' or '\\show' to display current hyperparameters.")
    if chat_file_path:
        print(f"[INFO] Chat will be saved to: {chat_file_path}\n")

    moshi_config, moshi_stt, moshi_tts, smoke_ran = init_moshi_components(args)
    if smoke_ran:
        return

    # Debug file setup: always write debug log by default
    # --debug enables console output, --debug-file overrides the file path
    # Path resolution: absolute paths used as-is, relative paths resolved relative to logs_dir
    # This matches --chat behavior (relative paths resolved relative to chats_dir)
    # Always write to debug file by default (unless explicitly disabled in future)
    debug_path = resolve_debug_path(args.debug_file, logs_dir)
    debug_tokens_mode = args.debug_tokens or "off"
    logger.info("Debug log: %s (tokens: %s)", debug_path, debug_tokens_mode)

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

    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters,
        prompt_config,
        generator.is_gpt_oss and generator.use_harmony,
    )
    last_saved_hyperparameters = (
        find_last_hyperparameters(conversation) or loaded_hyperparameters.copy()
    )
    chat_id = loaded_chat_id

    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=loaded_max_context_tokens,
        loaded_model_path=loaded_model_path,
        prompt_config=prompt_config,
        profile_data=profile_data,
        model_path=model_path,
    )

    # Debug file info is logged above before resuming chat.
    last_prompt_start_time: float | None = None
    generation_index = 0

    def _apply_user_token_limit(text: str) -> str | None:
        if not prompt_config:
            return text
        max_user_tokens = prompt_config.max_user_tokens
        if max_user_tokens is None:
            max_user_tokens = prompt_config.max_tokens
        if max_user_tokens is None:
            return text
        if not text.strip():
            return text
        tokens = generator.tokenizer.encode(text)
        token_count = len(tokens)
        limit = max_user_tokens
        if token_count <= limit:
            return text
        if sys.stdin.isatty():
            print(
                f"[WARN] Input is {token_count} tokens (limit {limit})."
                " Proceed with truncated input? [y/N]: ",
                end="",
                flush=True,
            )
            response = read_user_input("").strip().lower()
            if not response.startswith("y"):
                print("[INFO] Input discarded. Please try again.")
                return None
        else:
            logger.warning(
                "Input is %s tokens (limit %s). Truncating for non-interactive input.",
                token_count,
                limit,
            )
        truncated = generator.tokenizer.decode(tokens[:limit])
        print("[INFO] Using truncated input.")
        return truncated

    while True:
        try:
            if moshi_stt is not None:
                user_input = listen_for_user_input(moshi_stt, moshi_config)
                if user_input:
                    print(f"\n>> {user_input}")
            else:
                user_input = read_user_input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D (EOF) and Ctrl-C gracefully
            print()  # Newline for clean exit
            break
        if moshi_stt is not None and not user_input.strip():
            continue
        if user_input.strip().lower() == "q":
            break

        handled, should_apply, message, updates = parse_command(user_input, hyperparameters)
        if handled:
            if message:
                print(message)
            if should_apply and updates:
                hyperparameters.update(updates)
                if chat_file_path and conversation:
                    error = try_save_conversation(
                        chat_file_path,
                        conversation,
                        model_path,
                        prompt_config_path,
                        tools,
                        hyperparameters,
                    )
                    if error:
                        logger.warning(
                            "Failed to save updated hyperparameters: %s (check file path permissions)",
                            error,
                        )
            continue

        # Add timestamp to user message (turn)
        user_content = (
            apply_placeholders(user_input, prompt_config.placeholders)
            if prompt_config and prompt_config.placeholders
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
        user_content_limited = _apply_user_token_limit(user_content)
        if user_content_limited is None:
            continue
        user_content = user_content_limited

        # Main generation loop with tool call handling
        tool_iteration = 0
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
                generator=generator,
                conversation=conversation,
                system_message=system_message,
                max_context_tokens=max_context_tokens,
            )
            prompt_token_ids = build_prompt_token_ids(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
            )
            prepare_prompt(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
                debug_path=debug_path,
                debug=args.debug,
                debug_tokens=args.debug_tokens,
                prompt_token_ids=prompt_token_ids,
            )

            # Use hyperparameters dict (CLI args already merged with loaded values)
            # For Harmony models, we'll parse messages to extract final channel content
            # For Harmony models, use StreamableParser for incremental parsing
            # For non-Harmony models, stream tokens directly
            voice_status("Thinking") if moshi_stt is not None else None
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
                generator.streamable_parser = StreamableParser(
                    generator.encoding, Role.ASSISTANT, strict=False
                )

            seed_value = hyperparameters.get("seed", -1)
            reseed_each_turn = bool(hyperparameters.get("reseed_each_turn", False))
            effective_seed: int | None = None
            if isinstance(seed_value, int | float) and int(seed_value) >= 0:
                effective_seed = int(seed_value)
                if reseed_each_turn:
                    effective_seed += generation_index

            tts_stream_queue: queue.Queue[str | None] | None = None
            tts_stream_thread: threading.Thread | None = None
            tts_stream_buffer = ""

            def _start_tts_stream() -> None:
                nonlocal tts_stream_queue, tts_stream_thread
                if not moshi_tts or not moshi_config or not moshi_config.tts_stream:
                    return
                if moshi_config.barge_in:
                    raise RuntimeError("TTS streaming does not support barge-in yet.")
                tts_stream_queue = queue.Queue()

                def _tts_worker() -> None:
                    while True:
                        chunk = tts_stream_queue.get()
                        if chunk is None:
                            break
                        moshi_tts.speak(chunk)

                tts_stream_thread = threading.Thread(target=_tts_worker, daemon=True)
                tts_stream_thread.start()

            def _enqueue_tts_chunk(
                chunk: str,
                queue_ref: queue.Queue[str | None] | None,
            ) -> None:
                if queue_ref is None:
                    return
                if chunk.strip():
                    queue_ref.put(chunk.strip())

            def _flush_tts_stream(
                queue_ref: queue.Queue[str | None] | None,
                thread_ref: threading.Thread | None,
            ) -> None:
                nonlocal tts_stream_buffer
                if queue_ref is None:
                    return
                if tts_stream_buffer.strip():
                    _enqueue_tts_chunk(tts_stream_buffer, queue_ref)
                tts_stream_buffer = ""
                queue_ref.put(None)
                if thread_ref:
                    thread_ref.join()

            def _reset_tts_stream(
                queue_ref: queue.Queue[str | None] | None,
                thread_ref: threading.Thread | None,
            ) -> None:
                nonlocal tts_stream_buffer
                tts_stream_buffer = ""
                if queue_ref is None:
                    return
                queue_ref.put(None)
                if thread_ref:
                    thread_ref.join()

            def _maybe_emit_tts_stream(
                delta: str, channel: str | None, role: object | None
            ) -> None:
                nonlocal tts_stream_buffer
                if not moshi_config or not moshi_config.tts_stream:
                    return
                if role is not None and str(role) != "Role.ASSISTANT":
                    return
                if channel not in (None, "final", "commentary"):
                    return
                tts_stream_buffer += delta
                chunk_limit = moshi_config.tts_chunk_chars
                min_chars = moshi_config.tts_chunk_min_chars
                while True:
                    if len(tts_stream_buffer) < min_chars:
                        return
                    cut_at = -1
                    for mark in (".", "!", "?", ";", ":"):
                        idx = tts_stream_buffer.rfind(mark, 0, chunk_limit + 1)
                        if idx > cut_at:
                            cut_at = idx
                    if cut_at == -1 and len(tts_stream_buffer) < chunk_limit:
                        return
                    if cut_at == -1:
                        cut_at = chunk_limit
                    chunk = tts_stream_buffer[: cut_at + 1].strip()
                    tts_stream_buffer = tts_stream_buffer[cut_at + 1 :].lstrip()
                    _enqueue_tts_chunk(chunk, tts_stream_queue)  # noqa: B023

            _start_tts_stream()

            tokens, all_generated_tokens, streamed_text_parts = stream_generation(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
                prompt_token_ids=prompt_token_ids,
                hyperparameters=hyperparameters,
                seed=effective_seed,
                on_text=lambda text: print(text, end="", flush=True)
                if not (generator.is_gpt_oss and generator.use_harmony)
                else (lambda _text: None),
                on_harmony_text=_maybe_emit_tts_stream,
            )
            generation_index += 1

            # After generation, keep model parameters active to prevent swapping
            # This ensures buffers stay wired and don't get swapped out
            # Use generator's keepalive() method (unified behavior)
            generator.keepalive()

            generation_end_time = time.perf_counter()
            generation_elapsed = generation_end_time - generation_start_time
            num_generated_tokens = len(tokens)
            tokens_per_second = (
                num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )

            # Display generation stats
            print(
                f"\n[INFO] Generated {num_generated_tokens} tokens in {generation_elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)"
            )

            write_debug_metrics(
                debug_path=debug_path,
                metrics={
                    "prompt_tokens": prompt_token_count,
                    "generated_tokens": num_generated_tokens,
                    "elapsed_seconds": generation_elapsed,
                    "tokens_per_second": tokens_per_second,
                    "prompt_start_to_prompt_start_seconds": prompt_start_delta,
                    "max_context_tokens": max_context_tokens,
                },
            )

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

            should_continue, parsed_messages = handle_tool_calls(
                generator=generator,
                tokens=tokens,
                parsed_messages=parsed_messages,
                tools=tools,
                conversation=conversation,
                hyperparameters=hyperparameters,
            )
            if should_continue:
                if moshi_config and moshi_config.tts_stream:
                    _reset_tts_stream(tts_stream_queue, tts_stream_thread)
                tool_iteration += 1
                continue

            # No tool calls or non-GPT-OSS model: assistant_text already set above
            # (For Harmony models, it's the final channel content; for others, it's decoded tokens)
            if moshi_config and moshi_config.tts_stream and moshi_tts is not None:
                _flush_tts_stream(tts_stream_queue, tts_stream_thread)
            elif moshi_tts is not None and assistant_text.strip():
                chunk_chars = moshi_config.tts_chunk_chars if moshi_config else 180
                chunk_sentences = moshi_config.tts_chunk_sentences if moshi_config else True
                chunk_min_chars = moshi_config.tts_chunk_min_chars if moshi_config else 60
                for chunk in chunk_text(
                    assistant_text,
                    chunk_chars,
                    chunk_sentences,
                    min_chars=chunk_min_chars,
                ):
                    voice_status("Speaking")
                    stop_event = None
                    monitor_thread = None
                    if moshi_config and moshi_config.barge_in and moshi_stt is not None:
                        stop_event = threading.Event()

                        def _monitor_barge_in(stop_event_ref: threading.Event = stop_event) -> None:
                            try:
                                transcript = moshi_stt.listen_once(
                                    max_seconds=moshi_config.barge_in_window_seconds,
                                    vad=True,
                                    vad_threshold=moshi_config.stt_vad_threshold,
                                    vad_hits_required=moshi_config.stt_vad_hits,
                                    silence=moshi_config.stt_silence,
                                    silence_threshold=moshi_config.stt_silence_threshold,
                                    silence_ms=moshi_config.stt_silence_ms,
                                    min_speech_ms=moshi_config.stt_min_speech_ms,
                                )
                                if transcript:
                                    stop_event_ref.set()
                            except Exception as exc:
                                logger.warning("Barge-in monitor failed: %s", exc)

                        monitor_thread = threading.Thread(target=_monitor_barge_in, daemon=True)
                        monitor_thread.start()
                    tts_start = time.perf_counter()
                    moshi_tts.speak(chunk, stop_event=stop_event)
                    tts_elapsed = time.perf_counter() - tts_start
                    logger.info("Moshi TTS chunk duration: %.2fs", tts_elapsed)
                    if stop_event and stop_event.is_set():
                        print("[VOICE] Barge-in detected. Speak again to continue.")
                        break

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
            tool_calls_detected = has_tool_calls(
                parsed_messages=parsed_messages,
                tools=tools,
            )

            # Only save assistant turn if we have actual content (not just analysis or tool calls)
            # If tool calls exist and assistant_text is empty/whitespace, skip saving junk
            if tool_calls_detected and not assistant_text.strip():
                # Tool call issued without meaningful final content - skip saving
                # The tool results will be added in the next iteration (already handled above)
                pass
            elif assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
                # For non-Harmony models, always save; for Harmony models, only if we have final response
                parent_id = conversation[-1].get("id") if conversation else None
                message_id = make_message_id()
                assistant_turn = {
                    "id": message_id,
                    "parent_id": parent_id,
                    "cache_key": message_id,
                    "role": "assistant",
                    "content": assistant_text
                    if assistant_text
                    else "[No final response - see thinking above]",
                    "timestamp": make_timestamp(),
                }
                if hyperparameters and hyperparameters != last_saved_hyperparameters:
                    assistant_turn["hyperparameters"] = hyperparameters.copy()
                    last_saved_hyperparameters = hyperparameters.copy()
                # For Harmony models, also record analysis channel if present
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                    # Join with newlines to preserve structure when saving
                    assistant_turn["analysis"] = (
                        "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t")
                    )
                conversation.append(assistant_turn)
            else:
                # Harmony model with only analysis channel - save analysis separately
                if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
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
            if chat_file_path:
                error = try_save_conversation(
                    chat_file_path,
                    conversation,
                    model_path,
                    prompt_config_path,
                    prompt_config.model_dump() if prompt_config else None,
                    tools,
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
    if chat_file_path and conversation:
        error = try_save_conversation(
            chat_file_path,
            conversation,
            model_path,
            prompt_config_path,
            prompt_config.model_dump() if prompt_config else None,
            tools,
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
            print(f"\n[INFO] Chat saved to: {chat_file_path}")


if __name__ == "__main__":
    main()
