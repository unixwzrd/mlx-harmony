from __future__ import annotations

import threading
import time
from typing import Any

from openai_harmony import Role, StreamableParser
from unicodefix.transforms import clean_text

from mlx_harmony.cli.chat_commands import truncate_text
from mlx_harmony.cli.chat_prompt import prepare_generation_context
from mlx_harmony.cli.chat_voice import voice_status
from mlx_harmony.cli.tts_stream import TTSStreamController
from mlx_harmony.conversation.conversation_history import (
    make_message_id,
    make_timestamp,
    write_debug_metrics,
    write_debug_response,
    write_debug_tokens,
)
from mlx_harmony.conversation.conversation_io import try_save_conversation
from mlx_harmony.generation.generation_stream import stream_generation
from mlx_harmony.harmony.harmony_parser import parse_harmony_response
from mlx_harmony.harmony.tool_calls import handle_tool_calls, has_tool_calls
from mlx_harmony.logging import get_logger
from mlx_harmony.render_output import display_assistant, display_thinking
from mlx_harmony.speech.moshi.loader import chunk_text

logger = get_logger(__name__)


def run_generation_loop(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    prompt_config: Any | None,
    prompt_config_path: str | None,
    tools: list[Any],
    hyperparameters: dict[str, float | int | bool],
    assistant_name: str,
    thinking_limit: int,
    response_limit: int,
    render_markdown: bool,
    debug_path: str | None,
    args: Any,
    max_context_tokens: int | None,
    moshi_stt: Any | None,
    moshi_tts: Any | None,
    moshi_config: Any | None,
    last_saved_hyperparameters: dict[str, float | int | bool],
    last_prompt_start_time: float | None,
    generation_index: int,
    chat_file_path: str | None,
    model_path: str,
    chat_id: str | None,
    max_tool_iterations: int,
) -> tuple[float | None, int, dict[str, float | int | bool]]:
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

        system_message = None

        (
            prompt_conversation,
            prompt_token_count,
            prompt_token_ids,
        ) = prepare_generation_context(
            generator=generator,
            conversation=conversation,
            system_message=system_message,
            max_context_tokens=max_context_tokens,
            debug_path=debug_path,
            debug=args.debug,
            debug_tokens=args.debug_tokens,
        )

        voice_status("Thinking") if moshi_stt is not None else None
        generation_start_time = time.perf_counter()
        parsed_messages: Any | None = None
        analysis_text_parts: list[str] = []
        assistant_text = ""
        streamed_text_parts: list[str] = []

        if generator.is_gpt_oss and generator.use_harmony and generator.encoding:
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

        tts_stream = TTSStreamController(moshi_tts, moshi_config)
        tts_stream.start()

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
            on_harmony_text=tts_stream.maybe_emit,
        )
        generation_index += 1

        generator.keepalive()

        generation_end_time = time.perf_counter()
        generation_elapsed = generation_end_time - generation_start_time
        num_generated_tokens = len(tokens)
        tokens_per_second = (
            num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
        )

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
            print()
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
                tts_stream.reset()
            tool_iteration += 1
            continue

        if moshi_config and moshi_config.tts_stream and moshi_tts is not None:
            tts_stream.flush()
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

                    monitor_thread = threading.Thread(
                        target=_monitor_barge_in, daemon=True
                    )
                    monitor_thread.start()
                tts_start = time.perf_counter()
                moshi_tts.speak(chunk, stop_event=stop_event)
                tts_elapsed = time.perf_counter() - tts_start
                logger.info("Moshi TTS chunk duration: %.2fs", tts_elapsed)
                if stop_event and stop_event.is_set():
                    print("[VOICE] Barge-in detected. Speak again to continue.")
                    break

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

        tool_calls_detected = has_tool_calls(
            parsed_messages=parsed_messages,
            tools=tools,
        )

        if tool_calls_detected and not assistant_text.strip():
            pass
        elif assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
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
            if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
                assistant_turn["analysis"] = (
                    "\n".join(analysis_text_parts).lstrip(" \t").rstrip(" \t")
                )
            conversation.append(assistant_turn)
        else:
            if generator.is_gpt_oss and generator.use_harmony and analysis_text_parts:
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

    return last_prompt_start_time, generation_index, last_saved_hyperparameters
