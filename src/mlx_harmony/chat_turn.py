from __future__ import annotations

import time
from typing import Any, Callable

from mlx_harmony.chat_adapters import get_adapter
from mlx_harmony.chat_attempt import run_generation_attempt
from mlx_harmony.chat_prompt import (
    build_prompt_token_ids,
    prepare_prompt,
    truncate_conversation_for_context,
)
from mlx_harmony.chat_types import TurnResult
from mlx_harmony.tools.runner import has_tool_calls, run_tools_if_requested


def run_chat_turn(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    hyperparameters: dict[str, float | int | bool | str],
    last_saved_hyperparameters: dict[str, float | int | bool | str],
    assistant_name: str,
    thinking_limit: int | None,
    response_limit: int | None,
    render_markdown: bool,
    debug: bool,
    debug_path: Any,
    debug_tokens: str | None,
    enable_artifacts: bool,
    max_context_tokens: int | None,
    max_tool_iterations: int,
    max_resume_attempts: int,
    tools: list[Any],
    last_user_text: str | None,
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
    display_assistant: Callable[..., None],
    display_thinking: Callable[..., None],
    truncate_text: Callable[[str, int], str],
    collect_memory_stats: Callable[[], dict[str, Any]],
    write_debug_metrics: Callable[..., None],
    write_debug_response: Callable[..., None],
    write_debug_token_texts: Callable[..., None],
    write_debug_tokens: Callable[..., None],
    last_prompt_start_time: float | None,
    generation_index: int,
) -> TurnResult:
    tool_iteration = 0
    resume_attempts = 0
    resume_base_hyperparameters: dict[str, float | int | bool | str] | None = None
    pending_resume_prompt: str | None = None

    while tool_iteration < max_tool_iterations:
        prompt_start_time = time.perf_counter()
        prompt_start_delta = (
            prompt_start_time - last_prompt_start_time
            if last_prompt_start_time is not None
            else None
        )
        last_prompt_start_time = prompt_start_time

        working_conversation = conversation
        if pending_resume_prompt:
            working_conversation = list(conversation) + [
                {
                    "role": "user",
                    "content": pending_resume_prompt,
                }
            ]

        system_message = None
        max_context_tokens_margin: int | None = None
        if getattr(generator, "prompt_config", None) is not None:
            max_context_tokens_margin = getattr(
                generator.prompt_config, "max_context_tokens_margin", None
            )
        prompt_conversation, prompt_token_count = truncate_conversation_for_context(
            generator=generator,
            conversation=working_conversation,
            system_message=system_message,
            max_context_tokens=max_context_tokens,
            max_context_tokens_margin=max_context_tokens_margin,
        )
        prompt_token_ids = build_prompt_token_ids(
            generator=generator,
            conversation=prompt_conversation,
            system_message=system_message,
        )
        raw_prompt = prepare_prompt(
            generator=generator,
            conversation=prompt_conversation,
            system_message=system_message,
            debug_path=debug_path,
            debug=debug,
            debug_tokens=debug_tokens,
            prompt_token_ids=prompt_token_ids,
        )

        seed_value = hyperparameters.get("seed", -1)
        reseed_each_turn = bool(hyperparameters.get("reseed_each_turn", False))
        effective_seed: int | None = None
        if isinstance(seed_value, (int, float)) and int(seed_value) >= 0:
            effective_seed = int(seed_value)
            if reseed_each_turn:
                effective_seed += generation_index

        adapter = get_adapter(generator)
        logs_dir = debug_path.parent if debug_path else None
        outcome = run_generation_attempt(
            generator=generator,
            adapter=adapter,
            prompt_conversation=prompt_conversation,
            prompt_token_ids=prompt_token_ids,
            raw_prompt=raw_prompt,
            system_message=system_message,
            hyperparameters=hyperparameters,
            seed=effective_seed,
            on_text=lambda text: print(text, end="", flush=True),
            assistant_name=assistant_name,
            thinking_limit=thinking_limit,
            response_limit=response_limit,
            render_markdown=render_markdown,
            debug=debug,
            display_assistant=display_assistant,
            display_thinking=display_thinking,
            truncate_text=truncate_text,
            debug_path=debug_path,
            logs_dir=logs_dir,
            debug_tokens=debug_tokens,
            enable_artifacts=enable_artifacts,
            prompt_token_count=prompt_token_count,
            prompt_start_delta=prompt_start_delta,
            max_context_tokens=max_context_tokens or 0,
            collect_memory_stats=collect_memory_stats,
            write_debug_metrics=write_debug_metrics,
            write_debug_response=write_debug_response,
            write_debug_token_texts=write_debug_token_texts,
            write_debug_tokens=write_debug_tokens,
            resume_attempts=resume_attempts,
            max_resume_attempts=max_resume_attempts,
            last_user_text=last_user_text,
            turn_index=generation_index + 1,
            attempt_index=resume_attempts + 1,
        )

        generation_index += 1
        generator.keepalive()
        print(
            f"\n[INFO] Generated {outcome.num_generated_tokens} tokens in {outcome.generation_elapsed:.2f}s ("
            f"{outcome.tokens_per_second:.2f} tokens/s)"
        )

        if outcome.should_resume:
            resume_base_hyperparameters = hyperparameters.copy()
            if outcome.resume_reason == "loop_detected":
                print("[INFO] Response became repetitive; rethinking to complete the final answer...")
            else:
                print("[INFO] Response truncated; rethinking to complete the final answer...")

        if outcome.should_resume and outcome.resume_prompt:
            resume_attempts += 1
            hyperparameters = outcome.resume_hyperparameters or hyperparameters
            pending_resume_prompt = outcome.resume_prompt
            continue

        if resume_base_hyperparameters is not None and not outcome.should_resume:
            hyperparameters = resume_base_hyperparameters
            resume_base_hyperparameters = None
        pending_resume_prompt = None

        should_continue, parsed_messages = run_tools_if_requested(
            generator=generator,
            tokens=outcome.tokens,
            parsed_messages=outcome.parsed_messages,
            tools=tools,
            conversation=conversation,
            hyperparameters=hyperparameters,
        )
        if should_continue:
            tool_iteration += 1
            continue

        tool_calls_detected = has_tool_calls(
            parsed_messages=parsed_messages,
            tools=tools,
        )

        if tool_calls_detected and not outcome.assistant_text.strip():
            pass
        elif outcome.assistant_text or (not generator.is_gpt_oss or not generator.use_harmony):
            parent_id = conversation[-1].get("id") if conversation else None
            message_id = make_message_id()
            assistant_turn = {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": "assistant",
                "content": outcome.assistant_text if outcome.assistant_text else "[No final response - see thinking above]",
                "timestamp": make_timestamp(),
            }
            if hyperparameters and hyperparameters != last_saved_hyperparameters:
                assistant_turn["hyperparameters"] = hyperparameters.copy()
                last_saved_hyperparameters = hyperparameters.copy()
            if (
                generator.is_gpt_oss
                and generator.use_harmony
                and outcome.analysis_text_parts
            ):
                assistant_turn["analysis"] = "\n".join(outcome.analysis_text_parts).lstrip(" \t").rstrip(" \t")
            conversation.append(assistant_turn)
        elif generator.is_gpt_oss and generator.use_harmony and outcome.analysis_text_parts:
            parent_id = conversation[-1].get("id") if conversation else None
            message_id = make_message_id()
            assistant_turn = {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": "assistant",
                "content": "[Analysis only - no final response]",
                "analysis": "\n".join(outcome.analysis_text_parts).lstrip(" \t").rstrip(" \t"),
                "timestamp": make_timestamp(),
            }
            if hyperparameters and hyperparameters != last_saved_hyperparameters:
                assistant_turn["hyperparameters"] = hyperparameters.copy()
                last_saved_hyperparameters = hyperparameters.copy()
            conversation.append(assistant_turn)

        break

    return TurnResult(
        hyperparameters=hyperparameters,
        last_saved_hyperparameters=last_saved_hyperparameters,
        generation_index=generation_index,
        last_prompt_start_time=last_prompt_start_time,
    )
