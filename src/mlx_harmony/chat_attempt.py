from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from unicodefix.transforms import clean_text

from mlx_harmony.chat_retry import build_retry_plan
from mlx_harmony.chat_types import GenerationOutcome


def run_generation_attempt(
    *,
    generator: Any,
    adapter: Any,
    prompt_conversation: list[dict[str, Any]],
    prompt_token_ids: list[int] | None,
    raw_prompt: str,
    system_message: str | None,
    hyperparameters: dict[str, float | int | bool | str],
    seed: int | None,
    on_text: Callable[[str], None],
    assistant_name: str,
    thinking_limit: int | None,
    response_limit: int | None,
    render_markdown: bool,
    debug: bool,
    display_assistant: Callable[..., None],
    display_thinking: Callable[..., None],
    truncate_text: Callable[[str, int], str],
    debug_path: Any,
    logs_dir: Path | None,
    debug_tokens: str | None,
    enable_artifacts: bool,
    prompt_token_count: int,
    prompt_start_delta: float | None,
    max_context_tokens: int,
    collect_memory_stats: Callable[[], dict[str, Any]],
    write_debug_metrics: Callable[..., None],
    write_debug_response: Callable[..., None],
    write_debug_token_texts: Callable[..., None],
    write_debug_tokens: Callable[..., None],
    resume_attempts: int,
    max_resume_attempts: int,
    last_user_text: str | None,
    turn_index: int,
    attempt_index: int,
) -> GenerationOutcome:
    generation_start = time.perf_counter()
    tokens, all_generated_tokens, streamed_text_parts = adapter.stream(
        generator=generator,
        conversation=prompt_conversation,
        system_message=system_message,
        prompt_token_ids=prompt_token_ids,
        hyperparameters=hyperparameters,
        seed=seed,
        on_text=on_text,
    )
    generation_end = time.perf_counter()
    generation_elapsed = generation_end - generation_start
    num_generated_tokens = len(tokens)
    tokens_per_second = (
        num_generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
    )

    memory_stats = collect_memory_stats()
    write_debug_metrics(
        debug_path=debug_path,
        metrics={
            "prompt_tokens": prompt_token_count,
            "generated_tokens": num_generated_tokens,
            "elapsed_seconds": generation_elapsed,
            "tokens_per_second": tokens_per_second,
            "prompt_start_to_prompt_start_seconds": prompt_start_delta,
            "max_context_tokens": max_context_tokens,
            "prefill_start_offset": getattr(generator, "_last_prefill_start_offset", None),
            **memory_stats,
        },
    )

    write_debug_tokens(
        debug_path=debug_path,
        token_ids=all_generated_tokens,
        decode_tokens=generator.encoding.decode if generator.encoding else None,
        label="response",
        mode=debug_tokens or "off",
    )

    raw_response = adapter.decode_raw(
        generator=generator,
        prompt_token_ids=prompt_token_ids,
        all_generated_tokens=all_generated_tokens,
    )

    cleaned_response = clean_text(raw_response)
    repetition_detected = _detect_text_repetition(cleaned_response)
    retry_plan = build_retry_plan(
        last_finish_reason=generator.last_finish_reason,
        last_stop_reason=generator.last_stop_reason,
        repetition_detected=repetition_detected,
        resume_attempts=resume_attempts,
        max_resume_attempts=max_resume_attempts,
        last_user_text=last_user_text,
        hyperparameters=hyperparameters,
    )
    write_debug_response(
        debug_path=debug_path,
        raw_response=raw_response,
        cleaned_response=cleaned_response,
        show_console=debug and not retry_plan.should_retry,
    )
    write_debug_token_texts(
        debug_path=debug_path,
        token_ids=all_generated_tokens,
        decode_token=generator.encoding.decode if generator.encoding else generator.tokenizer.decode,
        label="response",
        mode=debug_tokens or "off",
    )

    parse_result = adapter.parse(
        generator=generator,
        tokens=tokens,
        streamed_text_parts=streamed_text_parts,
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
        debug=debug,
        display_assistant=display_assistant,
        display_thinking=display_thinking,
        truncate_text=truncate_text,
        suppress_display=retry_plan.should_retry,
    )

    _write_attempt_artifacts(
        logs_dir=logs_dir,
        raw_prompt=raw_prompt,
        prompt_token_ids=prompt_token_ids,
        all_generated_tokens=all_generated_tokens,
        raw_response=raw_response,
        cleaned_response=cleaned_response,
        channels=parse_result.channels,
        retry_plan=retry_plan,
        hyperparameters=hyperparameters,
        generator=generator,
        max_context_tokens=max_context_tokens,
        turn_index=turn_index,
        attempt_index=attempt_index,
        enabled=enable_artifacts,
    )

    return GenerationOutcome(
        tokens=tokens,
        all_generated_tokens=all_generated_tokens,
        streamed_text_parts=streamed_text_parts,
        assistant_text=parse_result.assistant_text,
        analysis_text_parts=parse_result.analysis_parts,
        parsed_messages=parse_result.parsed_messages,
        raw_response=raw_response,
        cleaned_response=cleaned_response,
        should_resume=retry_plan.should_retry,
        resume_reason=retry_plan.reason,
        resume_prompt=retry_plan.resume_prompt,
        resume_hyperparameters=retry_plan.resume_hyperparameters,
        generation_elapsed=generation_elapsed,
        tokens_per_second=tokens_per_second,
        num_generated_tokens=num_generated_tokens,
        memory_stats=memory_stats,
    )


def _detect_text_repetition(text: str) -> bool:
    if not text:
        return False
    normalized_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped in {"---", "***", "___"}:
            continue
        if stripped.startswith(("-", "*", "+")):
            stripped = stripped.lstrip("-*+").strip()
        elif stripped[:2].isdigit() and stripped[1] == ".":
            stripped = stripped[2:].strip()
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1].isdigit() and stripped[2] == ".":
            stripped = stripped[3:].strip()
        if stripped:
            normalized_lines.append(stripped.lower())
    normalized_text = " ".join(normalized_lines)
    tokens = normalized_text.split()
    if len(tokens) < 120:
        return False
    window = 240
    tail = tokens[-window:] if len(tokens) > window else tokens
    unique_ratio = len(set(tail)) / max(len(tail), 1)
    ngram_sizes = (6, 8)
    for ngram_size in ngram_sizes:
        if len(tail) < ngram_size * 3:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for idx in range(len(tail) - ngram_size + 1):
            gram = tuple(tail[idx : idx + ngram_size])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= 4 and unique_ratio < 0.65:
                return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 6:
        line_counts: dict[str, int] = {}
        for line in lines[-30:]:
            if len(line) < 40:
                continue
            line_counts[line] = line_counts.get(line, 0) + 1
            if line_counts[line] >= 3 and unique_ratio < 0.7:
                return True
    return False


def _write_attempt_artifacts(
    *,
    logs_dir: Path | None,
    raw_prompt: str,
    prompt_token_ids: list[int] | None,
    all_generated_tokens: list[int],
    raw_response: str,
    cleaned_response: str,
    channels: dict[str, str],
    retry_plan: Any,
    hyperparameters: dict[str, float | int | bool | str],
    generator: Any,
    max_context_tokens: int,
    turn_index: int,
    attempt_index: int,
    enabled: bool,
) -> None:
    if not enabled or logs_dir is None:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"turn{turn_index:03d}.attempt{attempt_index}"

    prompt_full_path = logs_dir / f"prompt.full.{stem}.txt"
    prompt_tokens_path = logs_dir / f"prompt.tokens.{stem}.json"
    completion_tokens_path = logs_dir / f"completion.tokens.{stem}.json"
    completion_raw_path = logs_dir / f"completion.raw.{stem}.txt"
    completion_clean_path = logs_dir / f"completion.cleaned.{stem}.txt"
    parse_channels_path = logs_dir / f"parse.channels.{stem}.json"
    retry_path = logs_dir / f"retry.decision.{stem}.json"

    with open(prompt_full_path, "w", encoding="utf-8") as handle:
        handle.write(raw_prompt)

    prompt_meta = {
        "prompt_token_ids": prompt_token_ids or [],
        "model_path": getattr(generator, "model_path", None),
        "encoding": getattr(getattr(generator, "encoding", None), "name", None),
        "max_context_tokens": max_context_tokens,
    }
    with open(prompt_tokens_path, "w", encoding="utf-8") as handle:
        json.dump(prompt_meta, handle, ensure_ascii=False, indent=2)

    with open(completion_tokens_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"completion_token_ids": all_generated_tokens},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with open(completion_raw_path, "w", encoding="utf-8") as handle:
        handle.write(raw_response)

    with open(completion_clean_path, "w", encoding="utf-8") as handle:
        handle.write(cleaned_response)

    with open(parse_channels_path, "w", encoding="utf-8") as handle:
        json.dump(channels, handle, ensure_ascii=False, indent=2)

    retry_payload = {
        "should_retry": retry_plan.should_retry,
        "reason": retry_plan.reason,
        "resume_prompt": retry_plan.resume_prompt,
        "resume_hyperparameters": retry_plan.resume_hyperparameters,
        "hyperparameters": hyperparameters,
    }
    with open(retry_path, "w", encoding="utf-8") as handle:
        json.dump(retry_payload, handle, ensure_ascii=False, indent=2)
