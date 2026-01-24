from __future__ import annotations

from typing import Any, Callable

from mlx_harmony.chat_types import RetryPlan


def build_retry_plan(
    *,
    last_finish_reason: str | None,
    last_stop_reason: str | None,
    repetition_detected: bool,
    resume_attempts: int,
    max_resume_attempts: int,
    last_user_text: str | None,
    hyperparameters: dict[str, float | int | bool | str],
) -> RetryPlan:
    resume_reason: str | None = None
    if last_finish_reason == "length":
        resume_reason = "length"
    elif last_stop_reason == "loop_detected":
        resume_reason = "loop_detected"
    elif repetition_detected:
        resume_reason = "repetitive_text"

    should_retry = resume_reason is not None and resume_attempts < max_resume_attempts
    if not should_retry:
        return RetryPlan(
            should_retry=False,
            reason=None,
            resume_prompt=None,
            resume_hyperparameters=None,
        )

    resume_hyperparameters = hyperparameters.copy()
    current_penalty = float(resume_hyperparameters.get("repetition_penalty") or 1.0)
    if resume_reason == "length":
        penalty_floor = 1.15
    elif resume_reason == "loop_detected":
        penalty_floor = 1.25
    else:
        penalty_floor = 1.2
    resume_hyperparameters["repetition_penalty"] = max(current_penalty, penalty_floor)
    current_context = int(resume_hyperparameters.get("repetition_context_size") or 0)
    if resume_reason == "loop_detected":
        context_floor = 2048
    else:
        context_floor = 1024
    resume_hyperparameters["repetition_context_size"] = max(current_context, context_floor)
    resume_hyperparameters["loop_detection"] = "full"

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
    elif resume_reason == "repetitive_text":
        if list_likely:
            resume_prompt = (
                "Your previous reply became repetitive. Provide a concise final answer from the "
                "beginning. Do not say 'continued' or refer to earlier output. If you use a list, "
                "limit it to at most 8 items and do not repeat any item."
            )
        else:
            resume_prompt = (
                "Your previous reply became repetitive. Provide a concise final answer from the "
                "beginning. Use a short paragraph, avoid repeating phrases, and do not say "
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

    return RetryPlan(
        should_retry=True,
        reason=resume_reason,
        resume_prompt=resume_prompt,
        resume_hyperparameters=resume_hyperparameters,
    )


def apply_retry_to_conversation(
    *,
    conversation: list[dict[str, Any]],
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
    resume_prompt: str,
    last_saved_hyperparameters: dict[str, float | int | bool | str],
    hyperparameters: dict[str, float | int | bool | str],
) -> dict[str, float | int | bool | str]:
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
    return last_saved_hyperparameters
