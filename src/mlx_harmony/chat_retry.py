from __future__ import annotations

from typing import Any, Callable

from mlx_harmony.chat_types import RetryPlan


def build_retry_plan(
    *,
    last_finish_reason: str | None,
    last_stop_reason: str | None,
    repetition_detected: bool,
    analysis_only: bool,
    resume_attempts: int,
    max_resume_attempts: int,
    last_user_text: str | None,
    hyperparameters: dict[str, float | int | bool | str],
    base_hyperparameters: dict[str, float | int | bool | str] | None,
) -> RetryPlan:
    resume_reason: str | None = None
    if last_finish_reason == "length":
        resume_reason = "length"
    elif last_stop_reason == "loop_detected":
        resume_reason = "loop_detected"
    elif analysis_only:
        resume_reason = "analysis_only"
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

    resume_hyperparameters = (
        base_hyperparameters.copy() if base_hyperparameters is not None else hyperparameters.copy()
    )
    current_max_tokens_raw = resume_hyperparameters.get("max_tokens")
    current_max_tokens: int | None = None
    if isinstance(current_max_tokens_raw, (int, float)):
        current_max_tokens = int(current_max_tokens_raw)
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
    if resume_reason == "length" and current_max_tokens:
        bumped_tokens = int(current_max_tokens * 1.25)
        resume_hyperparameters["max_tokens"] = max(current_max_tokens, bumped_tokens)

    if resume_reason == "analysis_only":
        resume_prompt = (
            "Your previous reply did not include a final answer. "
            "Please provide the final response now. "
            "Do not refer to earlier output."
        )
    elif resume_reason == "loop_detected":
        resume_prompt = (
            "Your previous reply was repetitive or unreadable. "
            "Please provide the complete answer again from the beginning. "
            "Do not refer to earlier output."
        )
    elif resume_reason == "repetitive_text":
        resume_prompt = (
            "Your previous reply was repetitive or unreadable. "
            "Please provide the complete answer again from the beginning. "
            "Do not refer to earlier output."
        )
    else:
        resume_prompt = (
            "Your previous reply was incomplete or unreadable. "
            "Please provide the complete answer again from the beginning. "
            "If needed, be slightly more concise to fit the available space. "
            "Do not refer to earlier output."
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
