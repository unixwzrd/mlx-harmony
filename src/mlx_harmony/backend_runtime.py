"""Shared backend turn preparation and execution helpers.

This module keeps request-to-turn wiring in one place so multiple entrypoints
can run the same backend generation path.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from mlx_harmony.api_contract import ChatRequest
from mlx_harmony.backend_api import build_conversation_from_messages, run_backend_chat
from mlx_harmony.chat_utils import (
    build_hyperparameters_from_request,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
)


@dataclass(frozen=True)
class BackendInputs:
    """All normalized backend inputs for one generation turn."""

    conversation: list[dict[str, Any]]
    hyperparameters: dict[str, float | int | bool | str]
    max_context_tokens: int
    last_user_text: str | None
    assistant_name: str
    thinking_limit: int
    response_limit: int
    tools: list[Any]
    render_markdown: bool


@dataclass(frozen=True)
class BackendState:
    """Mutable backend counters represented as immutable state snapshots."""

    last_prompt_start_time: float | None
    generation_index: int


def prepare_backend_inputs(
    *,
    request: ChatRequest,
    generator: Any,
    model_path: str,
    profile_data: dict[str, object] | None,
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
) -> BackendInputs:
    """Build shared backend inputs for stream and non-stream execution.

    Args:
        request: Parsed chat request payload.
        generator: Loaded token generator.
        model_path: Resolved model path for this request.
        profile_data: Optional profile metadata used during config resolution.
        make_message_id: Callback used to create stable message IDs.
        make_timestamp: Callback used to create message timestamps.

    Returns:
        BackendInputs ready to pass into `execute_backend_turn`.
    """

    prompt_config = getattr(generator, "prompt_config", None)
    conversation = build_conversation_from_messages(
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        prompt_config=prompt_config,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
    )
    hyperparameters = build_hyperparameters_from_request(
        request=request,
        prompt_config=prompt_config,
        is_harmony=bool(
            getattr(generator, "is_gpt_oss", False) and getattr(generator, "use_harmony", False)
        ),
    )
    max_context_tokens = resolve_max_context_tokens(
        args=SimpleNamespace(max_context_tokens=None),
        loaded_max_context_tokens=None,
        loaded_model_path=None,
        prompt_config=prompt_config,
        profile_data=profile_data,
        model_path=model_path,
    )
    last_user_text = None
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
    return BackendInputs(
        conversation=conversation,
        hyperparameters=hyperparameters,
        max_context_tokens=max_context_tokens,
        last_user_text=last_user_text,
        assistant_name=get_assistant_name(prompt_config),
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        tools=[],
        render_markdown=False,
    )


def execute_backend_turn(
    *,
    generator: Any,
    inputs: BackendInputs,
    state: BackendState,
    last_saved_hyperparameters: dict[str, float | int | bool | str] | None,
    debug_path: Any,
    make_message_id: Callable[[], str],
    make_timestamp: Callable[[], str],
    collect_memory_stats: Callable[[], dict[str, Any]],
    write_debug_metrics: Any,
    write_debug_response: Any,
    write_debug_info: Any,
    write_debug_token_texts: Any,
    write_debug_tokens: Any,
    run_backend_chat_fn: Any = run_backend_chat,
) -> tuple[Any, BackendState]:
    """Execute one backend turn and return updated generation state.

    Args:
        generator: Loaded token generator.
        inputs: Prepared backend inputs for this request.
        state: Current backend generation state.
        last_saved_hyperparameters: Last saved hyperparameter state.
        debug_path: Debug log file path.
        make_message_id: Callback used to create stable message IDs.
        make_timestamp: Callback used to create message timestamps.
        collect_memory_stats: Callback used for memory instrumentation.
        write_debug_metrics: Debug writer for metrics.
        write_debug_response: Debug writer for responses.
        write_debug_info: Debug writer for metadata.
        write_debug_token_texts: Debug writer for token text artifacts.
        write_debug_tokens: Debug writer for token artifacts.
        run_backend_chat_fn: Callable that executes a backend chat turn.

    Returns:
        Tuple of (`BackendChatResult`, updated `BackendState`).
    """

    result = run_backend_chat_fn(
        generator=generator,
        conversation=inputs.conversation,
        hyperparameters=inputs.hyperparameters,
        last_saved_hyperparameters=last_saved_hyperparameters,
        assistant_name=inputs.assistant_name,
        thinking_limit=inputs.thinking_limit,
        response_limit=inputs.response_limit,
        render_markdown=inputs.render_markdown,
        debug_path=debug_path,
        debug_tokens=None,
        enable_artifacts=True,
        max_context_tokens=inputs.max_context_tokens,
        max_tool_iterations=10,
        max_resume_attempts=2,
        tools=inputs.tools,
        last_user_text=inputs.last_user_text,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        collect_memory_stats=collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_info=write_debug_info,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        last_prompt_start_time=state.last_prompt_start_time,
        generation_index=state.generation_index,
    )
    return result, BackendState(
        last_prompt_start_time=result.last_prompt_start_time,
        generation_index=result.generation_index,
    )
