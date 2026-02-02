from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mlx_harmony.chat_turn import run_chat_turn
from mlx_harmony.generation.client import GenerationClient, GenerationRequest
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackendResult:
    assistant_text: str | None
    analysis_text: str | None
    handled_conversation: bool
    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    generation_index: int
    last_prompt_start_time: float | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class FrontendBackend(Protocol):
    def generate(
        self,
        *,
        conversation: list[dict[str, Any]],
        hyperparameters: dict[str, float | int | bool | str],
        last_saved_hyperparameters: dict[str, float | int | bool | str],
        last_user_text: str,
        max_context_tokens: int | None,
        last_prompt_start_time: float | None,
        generation_index: int,
        max_tool_iterations: int,
        max_resume_attempts: int,
        generator: Any,
        tools: list[Any],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        debug_path: Any,
        debug_tokens: str | None,
        enable_artifacts: bool,
        make_message_id: Any,
        make_timestamp: Any,
        display_assistant: Any,
        display_thinking: Any,
        truncate_text: Any,
        collect_memory_stats: Any,
        write_debug_metrics: Any,
        write_debug_response: Any,
        write_debug_token_texts: Any,
        write_debug_tokens: Any,
    ) -> BackendResult:
        ...


class LocalBackend:
    def generate(
        self,
        *,
        conversation: list[dict[str, Any]],
        hyperparameters: dict[str, float | int | bool | str],
        last_saved_hyperparameters: dict[str, float | int | bool | str],
        last_user_text: str,
        max_context_tokens: int | None,
        last_prompt_start_time: float | None,
        generation_index: int,
        max_tool_iterations: int,
        max_resume_attempts: int,
        generator: Any,
        tools: list[Any],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        debug_path: Any,
        debug_tokens: str | None,
        enable_artifacts: bool,
        make_message_id: Any,
        make_timestamp: Any,
        display_assistant: Any,
        display_thinking: Any,
        truncate_text: Any,
        collect_memory_stats: Any,
        write_debug_metrics: Any,
        write_debug_response: Any,
        write_debug_token_texts: Any,
        write_debug_tokens: Any,
    ) -> BackendResult:
        result = run_chat_turn(
            generator=generator,
            conversation=conversation,
            hyperparameters=hyperparameters,
            last_saved_hyperparameters=last_saved_hyperparameters,
            assistant_name=assistant_name,
            thinking_limit=thinking_limit,
            response_limit=response_limit,
            render_markdown=render_markdown,
            debug=debug,
            debug_path=debug_path,
            debug_tokens=debug_tokens,
            enable_artifacts=enable_artifacts,
            max_context_tokens=max_context_tokens,
            max_tool_iterations=max_tool_iterations,
            max_resume_attempts=max_resume_attempts,
            tools=tools,
            last_user_text=last_user_text,
            make_message_id=make_message_id,
            make_timestamp=make_timestamp,
            display_assistant=display_assistant,
            display_thinking=display_thinking,
            truncate_text=truncate_text,
            collect_memory_stats=collect_memory_stats,
            write_debug_metrics=write_debug_metrics,
            write_debug_response=write_debug_response,
            write_debug_token_texts=write_debug_token_texts,
            write_debug_tokens=write_debug_tokens,
            last_prompt_start_time=last_prompt_start_time,
            generation_index=generation_index,
        )
        return BackendResult(
            assistant_text=None,
            analysis_text=None,
            handled_conversation=True,
            hyperparameters=result.hyperparameters,
            last_saved_hyperparameters=result.last_saved_hyperparameters,
            generation_index=result.generation_index,
            last_prompt_start_time=result.last_prompt_start_time,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )


class ServerBackend:
    def __init__(self, client: GenerationClient) -> None:
        self._client = client

    def generate(
        self,
        *,
        conversation: list[dict[str, Any]],
        hyperparameters: dict[str, float | int | bool | str],
        last_saved_hyperparameters: dict[str, float | int | bool | str],
        last_user_text: str,
        max_context_tokens: int | None,
        last_prompt_start_time: float | None,
        generation_index: int,
        max_tool_iterations: int,
        max_resume_attempts: int,
        generator: Any,
        tools: list[Any],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        debug_path: Any,
        debug_tokens: str | None,
        enable_artifacts: bool,
        make_message_id: Any,
        make_timestamp: Any,
        display_assistant: Any,
        display_thinking: Any,
        truncate_text: Any,
        collect_memory_stats: Any,
        write_debug_metrics: Any,
        write_debug_response: Any,
        write_debug_token_texts: Any,
        write_debug_tokens: Any,
    ) -> BackendResult:
        messages = []
        for message in conversation:
            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                continue
            messages.append({"role": role, "content": content})

        request = GenerationRequest(
            messages=messages,
            temperature=float(hyperparameters.get("temperature", 1.0)),
            max_tokens=int(hyperparameters.get("max_tokens", 512)),
            top_p=float(hyperparameters.get("top_p", 0.0)),
            min_p=float(hyperparameters.get("min_p", 0.0)),
            top_k=int(hyperparameters.get("top_k", 0)),
            repetition_penalty=float(hyperparameters.get("repetition_penalty", 0.0)),
            repetition_context_size=int(
                hyperparameters.get("repetition_context_size", 20)
            ),
        )
        response = self._client.generate(request)
        return BackendResult(
            assistant_text=response.text,
            analysis_text=response.analysis_text,
            handled_conversation=False,
            hyperparameters=hyperparameters,
            last_saved_hyperparameters=last_saved_hyperparameters,
            generation_index=generation_index + 1,
            last_prompt_start_time=last_prompt_start_time,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
