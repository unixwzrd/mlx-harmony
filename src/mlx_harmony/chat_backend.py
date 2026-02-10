from __future__ import annotations

"""Backend adapters for local and server-backed chat execution."""

from mlx_harmony.backend_api import run_backend_chat
from mlx_harmony.backend_contract import (
    BackendGenerationRequest,
    BackendResult,
    FrontendBackend,
)
from mlx_harmony.generation.client import GenerationClient, GenerationRequest


def build_server_generation_request(request: BackendGenerationRequest) -> GenerationRequest:
    """Translate shared backend contract data into HTTP generation payload.

    Args:
        request: Shared backend generation request from the frontend loop.

    Returns:
        Transport-level generation request for HTTP client execution.
    """

    messages: list[dict[str, str]] = []
    for message in request.conversation:
        role = message.get("role")
        content = message.get("content")
        if not role or content is None:
            continue
        messages.append({"role": role, "content": content})

    return GenerationRequest(
        messages=messages,
        temperature=float(request.hyperparameters.get("temperature", 1.0)),
        max_tokens=int(request.hyperparameters.get("max_tokens", 512)),
        top_p=float(request.hyperparameters.get("top_p", 0.0)),
        min_p=float(request.hyperparameters.get("min_p", 0.0)),
        top_k=int(request.hyperparameters.get("top_k", 0)),
        repetition_penalty=float(request.hyperparameters.get("repetition_penalty", 0.0)),
        repetition_context_size=int(request.hyperparameters.get("repetition_context_size", 20)),
    )


class LocalBackend:
    def generate(self, request: BackendGenerationRequest) -> BackendResult:
        """Run a local chat turn using shared backend execution.

        Args:
            request: Backend generation request contract.

        Returns:
            BackendResult containing updated hyperparameters and token counts.
        """
        result = run_backend_chat(
            generator=request.generator,
            conversation=request.conversation,
            hyperparameters=request.hyperparameters,
            last_saved_hyperparameters=request.last_saved_hyperparameters,
            assistant_name=request.assistant_name,
            thinking_limit=request.thinking_limit,
            response_limit=request.response_limit,
            render_markdown=request.render_markdown,
            debug_path=request.debug_path,
            debug_tokens=request.debug_tokens,
            enable_artifacts=request.enable_artifacts,
            max_context_tokens=request.max_context_tokens,
            max_tool_iterations=request.max_tool_iterations,
            max_resume_attempts=request.max_resume_attempts,
            tools=request.tools,
            last_user_text=request.last_user_text,
            make_message_id=request.make_message_id,
            make_timestamp=request.make_timestamp,
            collect_memory_stats=request.collect_memory_stats,
            write_debug_metrics=request.write_debug_metrics,
            write_debug_response=request.write_debug_response,
            write_debug_info=request.write_debug_info,
            write_debug_token_texts=request.write_debug_token_texts,
            write_debug_tokens=request.write_debug_tokens,
            last_prompt_start_time=request.last_prompt_start_time,
            generation_index=request.generation_index,
        )
        return BackendResult(
            assistant_text=None,
            analysis_text=None,
            handled_conversation=True,
            hyperparameters=result.hyperparameters or request.hyperparameters,
            last_saved_hyperparameters=(
                result.last_saved_hyperparameters or request.last_saved_hyperparameters
            ),
            generation_index=result.generation_index,
            last_prompt_start_time=result.last_prompt_start_time,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )


class ServerBackend:
    def __init__(self, client: GenerationClient) -> None:
        self._client = client

    def generate(self, request: BackendGenerationRequest) -> BackendResult:
        """Run a chat turn by forwarding to a remote server.

        Args:
            request: Backend generation request contract.

        Returns:
            BackendResult containing assistant text and token counts.
        """
        generation_request = build_server_generation_request(request)
        response = self._client.generate(request=generation_request)
        return BackendResult(
            assistant_text=response.text,
            analysis_text=response.analysis_text,
            handled_conversation=False,
            hyperparameters=request.hyperparameters,
            last_saved_hyperparameters=request.last_saved_hyperparameters,
            generation_index=request.generation_index + 1,
            last_prompt_start_time=request.last_prompt_start_time,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
