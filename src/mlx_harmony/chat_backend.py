from __future__ import annotations

"""Backend adapters for local and server-backed chat execution."""

from mlx_harmony.backend_contract import (
    BackendGenerationRequest,
    BackendResult,
    FrontendBackend,
)
from mlx_harmony.backend_runtime import (
    BackendState,
    build_backend_inputs_from_generation_request,
    execute_backend_turn,
)
from mlx_harmony.chat_utils import transport_fields_from_hyperparameters
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

    transport_fields = transport_fields_from_hyperparameters(request.hyperparameters)
    return GenerationRequest(
        messages=messages,
        temperature=transport_fields["temperature"],
        max_tokens=transport_fields["max_tokens"],
        top_p=transport_fields["top_p"],
        min_p=transport_fields["min_p"],
        top_k=transport_fields["top_k"],
        repetition_penalty=transport_fields["repetition_penalty"],
        repetition_context_size=transport_fields["repetition_context_size"],
        xtc_probability=transport_fields["xtc_probability"],
        xtc_threshold=transport_fields["xtc_threshold"],
        seed=transport_fields["seed"],
        loop_detection=transport_fields["loop_detection"],
        reseed_each_turn=transport_fields["reseed_each_turn"],
    )


class LocalBackend:
    def generate(self, request: BackendGenerationRequest) -> BackendResult:
        """Run a local chat turn using shared backend execution.

        Args:
            request: Backend generation request contract.

        Returns:
            BackendResult containing updated hyperparameters and token counts.
        """
        inputs = build_backend_inputs_from_generation_request(request=request)
        result, updated_state = execute_backend_turn(
            generator=request.generator,
            inputs=inputs,
            state=BackendState(
                last_prompt_start_time=request.last_prompt_start_time,
                generation_index=request.generation_index,
            ),
            last_saved_hyperparameters=request.last_saved_hyperparameters,
            debug_path=request.debug_path,
            make_message_id=request.make_message_id,
            make_timestamp=request.make_timestamp,
            collect_memory_stats=request.collect_memory_stats,
            write_debug_metrics=request.write_debug_metrics,
            write_debug_response=request.write_debug_response,
            write_debug_info=request.write_debug_info,
            write_debug_token_texts=request.write_debug_token_texts,
            write_debug_tokens=request.write_debug_tokens,
            debug_tokens=request.debug_tokens,
            enable_artifacts=request.enable_artifacts,
            max_tool_iterations=request.max_tool_iterations,
            max_resume_attempts=request.max_resume_attempts,
        )
        return BackendResult(
            assistant_text=None,
            analysis_text=None,
            handled_conversation=True,
            hyperparameters=result.hyperparameters or request.hyperparameters,
            last_saved_hyperparameters=(
                result.last_saved_hyperparameters or request.last_saved_hyperparameters
            ),
            generation_index=updated_state.generation_index,
            last_prompt_start_time=updated_state.last_prompt_start_time,
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
