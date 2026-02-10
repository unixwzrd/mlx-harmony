"""Backend adapter contract shared by local and HTTP generation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendResult:
    """Normalized backend response consumed by the shared chat frontend.

    Attributes:
        assistant_text: Final assistant text when backend does not mutate the
            conversation in-place.
        analysis_text: Optional analysis channel text.
        handled_conversation: Whether backend already appended assistant turns.
        hyperparameters: Updated runtime hyperparameters after generation.
        last_saved_hyperparameters: Updated persisted hyperparameter snapshot.
        generation_index: Updated generation counter.
        last_prompt_start_time: Updated prompt start timestamp.
        prompt_tokens: Prompt token usage for this turn.
        completion_tokens: Completion token usage for this turn.
    """

    assistant_text: str | None
    analysis_text: str | None
    handled_conversation: bool
    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    generation_index: int
    last_prompt_start_time: float | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class BackendGenerationRequest:
    """Request payload for backend adapter generation.

    This captures the complete frontend-to-backend contract so adapters can
    remain transport-specific and thin.
    """

    conversation: list[dict[str, Any]]
    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    last_user_text: str
    max_context_tokens: int | None
    last_prompt_start_time: float | None
    generation_index: int
    max_tool_iterations: int
    max_resume_attempts: int
    generator: Any
    tools: list[Any]
    assistant_name: str
    thinking_limit: int | None
    response_limit: int | None
    render_markdown: bool
    debug: bool
    debug_path: Any
    debug_tokens: str | None
    enable_artifacts: bool
    make_message_id: Any
    make_timestamp: Any
    display_assistant: Any
    display_thinking: Any
    truncate_text: Any
    collect_memory_stats: Any
    write_debug_metrics: Any
    write_debug_response: Any
    write_debug_info: Any
    write_debug_token_texts: Any
    write_debug_tokens: Any


class FrontendBackend(Protocol):
    """Adapter interface used by the shared frontend loop."""

    def generate(self, request: BackendGenerationRequest) -> BackendResult:
        """Execute one generation turn for the provided request."""
        ...

