"""Shared backend turn preparation and execution helpers.

This module keeps request-to-turn wiring in one place so multiple entrypoints
can run the same backend generation path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from mlx_harmony.api_contract import ChatRequest
from mlx_harmony.backend_api import build_conversation_from_messages, run_backend_chat
from mlx_harmony.chat_utils import (
    build_hyperparameters_from_request,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
)
from mlx_harmony.config import PromptConfig, load_prompt_config
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.runtime.model_init import initialize_generator


@dataclass(frozen=True)
class BackendInputs:
    """All normalized backend inputs for one generation turn."""

    conversation: list[dict[str, Any]]
    hyperparameters: dict[str, float | int | bool | str]
    max_context_tokens: int | None
    last_user_text: str | None
    assistant_name: str
    thinking_limit: int | None
    response_limit: int | None
    tools: list[Any]
    render_markdown: bool


@dataclass(frozen=True)
class BackendState:
    """Mutable backend counters represented as immutable state snapshots."""

    last_prompt_start_time: float | None
    generation_index: int


class _InfoLogger(Protocol):
    """Logger protocol for model runtime cache operations."""

    def info(self, msg: str, *args: Any) -> None:
        """Log info-level messages."""


def load_runtime_generator(
    *,
    model_path: str,
    prompt_config_path: str | None,
    prompt_config: PromptConfig | None = None,
    lazy: bool = False,
    mlock: bool | None = None,
    logger: _InfoLogger | None = None,
) -> TokenGenerator:
    """Load a TokenGenerator using one shared lifecycle policy.

    Args:
        model_path: Model path to load.
        prompt_config_path: Optional prompt config path.
        prompt_config: Optional already-loaded prompt config object.
        lazy: Whether model weights should be loaded lazily.
        mlock: Optional explicit mlock override.
        logger: Optional logger for load lifecycle messages.

    Returns:
        Initialized TokenGenerator.
    """

    prompt_cfg = prompt_config
    if prompt_cfg is None and prompt_config_path:
        loaded_prompt_cfg = load_prompt_config(prompt_config_path)
        if loaded_prompt_cfg is not None and not isinstance(loaded_prompt_cfg, PromptConfig):
            prompt_cfg = PromptConfig.model_validate(loaded_prompt_cfg)
        else:
            prompt_cfg = loaded_prompt_cfg

    effective_mlock = mlock if mlock is not None else bool(getattr(prompt_cfg, "mlock", False))
    if logger is not None:
        logger.info("Loading model: %s (prompt_config=%s)", model_path, prompt_config_path or "none")
    return initialize_generator(
        model_path=model_path,
        prompt_config=prompt_cfg,
        prompt_config_path=prompt_config_path,
        lazy=lazy,
        mlock=effective_mlock,
    )


class GeneratorRuntimeCache:
    """Thread-safe runtime cache for loaded generator/model metadata."""

    def __init__(self, *, logger: _InfoLogger) -> None:
        self._logger = logger
        self._generator: TokenGenerator | None = None
        self._generator_prompt_config_path: str | None = None
        self._loaded_model_path: str | None = None
        self._loaded_at_unix: int | None = None
        self._lock = Lock()

    def get_generator(self, model: str, prompt_config_path: str | None) -> TokenGenerator:
        """Load or reuse the cached generator for the requested model/config."""

        with self._lock:
            if (
                self._generator is None
                or self._generator.model_path != model
                or self._generator_prompt_config_path != prompt_config_path
            ):
                self._generator = load_runtime_generator(
                    model_path=model,
                    prompt_config_path=prompt_config_path,
                    prompt_config=None,
                    lazy=False,
                    mlock=None,
                    logger=self._logger,
                )
                self._generator_prompt_config_path = prompt_config_path
                self._loaded_model_path = model
                self._loaded_at_unix = int(time.time())
            return self._generator

    def get_loaded_model_path(self) -> str | None:
        """Return currently loaded model path, if any."""

        return self._loaded_model_path

    def get_loaded_prompt_config_path(self) -> str | None:
        """Return currently loaded prompt config path, if any."""

        return self._generator_prompt_config_path

    def get_loaded_at_unix(self) -> int | None:
        """Return loaded timestamp, if model has been loaded."""

        return self._loaded_at_unix

    def is_model_loaded(self) -> bool:
        """Return whether a generator has been loaded."""

        return self._generator is not None


class TurnRuntimeState:
    """Thread-safe runtime state for chat turn counters/timestamps."""

    def __init__(self) -> None:
        self._state = BackendState(last_prompt_start_time=None, generation_index=0)
        self._lock = Lock()

    def read(self) -> BackendState:
        """Return the latest backend turn state snapshot."""

        with self._lock:
            return self._state

    def write(self, state: BackendState) -> None:
        """Store a new backend turn state snapshot."""

        with self._lock:
            self._state = state


def collect_mlx_memory_stats(*, enabled: bool) -> dict[str, float | int | str]:
    """Collect MLX Metal device stats when enabled.

    Args:
        enabled: Whether memory collection is enabled for this run.

    Returns:
        Flat memory stats dict suitable for debug artifacts.
    """

    if not enabled:
        return {}
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, float | int | str] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats


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


def build_backend_inputs_from_generation_request(*, request: Any) -> BackendInputs:
    """Build backend input payload from shared frontend backend request contract.

    Args:
        request: BackendGenerationRequest-compatible object from frontend flow.

    Returns:
        BackendInputs ready for `execute_backend_turn`.
    """

    return BackendInputs(
        conversation=request.conversation,
        hyperparameters=request.hyperparameters,
        max_context_tokens=request.max_context_tokens,
        last_user_text=request.last_user_text or None,
        assistant_name=request.assistant_name,
        thinking_limit=request.thinking_limit,
        response_limit=request.response_limit,
        tools=request.tools,
        render_markdown=request.render_markdown,
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
    debug_tokens: str | None = None,
    enable_artifacts: bool = True,
    max_tool_iterations: int = 10,
    max_resume_attempts: int = 2,
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
        debug_tokens: Optional token debug mode.
        enable_artifacts: Whether prompt/completion artifacts are written.
        max_tool_iterations: Maximum tool iterations for this turn.
        max_resume_attempts: Maximum resume attempts for this turn.
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
        debug_tokens=debug_tokens,
        enable_artifacts=enable_artifacts,
        max_context_tokens=inputs.max_context_tokens,
        max_tool_iterations=max_tool_iterations,
        max_resume_attempts=max_resume_attempts,
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
