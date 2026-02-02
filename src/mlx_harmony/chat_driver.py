from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from mlx_harmony.chat_io import read_user_input
from mlx_harmony.chat_render import display_assistant, display_thinking
from mlx_harmony.chat_utils import parse_command
from mlx_harmony.generation.client import GenerationClient, GenerationRequest, GenerationResult


@dataclass
class DriverConfig:
    assistant_name: str = "Assistant"
    render_markdown: bool = True
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 0.0
    repetition_context_size: int = 20


def _build_request(
    config: DriverConfig,
    messages: list[dict[str, str]],
    hyperparameters: dict[str, float | int | bool | str],
) -> GenerationRequest:
    return GenerationRequest(
        messages=messages,
        temperature=float(hyperparameters.get("temperature", config.temperature)),
        max_tokens=int(hyperparameters.get("max_tokens", config.max_tokens)),
        top_p=float(hyperparameters.get("top_p", config.top_p)),
        min_p=float(hyperparameters.get("min_p", config.min_p)),
        top_k=int(hyperparameters.get("top_k", config.top_k)),
        repetition_penalty=float(
            hyperparameters.get("repetition_penalty", config.repetition_penalty)
        ),
        repetition_context_size=int(
            hyperparameters.get("repetition_context_size", config.repetition_context_size)
        ),
    )


def run_prompt_stream(
    *,
    client: GenerationClient,
    prompts: Iterable[str],
    config: DriverConfig,
    hyperparameters: dict[str, float | int | bool | str] | None = None,
    on_result: Callable[[str, GenerationResult], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> int:
    conversation: list[dict[str, str]] = []
    params: dict[str, float | int | bool | str] = hyperparameters or {}

    for prompt in prompts:
        user_text = prompt.strip()
        if not user_text:
            continue
        conversation.append({"role": "user", "content": user_text})
        request = _build_request(config, conversation, params)
        try:
            result = client.generate(request)
        except Exception as exc:  # noqa: BLE001
            if on_error:
                on_error(user_text, exc)
            raise
        if result.analysis_text:
            display_thinking(result.analysis_text, config.render_markdown)
        display_assistant(result.text, config.assistant_name, config.render_markdown)
        if on_result:
            on_result(user_text, result)
        conversation.append({"role": "assistant", "content": result.text})
    return 0


def run_interactive(
    *,
    client: GenerationClient,
    config: DriverConfig,
    hyperparameters: dict[str, float | int | bool | str] | None = None,
    on_result: Callable[[str, GenerationResult], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> int:
    conversation: list[dict[str, str]] = []
    params: dict[str, float | int | bool | str] = hyperparameters or {}

    while True:
        try:
            user_input = read_user_input("\n>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() == "q":
            break
        handled, should_apply, message, updates = parse_command(user_input, params)
        if handled:
            if message:
                print(message)
            if should_apply and updates:
                params.update(updates)
            continue
        user_text = user_input.strip()
        if not user_text:
            continue
        conversation.append({"role": "user", "content": user_text})
        request = _build_request(config, conversation, params)
        try:
            result = client.generate(request)
        except Exception as exc:  # noqa: BLE001
            if on_error:
                on_error(user_text, exc)
            raise
        if result.analysis_text:
            display_thinking(result.analysis_text, config.render_markdown)
        display_assistant(result.text, config.assistant_name, config.render_markdown)
        if on_result:
            on_result(user_text, result)
        conversation.append({"role": "assistant", "content": result.text})
    return 0
