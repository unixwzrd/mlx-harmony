from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedOutput:
    channels: dict[str, str]
    assistant_text: str
    analysis_parts: list[str]
    parsed_messages: Any | None = None


@dataclass(frozen=True)
class GenerationArtifacts:
    tokens: list[int]
    all_generated_tokens: list[int]
    streamed_text_parts: list[str]
    raw_response: str
    cleaned_response: str
    prompt_token_count: int
    prompt_conversation: list[dict[str, Any]]
    prompt_token_ids: list[int] | None


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    reason: str | None = None


@dataclass(frozen=True)
class RetryPlan:
    should_retry: bool
    reason: str | None
    resume_prompt: str | None
    resume_hyperparameters: dict[str, float | int | bool | str] | None


@dataclass(frozen=True)
class GenerationOutcome:
    tokens: list[int]
    all_generated_tokens: list[int]
    streamed_text_parts: list[str]
    assistant_text: str
    analysis_text_parts: list[str]
    parsed_messages: Any | None
    raw_response: str
    cleaned_response: str
    should_resume: bool
    resume_reason: str | None
    resume_prompt: str | None
    resume_hyperparameters: dict[str, float | int | bool | str] | None
    generation_elapsed: float
    tokens_per_second: float
    num_generated_tokens: int
    memory_stats: dict[str, Any]


@dataclass(frozen=True)
class TurnResult:
    hyperparameters: dict[str, float | int | bool | str]
    last_saved_hyperparameters: dict[str, float | int | bool | str]
    generation_index: int
    last_prompt_start_time: float | None
