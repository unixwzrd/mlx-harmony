from __future__ import annotations

from typing import Any, Callable, Protocol

from unicodefix.transforms import clean_text

from mlx_harmony.chat_generation import stream_generation
from mlx_harmony.chat_harmony import parse_harmony_response
from mlx_harmony.chat_types import ParsedOutput


class ModelAdapter(Protocol):
    def stream(
        self,
        *,
        generator: Any,
        conversation: list[dict[str, Any]],
        system_message: str | None,
        prompt_token_ids: list[int] | None,
        hyperparameters: dict[str, float | int | bool | str],
        seed: int | None,
        on_text: Callable[[str], None],
    ) -> tuple[list[int], list[int], list[str]]: ...

    def decode_raw(
        self,
        *,
        generator: Any,
        prompt_token_ids: list[int] | None,
        all_generated_tokens: list[int],
    ) -> str: ...

    def parse(
        self,
        *,
        generator: Any,
        tokens: list[int],
        streamed_text_parts: list[str],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        display_assistant: Callable[..., None],
        display_thinking: Callable[..., None],
        truncate_text: Callable[[str, int], str],
        suppress_display: bool,
    ) -> ParsedOutput: ...


class HarmonyAdapter:
    def stream(
        self,
        *,
        generator: Any,
        conversation: list[dict[str, Any]],
        system_message: str | None,
        prompt_token_ids: list[int] | None,
        hyperparameters: dict[str, float | int | bool | str],
        seed: int | None,
        on_text: Callable[[str], None],
    ) -> tuple[list[int], list[int], list[str]]:
        return stream_generation(
            generator=generator,
            conversation=conversation,
            system_message=system_message,
            prompt_token_ids=prompt_token_ids,
            hyperparameters=hyperparameters,
            seed=seed,
            on_text=None,
        )

    def decode_raw(
        self,
        *,
        generator: Any,
        prompt_token_ids: list[int] | None,
        all_generated_tokens: list[int],
    ) -> str:
        if generator.encoding:
            decode_utf8 = getattr(generator.encoding, "decode_utf8", None)
            if callable(decode_utf8):
                return decode_utf8(all_generated_tokens)
            return generator.encoding.decode(all_generated_tokens)
        return generator.tokenizer.decode(all_generated_tokens)

    def parse(
        self,
        *,
        generator: Any,
        tokens: list[int],
        streamed_text_parts: list[str],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        display_assistant: Callable[..., None],
        display_thinking: Callable[..., None],
        truncate_text: Callable[[str, int], str],
        suppress_display: bool,
    ) -> ParsedOutput:
        parse_result = parse_harmony_response(
            generator=generator,
            tokens=tokens,
            streamed_text_parts=streamed_text_parts,
            assistant_name=assistant_name,
            thinking_limit=thinking_limit or 0,
            response_limit=response_limit or 0,
            render_markdown=render_markdown,
            debug=debug,
            display_assistant=display_assistant,
            display_thinking=display_thinking,
            truncate_text=truncate_text,
            suppress_display=suppress_display,
        )
        channels: dict[str, str] = {}
        if parse_result.analysis_text_parts:
            channels["analysis"] = "\n".join(parse_result.analysis_text_parts)
        if parse_result.assistant_text:
            channels["final"] = parse_result.assistant_text
        return ParsedOutput(
            channels=channels,
            assistant_text=parse_result.assistant_text,
            analysis_parts=parse_result.analysis_text_parts,
            parsed_messages=parse_result.parsed_messages,
        )


class NativeAdapter:
    def stream(
        self,
        *,
        generator: Any,
        conversation: list[dict[str, Any]],
        system_message: str | None,
        prompt_token_ids: list[int] | None,
        hyperparameters: dict[str, float | int | bool | str],
        seed: int | None,
        on_text: Callable[[str], None],
    ) -> tuple[list[int], list[int], list[str]]:
        def handle_token(token_id: int) -> None:
            text = generator.tokenizer.decode([token_id])
            text = clean_text(text)
            on_text(text)

        return stream_generation(
            generator=generator,
            conversation=conversation,
            system_message=system_message,
            prompt_token_ids=prompt_token_ids,
            hyperparameters=hyperparameters,
            seed=seed,
            on_text=handle_token,
        )

    def decode_raw(
        self,
        *,
        generator: Any,
        prompt_token_ids: list[int] | None,
        all_generated_tokens: list[int],
    ) -> str:
        return generator.tokenizer.decode(all_generated_tokens)

    def parse(
        self,
        *,
        generator: Any,
        tokens: list[int],
        streamed_text_parts: list[str],
        assistant_name: str,
        thinking_limit: int | None,
        response_limit: int | None,
        render_markdown: bool,
        debug: bool,
        display_assistant: Callable[..., None],
        display_thinking: Callable[..., None],
        truncate_text: Callable[[str, int], str],
        suppress_display: bool,
    ) -> ParsedOutput:
        assistant_text = clean_text("".join(streamed_text_parts))
        if response_limit:
            assistant_text = truncate_text(assistant_text, response_limit)
        if assistant_text and not suppress_display:
            display_assistant(assistant_text, assistant_name, render_markdown)
        return ParsedOutput(
            channels={"final": assistant_text} if assistant_text else {},
            assistant_text=assistant_text,
            analysis_parts=[],
            parsed_messages=None,
        )


def get_adapter(generator: Any) -> ModelAdapter:
    if getattr(generator, "is_gpt_oss", False) and getattr(generator, "use_harmony", False):
        return HarmonyAdapter()
    return NativeAdapter()
