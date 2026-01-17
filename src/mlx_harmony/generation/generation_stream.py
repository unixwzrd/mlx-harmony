from __future__ import annotations

from typing import Any, Callable

from unicodefix.transforms import clean_text


def stream_generation(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
    prompt_token_ids: list[int] | None,
    hyperparameters: dict[str, float | int | bool],
    seed: int | None,
    on_text: Callable[[str], None],
    on_harmony_text: Callable[[str, str | None, object | None], None] | None = None,
) -> tuple[list[int], list[int], list[str]]:
    tokens: list[int] = []
    all_generated_tokens: list[int] = []
    streamed_text_parts: list[str] = []

    for token_id in generator.generate(
        prompt_tokens=prompt_token_ids,
        messages=conversation,
        temperature=hyperparameters.get("temperature"),
        max_tokens=hyperparameters.get("max_tokens"),
        top_p=hyperparameters.get("top_p"),
        min_p=hyperparameters.get("min_p"),
        top_k=hyperparameters.get("top_k"),
        repetition_penalty=hyperparameters.get("repetition_penalty"),
        repetition_context_size=hyperparameters.get("repetition_context_size"),
        system_message=system_message,
        seed=seed,
    ):
        token_int = int(token_id)
        tokens.append(token_int)
        all_generated_tokens.append(token_int)

        if generator.is_gpt_oss and generator.use_harmony and generator.streamable_parser:
            try:
                generator.streamable_parser.process(int(token_id))
                if on_harmony_text and generator.streamable_parser.last_content_delta:
                    on_harmony_text(
                        generator.streamable_parser.last_content_delta,
                        generator.streamable_parser.current_channel,
                        generator.streamable_parser.current_role,
                    )
            except Exception:
                # Streaming parser errors are handled later in chat.py
                pass
        else:
            text = generator.tokenizer.decode([int(token_id)])
            text = clean_text(text)
            on_text(text)
            streamed_text_parts.append(text)

    return tokens, all_generated_tokens, streamed_text_parts
