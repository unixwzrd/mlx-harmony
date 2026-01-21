from __future__ import annotations

from typing import Any, Callable

from openai_harmony import Role, StreamableParser
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
) -> tuple[list[int], list[int], list[str]]:
    tokens: list[int] = []
    all_generated_tokens: list[int] = []
    streamed_text_parts: list[str] = []

    if (
        generator.is_gpt_oss
        and generator.use_harmony
        and generator.streamable_parser
        and generator.encoding is not None
    ):
        parser = generator.streamable_parser
        if hasattr(parser, "reset"):
            parser.reset()
        elif hasattr(parser, "reset_state"):
            parser.reset_state()
        else:
            parser = StreamableParser(generator.encoding, Role.ASSISTANT, strict=False)
            generator.streamable_parser = parser

        if prompt_token_ids:
            start_token_id = 200006
            last_start_idx = -1
            for idx in range(len(prompt_token_ids) - 1, -1, -1):
                if prompt_token_ids[idx] == start_token_id:
                    last_start_idx = idx
                    break
            if last_start_idx >= 0:
                for token_id in prompt_token_ids[last_start_idx:]:
                    try:
                        parser.process(int(token_id))
                    except Exception:
                        # If prompt tokens are malformed, parsing will be handled later.
                        break

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
            except Exception:
                # Streaming parser errors are handled later in chat.py
                pass
        else:
            text = generator.tokenizer.decode([int(token_id)])
            text = clean_text(text)
            on_text(text)
            streamed_text_parts.append(text)

    return tokens, all_generated_tokens, streamed_text_parts
