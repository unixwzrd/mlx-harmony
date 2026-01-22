from __future__ import annotations

from typing import Any, Callable

from openai_harmony import Role, StreamableParser
from unicodefix.transforms import clean_text

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


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
    final_boundary_detected = False
    last_message_count = 0
    loop_sizes = (8, 16, 32)
    repeat_token_count = 16
    low_var_window = 64
    low_var_unique_max = 4
    analysis_char_limit = None
    analysis_char_count = 0

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
        last_message_count = len(parser.messages)
        analysis_char_limit = (
            generator.prompt_config.truncate_thinking
            if getattr(generator, "prompt_config", None)
            and generator.prompt_config.truncate_thinking is not None
            else None
        )

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
            if generator.streamable_parser.messages:
                current_count = len(generator.streamable_parser.messages)
                if current_count > last_message_count:
                    last_message_count = current_count
                    last_msg = generator.streamable_parser.messages[-1]
                    channel = getattr(last_msg, "channel", None)
                    if channel == "final":
                        final_boundary_detected = True
                        generator.last_finish_reason = "stop"
                        break
                    if channel in ("tool", "tool_call"):
                        generator.last_finish_reason = "stop"
                        break
            current_channel = getattr(generator.streamable_parser, "current_channel", None)
            if (
                analysis_char_limit is not None
                and current_channel in ("analysis", "commentary")
                and generator.encoding is not None
            ):
                try:
                    analysis_char_count += len(generator.encoding.decode([token_int]))
                except Exception:
                    analysis_char_count += 1
                if analysis_char_count >= analysis_char_limit:
                    logger.warning(
                        "analysis_budget_reached: chars=%d limit=%d tokens=%d",
                        analysis_char_count,
                        analysis_char_limit,
                        len(tokens),
                    )
                    generator.last_finish_reason = "length"
                    if hasattr(generator, "last_stop_reason"):
                        generator.last_stop_reason = "analysis_budget"
                    break
        else:
            text = generator.tokenizer.decode([int(token_id)])
            text = clean_text(text)
            on_text(text)
            streamed_text_parts.append(text)

        if len(tokens) >= repeat_token_count:
            recent = tokens[-repeat_token_count:]
            if len(set(recent)) == 1:
                logger.warning(
                    "loop_detected: repeated_single_token count=%d tokens=%d",
                    repeat_token_count,
                    len(tokens),
                )
                generator.last_finish_reason = "stop"
                break

        for loop_size in loop_sizes:
            if len(tokens) >= loop_size * 2:
                if tokens[-loop_size:] == tokens[-2 * loop_size:-loop_size]:
                    logger.warning(
                        "loop_detected: repeat_size=%d tokens=%d",
                        loop_size,
                        len(tokens),
                    )
                    generator.last_finish_reason = "stop"
                    break
        else:
            loop_size = None

        if loop_size is not None:
            break

        if len(tokens) >= low_var_window:
            if len(set(tokens[-low_var_window:])) <= low_var_unique_max:
                logger.warning(
                    "loop_detected: low_variance window=%d unique_max=%d tokens=%d",
                    low_var_window,
                    low_var_unique_max,
                    len(tokens),
                )
                generator.last_finish_reason = "stop"
                break

    if (
        generator.is_gpt_oss
        and generator.use_harmony
        and not final_boundary_detected
        and isinstance(hyperparameters.get("max_tokens"), int)
        and hyperparameters["max_tokens"] > 0
        and len(tokens) >= hyperparameters["max_tokens"]
    ):
        logger.warning(
            "harmony_final_boundary_missing: tokens=%d max_tokens=%d",
            len(tokens),
            hyperparameters["max_tokens"],
        )

    return tokens, all_generated_tokens, streamed_text_parts
