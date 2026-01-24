from __future__ import annotations

from typing import Any, Callable

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def stream_generation(
    *,
    generator: Any,
    conversation: list[dict[str, Any]],
    system_message: str | None,
    prompt_token_ids: list[int] | None,
    hyperparameters: dict[str, float | int | bool | str],
    seed: int | None,
    on_text: Callable[[int], None] | None,
) -> tuple[list[int], list[int], list[str]]:
    tokens: list[int] = []
    all_generated_tokens: list[int] = []
    streamed_text_parts: list[str] = []
    loop_detection = str(hyperparameters.get("loop_detection") or "cheap").lower()
    loop_sizes = (8, 16, 32)
    repeat_token_count = 16
    low_var_window = 64
    low_var_unique_max = 4

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
        loop_detection=hyperparameters.get("loop_detection"),
        system_message=system_message,
        seed=seed,
    ):
        token_int = int(token_id)
        tokens.append(token_int)
        all_generated_tokens.append(token_int)

        if on_text is not None:
            on_text(token_int)

        if loop_detection != "off":
            if len(tokens) >= repeat_token_count:
                recent = tokens[-repeat_token_count:]
                if len(set(recent)) == 1:
                    logger.warning(
                        "loop_detected: repeated_single_token count=%d tokens=%d",
                        repeat_token_count,
                        len(tokens),
                    )
                    generator.last_finish_reason = "stop"
                    generator.last_stop_reason = "loop_detected"
                    break

            if loop_detection == "full":
                for loop_size in loop_sizes:
                    if len(tokens) >= loop_size * 2:
                        if tokens[-loop_size:] == tokens[-2 * loop_size:-loop_size]:
                            logger.warning(
                                "loop_detected: repeat_size=%d tokens=%d",
                                loop_size,
                                len(tokens),
                            )
                            generator.last_finish_reason = "stop"
                            generator.last_stop_reason = "loop_detected"
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
                        generator.last_stop_reason = "loop_detected"
                        break

    return tokens, all_generated_tokens, streamed_text_parts
