"""
Standalone token generation implementation for MLX Harmony.

This module provides streaming token generation without depending on mlx-lm.
"""

from collections.abc import Callable, Generator
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_harmony.cache import KVCache, make_prompt_cache
from mlx_harmony.logging import get_logger
from mlx_harmony.runtime.metrics import TimingStats, timer
from mlx_harmony.sampling import apply_logits_processors

logger = get_logger(__name__)


class GenerationResponse:
    """Response object for streaming generation."""

    def __init__(
        self,
        token: int,
        text: str,
        logprobs: mx.array | None = None,
        finish_reason: str | None = None,
    ):
        self.token = token
        self.text = text
        self.logprobs = logprobs
        self.finish_reason = finish_reason


def _prompt_to_tokens(tokenizer, prompt: str | mx.array | list[int]) -> mx.array:
    """Normalize prompt input to an mx.array of token IDs."""
    if isinstance(prompt, mx.array):
        return prompt
    if isinstance(prompt, str):
        if not hasattr(tokenizer, "encode"):
            raise ValueError("Tokenizer must have encode() method")
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt, dtype=mx.uint32)


def _ensure_prompt_cache(
    model: nn.Module,
    prompt_cache: list[KVCache] | None,
) -> list[KVCache]:
    """Create a prompt cache when one is not provided."""
    return make_prompt_cache(model) if prompt_cache is None else prompt_cache


def _resolve_sampler(
    sampler: Callable[[mx.array], mx.array] | None,
) -> Callable[[mx.array], mx.array]:
    """Resolve a sampler, defaulting to greedy argmax."""
    if sampler is not None:
        return sampler

    def default_sampler(logprobs: mx.array) -> mx.array:
        return mx.argmax(logprobs, axis=-1)

    return default_sampler


def _prefill_kv_cache(
    *,
    model: nn.Module,
    prompt_tokens: mx.array,
    prompt_cache: list[KVCache],
    prefill_step_size: int,
    clear_cache: bool,
    clear_cache_interval: int,
    prefill_start_offset: int = 0,
) -> None:
    """Run the prompt through the model to build KV cache state."""
    total_prompt_tokens = len(prompt_tokens)
    logger.info(
        "prefill_kv_cache: total_tokens=%d step_size=%d start_offset=%d clear_cache=%s clear_cache_interval=%d",
        total_prompt_tokens,
        prefill_step_size,
        prefill_start_offset,
        clear_cache,
        clear_cache_interval,
    )
    processed_tokens = min(max(prefill_start_offset, 0), total_prompt_tokens)
    chunk_index = 0
    while processed_tokens < total_prompt_tokens - 1:
        remaining = total_prompt_tokens - processed_tokens - 1
        chunk_size = min(prefill_step_size, remaining)
        chunk = prompt_tokens[processed_tokens: processed_tokens + chunk_size]

        _ = model(chunk[None], cache=prompt_cache)
        cache_keys = []
        for cache in prompt_cache:
            state = cache.state
            if state[0] is not None:
                cache_keys.append(state[0])
        if cache_keys:
            mx.eval(cache_keys)
        if clear_cache and clear_cache_interval > 0:
            if chunk_index % clear_cache_interval == 0:
                mx.clear_cache()

        processed_tokens += chunk_size
        chunk_index += 1


def _log_memory_stats(phase: str) -> None:
    if not hasattr(mx, "metal"):
        return
    try:
        info = mx.metal.device_info()
    except Exception:
        return

    if not isinstance(info, dict):
        logger.info("memory_stats[%s]: %s", phase, info)
        return

    metrics = {
        key: value
        for key, value in info.items()
        if isinstance(value, (int, float))
    }
    if metrics:
        logger.info("memory_stats[%s]: %s", phase, metrics)
    else:
        logger.info("memory_stats[%s]: %s", phase, info)


def _init_decoder(tokenizer) -> tuple[bool, Any, str, int]:
    """Prepare incremental decoding state."""
    use_detokenizer = hasattr(tokenizer, "detokenizer")
    if use_detokenizer:
        return True, tokenizer.detokenizer, "", 0
    return False, None, "", 0


def _decode_next_token(
    *,
    tokenizer,
    token_id: int,
    generated_tokens: list[int],
    use_detokenizer: bool,
    detokenizer: Any,
    last_segment: str,
    last_decode_length: int,
) -> tuple[str, str, int]:
    """Decode the latest token and update incremental state."""
    if use_detokenizer:
        detokenizer.add_token(token_id)
        new_text = detokenizer.last_segment
        return new_text, new_text, last_decode_length
    try:
        full_decoded = tokenizer.decode(generated_tokens)
        new_text = full_decoded[last_decode_length:]
        last_decode_length = len(full_decoded)
    except Exception:
        try:
            new_text = tokenizer.decode([token_id])
        except Exception:
            new_text = ""
    return new_text, last_segment, last_decode_length


def stream_generate(
    model: nn.Module,
    tokenizer,
    prompt: str | mx.array | list[int],
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array] | None = None,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    prompt_cache: list[KVCache] | None = None,
    prefill_start_offset: int = 0,
    prefill_step_size: int = 2048,
    stop_tokens: list[int] | None = None,
    clear_cache: bool = True,
    clear_cache_interval: int = 1,
    log_memory_stats: bool = False,
    log_timing_stats: bool = False,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate tokens from a model in a streaming fashion.

    Args:
        model: MLX model for generation
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt (string, mx.array, or list of token IDs)
        max_tokens: Maximum tokens to generate
        sampler: Optional sampler function (default: greedy/argmax)
        logits_processors: Optional list of logits processors
        prompt_cache: Optional pre-computed KV cache
        prefill_step_size: Chunk size for processing prompt
        stop_tokens: Optional list of token IDs to stop generation

    Yields:
        GenerationResponse objects with token, text, and metadata
    """
    prompt_tokens = _prompt_to_tokens(tokenizer, prompt)
    prompt_cache = _ensure_prompt_cache(model, prompt_cache)
    sampler = _resolve_sampler(sampler)
    timing_stats = TimingStats() if log_timing_stats else None

    if log_memory_stats:
        _log_memory_stats("prefill_start")

    if timing_stats is not None:
        with timer(timing_stats, "prefill"):
            _prefill_kv_cache(
                model=model,
                prompt_tokens=prompt_tokens,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                clear_cache=clear_cache,
                clear_cache_interval=clear_cache_interval,
                prefill_start_offset=prefill_start_offset,
            )
    else:
        _prefill_kv_cache(
            model=model,
            prompt_tokens=prompt_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=prefill_step_size,
            clear_cache=clear_cache,
            clear_cache_interval=clear_cache_interval,
            prefill_start_offset=prefill_start_offset,
        )

    if log_memory_stats:
        _log_memory_stats("prefill_end")

    current_token = prompt_tokens[-1:][None] if len(prompt_tokens) > 0 else None
    generated_tokens: list[int] = []

    use_detokenizer, detokenizer, last_segment, last_decode_length = _init_decoder(
        tokenizer
    )

    for n in range(max_tokens):  # noqa: B007
        if current_token is None:
            break

        if timing_stats is not None:
            with timer(timing_stats, "model"):
                logits = model(current_token, cache=prompt_cache)
                logits = logits[:, -1, :]
        else:
            logits = model(current_token, cache=prompt_cache)
            logits = logits[:, -1, :]

        if generated_tokens:
            all_tokens = mx.concatenate(
                [prompt_tokens, mx.array(generated_tokens, dtype=mx.uint32)]
            )
        else:
            all_tokens = prompt_tokens
        if timing_stats is not None:
            with timer(timing_stats, "logits_processors"):
                logits = apply_logits_processors(
                    tokens=all_tokens,
                    logits=logits,
                    processors=logits_processors,
                )
        else:
            logits = apply_logits_processors(
                tokens=all_tokens,
                logits=logits,
                processors=logits_processors,
            )

        if timing_stats is not None:
            with timer(timing_stats, "sampler"):
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                token_id = int(sampler(logprobs))
        else:
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token_id = int(sampler(logprobs))

        if stop_tokens and token_id in stop_tokens:
            if use_detokenizer:
                text = last_segment
            else:
                if timing_stats is not None:
                    with timer(timing_stats, "decode"):
                        text = tokenizer.decode(generated_tokens) if generated_tokens else ""
                        text = text[last_decode_length:]
                else:
                    text = tokenizer.decode(generated_tokens) if generated_tokens else ""
                    text = text[last_decode_length:]
            yield GenerationResponse(
                token=token_id,
                text=text,
                logprobs=logprobs,
                finish_reason="stop",
            )
            break

        generated_tokens.append(token_id)
        current_token = mx.array([[token_id]], dtype=mx.uint32)

        if timing_stats is not None:
            with timer(timing_stats, "decode"):
                new_text, last_segment, last_decode_length = _decode_next_token(
                    tokenizer=tokenizer,
                    token_id=token_id,
                    generated_tokens=generated_tokens,
                    use_detokenizer=use_detokenizer,
                    detokenizer=detokenizer,
                    last_segment=last_segment,
                    last_decode_length=last_decode_length,
                )
        else:
            new_text, last_segment, last_decode_length = _decode_next_token(
                tokenizer=tokenizer,
                token_id=token_id,
                generated_tokens=generated_tokens,
                use_detokenizer=use_detokenizer,
                detokenizer=detokenizer,
                last_segment=last_segment,
                last_decode_length=last_decode_length,
            )

        is_last_token = (n == max_tokens - 1)
        yield GenerationResponse(
            token=token_id,
            text=new_text,
            logprobs=logprobs,
            finish_reason="length" if is_last_token else None,
        )

        if is_last_token:
            break

        if clear_cache and clear_cache_interval > 0:
            if n > 0 and n % (clear_cache_interval * 256) == 0:
                mx.clear_cache()

    if use_detokenizer:
        detokenizer.finalize()
        last_segment = detokenizer.last_segment

    if log_memory_stats:
        _log_memory_stats("generation_end")

    if timing_stats is not None:
        logger.info("timing_stats: %s", timing_stats.snapshot())

    # Note: We no longer yield a final response here because:
    # 1. The loop already yields all tokens including the last one
    # 2. The last token has finish_reason="length" when we hit max_tokens
    # 3. Yielding again here was causing an extra duplicate token
