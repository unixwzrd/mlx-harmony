"""
Standalone token generation implementation for MLX Harmony.

This module provides streaming token generation without depending on mlx-lm.
"""

from collections.abc import Callable, Generator
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_harmony.cache import KVCache, make_prompt_cache


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
) -> None:
    """Run the prompt through the model to build KV cache state."""
    total_prompt_tokens = len(prompt_tokens)
    processed_tokens = 0
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
        mx.clear_cache()

        processed_tokens += chunk_size


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
    prefill_step_size: int = 2048,
    stop_tokens: list[int] | None = None,
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

    _prefill_kv_cache(
        model=model,
        prompt_tokens=prompt_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
    )

    current_token = prompt_tokens[-1:][None] if len(prompt_tokens) > 0 else None
    generated_tokens: list[int] = []

    use_detokenizer, detokenizer, last_segment, last_decode_length = _init_decoder(
        tokenizer
    )

    for n in range(max_tokens):  # noqa: B007
        if current_token is None:
            break

        logits = model(current_token, cache=prompt_cache)
        logits = logits[:, -1, :]

        if logits_processors:
            if generated_tokens:
                all_tokens = mx.concatenate(
                    [prompt_tokens, mx.array(generated_tokens, dtype=mx.uint32)]
                )
            else:
                all_tokens = prompt_tokens
            for processor in logits_processors:
                logits = processor(all_tokens, logits)

        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token_id = int(sampler(logprobs))

        if stop_tokens and token_id in stop_tokens:
            if use_detokenizer:
                text = last_segment
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

        if n > 0 and n % 256 == 0:
            mx.clear_cache()

    if use_detokenizer:
        detokenizer.finalize()
        last_segment = detokenizer.last_segment

    # Note: We no longer yield a final response here because:
    # 1. The loop already yields all tokens including the last one
    # 2. The last token has finish_reason="length" when we hit max_tokens
    # 3. Yielding again here was causing an extra duplicate token
