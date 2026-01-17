"""
Standalone token generation implementation for MLX Harmony.

This module provides streaming token generation without depending on mlx-lm.
"""

import time
from collections.abc import Callable, Generator

import mlx.core as mx
import mlx.nn as nn

from mlx_harmony.generation.cache import KVCache, make_prompt_cache


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
    # Convert prompt to token array
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Encode string to tokens
            if hasattr(tokenizer, "encode"):
                prompt = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                raise ValueError("Tokenizer must have encode() method")
        prompt = mx.array(prompt, dtype=mx.uint32)

    # Create or use provided cache
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)

    # Default sampler (greedy)
    def default_sampler(logprobs: mx.array) -> mx.array:
        return mx.argmax(logprobs, axis=-1)

    if sampler is None:
        sampler = default_sampler

    # Process prompt in chunks (prefill phase)
    prompt_tokens = prompt
    total_prompt_tokens = len(prompt_tokens)

    # Process prompt in chunks to build KV cache
    processed_tokens = 0
    while processed_tokens < total_prompt_tokens - 1:
        remaining = total_prompt_tokens - processed_tokens - 1
        chunk_size = min(prefill_step_size, remaining)
        chunk = prompt_tokens[processed_tokens:processed_tokens + chunk_size]

        # Forward pass through model to build cache
        _ = model(chunk[None], cache=prompt_cache)
        mx.eval([c.state[0] for c in prompt_cache if c.state[0] is not None])
        mx.clear_cache()

        processed_tokens += chunk_size

    # Now generate tokens autoregressively
    current_token = prompt_tokens[-1:][None] if len(prompt_tokens) > 0 else None
    generated_tokens: list[int] = []
    start_time = time.perf_counter()

    # Track decoded text (for incremental decoding)
    # Use tokenizer's detokenizer if available for better incremental decoding
    use_detokenizer = hasattr(tokenizer, "detokenizer")
    if use_detokenizer:
        detokenizer = tokenizer.detokenizer
        last_segment = ""
    else:
        detokenizer = None
        decoded_text = ""
        last_decode_length = 0

    for n in range(max_tokens):  # noqa: B007
        if current_token is None:
            break

        # Forward pass: get logits for next token
        logits = model(current_token, cache=prompt_cache)
        logits = logits[:, -1, :]  # Get logits for last position

        # Apply logits processors (repetition penalty, bias, etc.)
        if logits_processors:
            # Build token sequence for processors (all tokens seen so far)
            if generated_tokens:
                all_tokens = mx.concatenate([prompt_tokens, mx.array(generated_tokens, dtype=mx.uint32)])
            else:
                all_tokens = prompt_tokens
            for processor in logits_processors:
                logits = processor(all_tokens, logits)

        # Convert to log probabilities
        logprobs = logits - mx.logsumexp(logits, keepdims=True)

        # Sample next token
        next_token = sampler(logprobs)
        token_id = int(next_token.item())

        # Check for stop tokens (simple single-token check)
        # TODO: Support multi-token stop sequences
        if stop_tokens and token_id in stop_tokens:
            # Yield final response with stop reason (don't include stop token)
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

        # Add to generated tokens
        generated_tokens.append(token_id)
        current_token = mx.array([[token_id]], dtype=mx.uint32)

        # Decode incrementally to get new text segment
        if use_detokenizer:
            # Use streaming detokenizer for better incremental decoding
            detokenizer.add_token(token_id)
            new_text = detokenizer.last_segment
            last_segment = new_text
        else:
            # Fallback: decode incrementally
            try:
                full_decoded = tokenizer.decode(generated_tokens)
                new_text = full_decoded[last_decode_length:]
                last_decode_length = len(full_decoded)
            except Exception:
                # Fallback: decode just the new token if available
                try:
                    new_text = tokenizer.decode([token_id])
                except Exception:
                    new_text = ""

        # Yield response
        yield GenerationResponse(
            token=token_id,
            text=new_text,
            logprobs=logprobs,
            finish_reason=None,
        )

        # Periodic cache clearing
        if n > 0 and n % 256 == 0:
            mx.clear_cache()

    # Finalize detokenizer if using one
    if use_detokenizer:
        detokenizer.finalize()
        last_segment = detokenizer.last_segment

    # Final response if we hit max_tokens
    if len(generated_tokens) > 0 and n == max_tokens - 1:
        if use_detokenizer:
            final_text = last_segment
        else:
            final_text = tokenizer.decode(generated_tokens)
            final_text = final_text[last_decode_length:] if 'last_decode_length' in locals() else final_text
        yield GenerationResponse(
            token=generated_tokens[-1],
            text=final_text,
            logprobs=logprobs if 'logprobs' in locals() else None,
            finish_reason="length",
        )
