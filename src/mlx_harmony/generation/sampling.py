"""
Standalone sampling utilities for MLX Harmony.

This module provides sampling functions (temperature, top_p, top_k, min_p, etc.)
and logits processors (repetition penalty, logit bias) without depending on mlx-lm.
"""

from collections.abc import Callable

import mlx.core as mx


def sample_temperature(logprobs: mx.array, temperature: float) -> mx.array:
    """
    Apply temperature sampling to log probabilities.

    Args:
        logprobs: Log probabilities of shape (vocab_size,)
        temperature: Temperature value (1.0 = no change, >1.0 = more random, <1.0 = more deterministic)

    Returns:
        Modified log probabilities
    """
    if temperature <= 0.0:
        # Greedy decoding (deterministic)
        return logprobs
    return logprobs / temperature


def sample_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    """
    Keep only the top-k log probabilities, mask others with -inf.

    Implementation matches mlx-lm's apply_top_k using argpartition and put_along_axis.

    Args:
        logprobs: Log probabilities of shape (vocab_size,)
        top_k: Number of top tokens to keep (0 = disabled)

    Returns:
        Modified log probabilities with top-k masking
    """
    if top_k <= 0:
        return logprobs

    vocab_size = logprobs.shape[-1]
    if top_k >= vocab_size:
        return logprobs

    # Use argpartition to find indices of tokens NOT in top-k (to mask)
    # argpartition(-logprobs, kth=top_k-1) partitions so that indices [..., top_k:] are the bottom tokens
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]

    # Set masked tokens to -inf using put_along_axis
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


def sample_top_p(logprobs: mx.array, top_p: float, min_tokens_to_keep: int = 1) -> mx.array:
    """
    Nucleus sampling: keep tokens with cumulative probability <= top_p.

    Implementation based on mlx-lm's apply_top_p, which doesn't require searchsorted.
    Uses take_along_axis/put_along_axis pattern to rearrange probabilities.

    Args:
        logprobs: Log probabilities of shape (vocab_size,)
        top_p: Cumulative probability threshold (0.0-1.0, 1.0 = disabled)
        min_tokens_to_keep: Minimum number of tokens to keep regardless of top_p

    Returns:
        Modified log probabilities with top-p masking
    """
    if top_p >= 1.0:
        return logprobs

    # Convert logprobs to probs
    probs = mx.exp(logprobs)

    # Sort in ascending order (by logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Calculate cumulative probabilities along sorted axis
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange cumulative probs back to original order
    # Create inverse mapping: for each original position, find its sorted position
    vocab_size = sorted_indices.shape[-1]
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(vocab_size, dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # Select tokens where cumulative_probs > (1 - top_p)
    # mlx-lm keeps tokens where cumulative_probs > (1 - top_p)
    # This keeps the highest probability tokens (nucleus sampling)
    threshold = 1.0 - top_p

    # Base mask: keep tokens where cumulative_probs > threshold (matching mlx-lm)
    mask = cumulative_probs > threshold

    # Ensure we keep at least min_tokens_to_keep tokens
    # Get top min_tokens_to_keep by logprobs (highest logprobs = lowest indices when sorted ascending)
    if min_tokens_to_keep > 0 and min_tokens_to_keep < vocab_size:
        # Get indices of top min_tokens_to_keep tokens (highest logprobs)
        top_indices = mx.argsort(logprobs, axis=-1)[-min_tokens_to_keep:]
        # Create mask for these top tokens
        # MLX's zeros_like doesn't accept dtype, create zero array and cast to bool
        min_mask = mx.zeros_like(logprobs).astype(mx.bool_)
        # Create ones array and cast to bool for put_along_axis
        ones_bool = mx.ones(min_tokens_to_keep).astype(mx.bool_)
        min_mask = mx.put_along_axis(
            min_mask,
            top_indices,
            ones_bool,
            axis=-1,
        )
        # Combine masks: keep tokens that pass top_p OR are in top min_tokens_to_keep
        mask = mx.logical_or(mask, min_mask)

    # Mask out non-selected tokens (return -inf for masked tokens)
    return mx.where(mask, logprobs, mx.array(float("-inf")))


def sample_min_p(logprobs: mx.array, min_p: float) -> mx.array:
    """
    Filter tokens with probability < min_p relative to the maximum probability.

    Args:
        logprobs: Log probabilities of shape (vocab_size,)
        min_p: Minimum probability threshold (0.0-1.0, 0.0 = disabled)

    Returns:
        Modified log probabilities with min-p filtering
    """
    if min_p <= 0.0:
        return logprobs

    # Convert to probabilities
    probs = mx.exp(logprobs)
    # Find maximum probability
    max_prob = mx.max(probs)
    # Threshold: keep tokens with prob >= min_p * max_prob
    threshold = min_p * max_prob
    mask = probs >= threshold

    # Mask out tokens below threshold
    return mx.where(mask, logprobs, mx.array(float("-inf")))


def sample_xtc(
    logprobs: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: list[int],
) -> mx.array:
    """
    Apply XTC (Experimental Token Control) sampling to log probabilities.

    Args:
        logprobs: Log probabilities of shape (vocab_size,)
        xtc_probability: Probability of applying XTC sampling (0.0-1.0, 0.0 = disabled)
        xtc_threshold: Threshold for probabilities to be sampled (0.0-0.5)
        xtc_special_tokens: List of special token IDs to exclude from XTC sampling

    Returns:
        Modified log probabilities with XTC masking
    """
    if xtc_probability <= 0.0:
        return logprobs

    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`xtc_threshold` must be in [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`xtc_probability` must be in [0, 1] interval, but is {xtc_probability}"
        )

    # Convert to probabilities
    probs = mx.softmax(logprobs, axis=-1)

    # Find minimum probability above threshold
    # Match mlx-lm's approach: use where to avoid boolean indexing
    # where(probs > threshold, probs, inf).min() gives us min of probs above threshold
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.array(float("inf"))).min()

    # Exclude special tokens from mask
    # mlx-lm uses mask[..., xtc_special_tokens] = False which works with array indexing
    if xtc_special_tokens:
        # Filter to valid token IDs and convert to array
        valid_token_ids = [tid for tid in xtc_special_tokens if 0 <= tid < len(mask)]
        if valid_token_ids:
            token_indices = mx.array(valid_token_ids)
            # Use array indexing to set mask values to False (MLX supports this)
            mask[..., token_indices] = False

    # Apply XTC: if random value > xtc_probability, use original logprobs,
    # otherwise mask tokens that don't meet threshold
    rand_val = mx.random.uniform(0.0, 1.0)
    if rand_val > xtc_probability:
        # Don't apply XTC, return original
        return logprobs
    else:
        # Apply XTC: mask tokens not meeting threshold
        return mx.where(mask, logprobs, mx.array(float("-inf")))


def make_sampler(
    temp: float = 1.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> Callable[[mx.array], mx.array]:
    """
    Create a sampler function that applies temperature, top_p, min_p, top_k, and XTC sampling.

    Args:
        temp: Sampling temperature (1.0 = no change, 0.0 = greedy)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        min_p: Minimum probability threshold (0.0 = disabled)
        top_k: Top-k sampling (0 = disabled)
        min_tokens_to_keep: Minimum tokens to keep for top_p
        xtc_probability: Probability of applying XTC sampling (0.0 = disabled)
        xtc_threshold: Threshold for XTC sampling (0.0-0.5)
        xtc_special_tokens: List of special token IDs to exclude from XTC

    Returns:
        A function that takes logprobs and returns sampled token index
    """
    if xtc_special_tokens is None:
        xtc_special_tokens = []

    def sampler(logprobs: mx.array) -> mx.array:
        # Greedy decoding if temp is 0
        if temp <= 0.0:
            return mx.argmax(logprobs, axis=-1)

        # Apply sampling filters first (top_p, min_p, xtc, top_k)
        # These modify logprobs by setting filtered tokens to -inf
        if top_p < 1.0:
            logprobs = sample_top_p(logprobs, top_p, min_tokens_to_keep)

        if min_p > 0.0:
            logprobs = sample_min_p(logprobs, min_p)

        if xtc_probability > 0.0:
            logprobs = sample_xtc(logprobs, xtc_probability, xtc_threshold, xtc_special_tokens)

        if top_k > 0:
            logprobs = sample_top_k(logprobs, top_k)

        # Apply temperature: multiply by (1/temp) which is equivalent to dividing by temp
        # This matches mlx-lm's categorical_sampling exactly
        # mx.random.categorical will handle normalization internally
        return mx.random.categorical(logprobs * (1.0 / temp))

    return sampler


def repetition_penalty_processor(
    repetition_penalty: float,
    repetition_context_size: int,
) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Create a logits processor that applies repetition penalty.

    Args:
        repetition_penalty: Penalty value (1.0 = no penalty, >1.0 = penalize repetition)
        repetition_context_size: Number of previous tokens to consider for penalty

    Returns:
        A function that takes (tokens, logits) and returns modified logits
    """
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if repetition_penalty == 1.0 or len(tokens) == 0:
            return logits

        # Get recent tokens for penalty calculation
        context_tokens = tokens[-repetition_context_size:] if len(tokens) > repetition_context_size else tokens

        # Apply penalty: reduce logits for tokens that appear in recent context
        if isinstance(context_tokens, mx.array):
            context_ids = context_tokens.flatten().tolist()
        else:
            context_ids = list(context_tokens)

        for token_id in set(context_ids):
            token_id = int(token_id)
            if token_id < logits.shape[-1]:
                idx = (0, token_id) if logits.ndim == 2 else (token_id,)
                value = logits[idx]
                logits[idx] = mx.where(
                    value > 0,
                    value / repetition_penalty,
                    value * repetition_penalty,
                )

        return logits

    return processor


def logit_bias_processor(logit_bias: dict[int, float]) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Create a logits processor that applies logit bias.

    Args:
        logit_bias: Dictionary mapping token_id -> bias value (added to logits)

    Returns:
        A function that takes (tokens, logits) and returns modified logits
    """
    if not logit_bias:
        # Return no-op processor
        def processor(tokens: mx.array, logits: mx.array) -> mx.array:
            return logits
        return processor

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        # Apply bias to specified token IDs
        for token_id, bias in logit_bias.items():
            if 0 <= token_id < logits.shape[-1]:
                idx = (0, token_id) if logits.ndim == 2 else (token_id,)
                logits[idx] = logits[idx] + bias
        return logits

    return processor


def make_logits_processors(
    logit_bias: dict[int, float] | None = None,
    repetition_penalty: float | None = None,
    repetition_context_size: int = 20,
) -> list[Callable[[mx.array, mx.array], mx.array]]:
    """
    Create a list of logits processors to apply in sequence.

    Args:
        logit_bias: Optional dictionary mapping token_id -> bias
        repetition_penalty: Optional repetition penalty (1.0 = no penalty)
        repetition_context_size: Context size for repetition penalty

    Returns:
        List of processor functions
    """
    processors = []

    # Apply logit bias first (if specified)
    if logit_bias:
        processors.append(logit_bias_processor(logit_bias))

    # Apply repetition penalty (if specified and != 1.0)
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(repetition_penalty_processor(repetition_penalty, repetition_context_size))

    return processors
