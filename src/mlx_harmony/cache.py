"""
Standalone KV cache implementation for MLX Harmony.

This module provides KV cache classes for efficient autoregressive generation
without depending on mlx-lm.
"""

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """
    Simple KV cache for autoregressive generation.

    Stores keys and values from attention layers to avoid recomputing them
    during token generation.
    """

    step = 256  # Allocate cache in chunks of this size

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return concatenated cache.

        Args:
            keys: New keys of shape (B, n_kv_heads, S, head_dim)
            values: New values of shape (B, n_kv_heads, S, head_dim)

        Returns:
            Tuple of (all_keys, all_values) concatenated from cache
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            # Need to allocate more space
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            # Allocate in chunks
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                # Trim to exact size if not aligned to step boundary
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                # Concatenate new allocation
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        # Store new keys/values
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values

        # Return only the used portion
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    def __len__(self) -> int:
        """Return the number of cached tokens."""
        return self.offset

    def is_trimmable(self) -> bool:
        """Check if cache can be trimmed (always True for KVCache)."""
        return True

    def trim(self, n: int) -> int:
        """
        Trim n tokens from the start of the cache.

        Args:
            n: Number of tokens to trim

        Returns:
            Actual number of tokens trimmed
        """
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        """Get current cache state (keys, values)."""
        if self.keys is None:
            return None, None
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]):
        """Set cache state from (keys, values)."""
        if v is None or v[0] is None:
            self.keys = None
            self.values = None
            self.offset = 0
        else:
            self.keys, self.values = v
            self.offset = self.keys.shape[2]

    def __bool__(self) -> bool:
        """Always return True to allow 'cache or make_cache()' pattern."""
        return True


def make_prompt_cache(model: nn.Module, max_kv_size: int | None = None) -> list[KVCache]:
    """
    Create a KV cache for a model.

    Args:
        model: MLX model (must have .layers attribute)
        max_kv_size: Optional maximum cache size (not used in basic KVCache,
                     but reserved for future RotatingKVCache support)

    Returns:
        List of KVCache objects, one per model layer
    """
    # If model has its own make_cache method, use it
    if hasattr(model, "make_cache"):
        return model.make_cache()

    # Otherwise, create one cache per layer
    num_layers = len(model.layers) if hasattr(model, "layers") else 1
    return [KVCache() for _ in range(num_layers)]
