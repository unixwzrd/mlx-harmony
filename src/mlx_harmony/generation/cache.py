"""
Standalone KV cache implementation for MLX Harmony.

This module provides KV cache classes for efficient autoregressive generation
without depending on mlx-lm.
"""

from typing import Optional

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


class RotatingKVCache:
    """
    Rotating KV cache for sliding window attention.

    This cache maintains a fixed-size window and rotates when full,
    keeping only the most recent tokens up to max_size.
    """

    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self._offset = 0
        self.max_size = max_size
        self._idx = 0

    @property
    def offset(self):
        """Current offset in the cache."""
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """Rearrange the cache into temporal order, slicing off the end if unused."""
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def __len__(self):
        return min(self._offset, self.max_size)

    def make_mask(self, N: int, window_size: Optional[int] = None, return_array: bool = False):
        """Create attention mask for sliding window."""
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self._offset)
            if offset + N > window_size or return_array:
                from mlx_harmony.models.base import create_causal_mask
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self._offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self._offset < self.max_size:
                    mask_size = self._offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        return n

    @property
    def state(self):
        """Get current cache state (keys, values)."""
        if self.keys is None:
            return None, None
        # Return keys and values in temporal order
        if self._offset < self.max_size:
            return self.keys[..., : self._offset, :], self.values[..., : self._offset, :]
        else:
            # Cache is full, return in temporal order
            keys_ordered = self._temporal_order(self.keys)
            values_ordered = self._temporal_order(self.values)
            return keys_ordered, values_ordered

    @state.setter
    def state(self, v):
        """Set cache state from (keys, values)."""
        if v is None or v[0] is None:
            self.keys = None
            self.values = None
            self._offset = 0
            self._idx = 0
        else:
            self.keys, self.values = v
            self._offset = self.keys.shape[2]
            self._idx = self._offset

    def __bool__(self):
        """Always return True to allow 'cache or make_cache()' pattern."""
        return True


def make_prompt_cache(model: nn.Module, max_kv_size: int | None = None) -> list[KVCache | RotatingKVCache]:
    """
    Create a KV cache for a model.

    Args:
        model: MLX model (must have .layers attribute)
        max_kv_size: Optional maximum cache size (used for RotatingKVCache if provided)

    Returns:
        List of KVCache or RotatingKVCache objects, one per model layer
    """
    # If model has its own make_cache method, use it
    if hasattr(model, "make_cache"):
        return model.make_cache()

    # Otherwise, create one cache per layer
    num_layers = len(model.layers) if hasattr(model, "layers") else 1
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size) for _ in range(num_layers)]
    return [KVCache() for _ in range(num_layers)]
