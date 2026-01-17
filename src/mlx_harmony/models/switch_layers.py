"""
Switch Transformer layers for MLX Harmony.

This module provides switch layers for mixture-of-experts models without depending on mlx-lm.
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        # Initialize with dummy quantized weights - actual weights will be loaded from file
        # Make sure input_dims is divisible by group_size for quantization
        dummy_weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            dummy_weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        # For mxfp8 mode, biases are not used (must be None)
        # Don't set self.biases attribute at all for mxfp8 mode so load_weights doesn't expect them
        # For other modes, initialize biases attribute so it can be loaded from weights
        if mode != "mxfp8":
            if biases and len(biases) > 0:
                self.biases = biases[0]
            else:
                # Create biases array with same shape and dtype as scales for loading
                # Scales shape is (num_experts, output_dims, num_groups)
                # where num_groups = input_dims // group_size
                self.biases = mx.zeros(self.scales.shape, dtype=self.scales.dtype)
        # For mxfp8, don't set self.biases at all (not a model parameter)

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        # For mxfp8 mode, biases must be None (not used)
        # For other modes, use biases if available
        if self.mode == "mxfp8":
            biases = None
        else:
            biases = self.get("biases") if hasattr(self, "biases") else None
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        """Convert SwitchLinear to QuantizedSwitchLinear."""
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,  # Don't create bias in QuantizedSwitchLinear, we'll copy it if needed
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


@partial(mx.compile, shapeless=True)
def swiglu(x, gate):
    return nn.silu(gate) * x


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(x, gate)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class QuantizedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        self.gate_proj = QuantizedSwitchLinear(
            input_dims, hidden_dims, num_experts, bias=bias, group_size=group_size, bits=bits, mode=mode
        )
        self.up_proj = QuantizedSwitchLinear(
            input_dims, hidden_dims, num_experts, bias=bias, group_size=group_size, bits=bits, mode=mode
        )
        self.down_proj = QuantizedSwitchLinear(
            hidden_dims, input_dims, num_experts, bias=bias, group_size=group_size, bits=bits, mode=mode
        )
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
