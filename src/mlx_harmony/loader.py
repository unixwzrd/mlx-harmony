"""
Standalone model loading for MLX Harmony.

This module provides model loading without depending on mlx-lm's high-level load function.
For GPT-OSS models, we use our own model architecture (no mlx_lm dependency).
For other models, we fall back to mlx-lm model architectures (can be extracted later).
"""

import glob
import importlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

# Model type remapping (matches mlx-lm)
MODEL_REMAPPING = {
    "mistral": "llama",
    "llava": "mistral3",
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
    "kimi_k2": "deepseek_v3",
    "qwen2_5_vl": "qwen2_vl",
    "minimax_m2": "minimax",
    "iquestcoder": "llama",
}


def _download_model(path_or_hf_repo: str, revision: str | None = None) -> Path:
    """
    Download model from HuggingFace Hub if needed, or return local path.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        revision: Optional revision (branch, tag, or commit)

    Returns:
        Path to local model directory
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        # Download from HuggingFace Hub
        allow_patterns = [
            "*.json",
            "model*.safetensors",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ]
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        )

    return model_path


def _load_config(model_path: Path) -> dict[str, Any]:
    """Load model configuration from config.json."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Merge generation_config.json if it exists
    generation_config_file = model_path / "generation_config.json"
    if generation_config_file.exists():
        try:
            with open(generation_config_file, "r", encoding="utf-8") as f:
                generation_config = json.load(f)
                # Extract eos_token_id if present
                if "eos_token_id" in generation_config:
                    config["eos_token_id"] = generation_config["eos_token_id"]
        except (json.JSONDecodeError, IOError):
            pass

    return config


def _get_model_classes(config: dict[str, Any]) -> tuple[type[nn.Module], type]:
    """
    Get model and ModelArgs classes from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Tuple of (Model class, ModelArgs class)
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    # For GPT-OSS models, use our local model architecture (no mlx_lm dependency)
    if model_type == "gpt_oss":
        try:
            from mlx_harmony.models.gpt_oss import Model, ModelArgs
            return Model, ModelArgs
        except ImportError as e:
            raise ValueError(f"Failed to load GPT-OSS model architecture: {e}") from e

    # For other models, fall back to mlx_lm (for now - can be extracted later)
    try:
        # Import model architecture from mlx_lm.models
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError as e:
        raise ValueError(f"Model type {model_type} not supported: {e}") from e

    if not hasattr(arch, "Model") or not hasattr(arch, "ModelArgs"):
        raise ValueError(f"Model module {model_type} missing Model or ModelArgs class")

    return arch.Model, arch.ModelArgs


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    """Load model weights from safetensors files."""
    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def _apply_quantization(model: nn.Module, config: dict[str, Any], weights: dict[str, mx.array]) -> None:
    """Apply quantization to model if specified in config."""
    # Check for quantization config
    quantization = config.get("quantization")
    quantization_config = config.get("quantization_config")

    if quantization:
        # New format quantization
        def class_predicate(path, module):
            # Handle custom per layer quantizations
            if path in config.get("quantization", {}):
                return config["quantization"][path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    elif quantization_config:
        # Legacy quantization format (quantization_config)
        quant_method = quantization_config.get("quant_method")
        if quant_method == "mxfp4":
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights)
        elif quant_method == "mxfp8":
            # New: mxfp8 quantization format (added in MLX upgrade)
            quantization = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights)
        elif quant_method == "nvfp4":
            # New: nvfp4 quantization format (added in MLX upgrade)
            # Note: nvfp4 requires group_size=16 (not 32 like mxfp4/mxfp8)
            quantization = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights)
        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ weights - this is complex, for now just apply basic quantization
            # TODO: Implement proper AWQ/GPTQ transformation if needed
            print(f"[WARNING] {quant_method} quantization detected but transformation not fully implemented")
            quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            config["quantization"] = quantization
            _apply_quantization(model, config, weights)


def _get_model_size_from_index(model_path: Path) -> int | None:
    """Get model size in bytes from safetensors index file."""
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return None

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            if "metadata" in index_data and "total_size" in index_data["metadata"]:
                return index_data["metadata"]["total_size"]
    except (json.JSONDecodeError, IOError, KeyError):
        pass

    return None


def load_model_standalone(
    path_or_hf_repo: str,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    adapter_path: str | None = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: str | None = None,
    mlock: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, dict[str, Any]]:
    """
    Load a model standalone without mlx-lm's load function.

    Args:
        path_or_hf_repo: Path to model directory or HuggingFace repo ID
        tokenizer_config: Optional tokenizer configuration
        model_config: Optional model configuration (merged into config)
        adapter_path: Optional path to LoRA adapters (not yet implemented)
        lazy: If False, evaluate model parameters immediately
        return_config: If True, return model config as third element
        revision: Optional HuggingFace revision
        mlock: If True, lock model weights in memory using MLX's wired limit

    Returns:
        Tuple of (model, tokenizer) or (model, tokenizer, config) if return_config=True
    """
    # Download model if needed
    model_path = _download_model(path_or_hf_repo, revision)

    # Get model size estimate BEFORE setting up memory locking
    # We'll set the wired limit to model size + small margin
    estimated_model_size = _get_model_size_from_index(model_path)

    # Set up memory locking BEFORE loading
    old_wired_limit = None
    old_cache_limit = None

    if mlock:
        if not mx.metal.is_available():
            print("[WARNING] Wired memory requires macOS 15.0+ with Metal backend.")
            mlock = False
        else:
            try:
                # CRITICAL STEP 1: Clear cache before setting limits
                mx.clear_cache()

                # CRITICAL STEP 2: Disable caching to prevent unwiring
                # When buffers are freed, if caching is enabled they go to cache and are unwired.
                # With cache disabled (limit=0), freed buffers are deallocated (also unwired),
                # but they won't be reused from cache (which wouldn't be wired).
                # The key is to prevent buffers from being freed in the first place.
                old_cache_limit = mx.set_cache_limit(0)

                # CRITICAL STEP 3: Set wired limit to model size + small margin BEFORE loading
                # This ensures we wire memory as we read it in, without over-allocating.
                # Buffers are only wired when newly allocated (not from cache).
                # With cache disabled, all allocations are fresh and will be wired.
                if estimated_model_size:
                    # Set wired limit to model size + 10% margin
                    wired_limit = int(estimated_model_size * 1.1)
                    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]

                    # Don't exceed system maximum
                    if wired_limit > max_rec_size:
                        print(
                            f"[WARNING] Model size ({estimated_model_size / (1024**3):.2f} GB) + margin "
                            f"exceeds max recommended ({max_rec_size / (1024**3):.2f} GB). "
                            f"Using max recommended instead."
                        )
                        wired_limit = max_rec_size

                    old_wired_limit = mx.set_wired_limit(wired_limit)
                    print(
                        f"[INFO] Set wired memory limit to {wired_limit / (1024**3):.2f} GB "
                        f"(model size: {estimated_model_size / (1024**3):.2f} GB + 10% margin). "
                        f"Cache disabled to prevent unwiring."
                    )
                else:
                    # Fallback: use max recommended if we can't estimate model size
                    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
                    old_wired_limit = mx.set_wired_limit(max_rec_size)
                    print(
                        f"[INFO] Could not estimate model size, using max recommended: "
                        f"{max_rec_size / (1024**3):.2f} GB. Cache disabled to prevent unwiring."
                    )

                if lazy:
                    print(
                        "[WARNING] Lazy loading enabled. Use lazy=False for best wired memory effectiveness."
                    )
            except Exception as e:
                print(f"[WARNING] Failed to set wired limit: {e}")
                mlock = False
                if old_cache_limit is not None:
                    mx.set_cache_limit(old_cache_limit)
                old_wired_limit = None
                old_cache_limit = None

    # Load configuration
    config = _load_config(model_path)
    if model_config:
        config.update(model_config)

    # Get model architecture classes
    model_class, model_args_class = _get_model_classes(config)

    # Load weights before instantiating model (need to check for quantized weights)
    weights = _load_weights(model_path)

    # Check if model has quantized expert weights (GPT-OSS specific)
    # Check by looking for quantized expert weight keys in the loaded weights
    model_type = config.get("model_type", "")
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    # Instantiate model (always use regular layers, let nn.quantize() handle quantization)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Sanitize weights if model has sanitize method
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Apply quantization if needed (nn.quantize() will handle conversion of layers)
    # This matches mlx_lm and mlx-examples approach: use regular layers, then nn.quantize()
    # SwitchLinear has to_quantized() method, so nn.quantize() will convert it when scales are in weights
    _apply_quantization(model, config, weights)

    # Load weights into model
    model.load_weights(list(weights.items()), strict=True)

    # Evaluate parameters if not lazy (forces allocation while wired limit is active)
    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Load tokenizer using pure Python native implementation
    # No mlx-lm, no transformers, no PyTorch - just pure Python + MLX
    from mlx_harmony.tokenizer_native import load_tokenizer_native

    tokenizer = load_tokenizer_native(model_path)

    # CRITICAL STEP 4: After loading, ensure parameters are allocated and stay wired
    if mlock and not lazy:
        try:
            # Get all parameter arrays
            all_params = list(tree_flatten(model.parameters()))
            param_arrays = [p for _, p in all_params if isinstance(p, mx.array)]

            if param_arrays:
                # Force evaluation to ensure all parameters are allocated
                # This happens while wired limit is active and cache is disabled
                mx.eval(param_arrays)
                mx.synchronize()

                # CRITICAL: Store strong references to prevent deallocation
                # If parameter arrays are deallocated, their buffers are freed and unwired.
                # By keeping explicit references, we ensure they stay allocated.
                model._mlx_harmony_param_refs = param_arrays
                model._mlx_harmony_param_tree = all_params

                # Store old limits for potential restoration
                if old_wired_limit is not None:
                    model._mlx_harmony_old_wired_limit = old_wired_limit
                if old_cache_limit is not None:
                    model._mlx_harmony_old_cache_limit = old_cache_limit

                # Report model size
                actual_bytes = sum(p.nbytes for p in param_arrays)
                print(
                    f"[INFO] Model loaded: {actual_bytes / (1024**3):.2f} GB. "
                    f"Wired limit remains set - parameters should stay wired as long as model is alive."
                )
                print(
                    f"[INFO] To verify wired memory, check Activity Monitor: "
                    f"Python process should show ~{actual_bytes / (1024**3):.2f} GB in 'Wired Memory' column."
                )
            else:
                print("[WARNING] No parameter arrays found in model.")
        except Exception as e:
            print(f"[WARNING] Could not wire model parameters: {e}")

    # NOTE: We intentionally do NOT restore the wired limit here.
    # The wired limit must stay set for the lifetime of the model to keep buffers wired.
    # The caller can restore it when the model is unloaded if needed.

    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer
