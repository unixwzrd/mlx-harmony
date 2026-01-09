"""
Optimized model loading utilities with filesystem cache pre-warming and memory locking (mlock).

This module provides utilities to improve disk I/O performance when loading large models:
- Pre-warm filesystem cache by reading weight files
- Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal backend)
"""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load
from mlx_lm.utils import TokenizerWrapper, _download


def _prewarm_filesystem_cache(file_path: Path) -> None:
    """Pre-warm filesystem cache by reading a file into OS cache."""
    try:
        with open(file_path, "rb", buffering=1024 * 1024) as f:  # 1MB buffer
            chunk_size = 1024 * 1024 * 16  # 16MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
    except (IOError, OSError):
        pass


def _get_model_size_from_files(model_path: Path) -> Optional[int]:
    """Get model size in bytes from safetensors index file or file sizes."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    # Try index file first for exact size
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r") as f:
                index_data = json.load(f)
                if "metadata" in index_data and "total_size" in index_data["metadata"]:
                    return index_data["metadata"]["total_size"]
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # Fallback: sum file sizes
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        return None

    total_size = 0
    for wf in weight_files:
        try:
            total_size += Path(wf).stat().st_size
        except (OSError, IOError):
            continue

    return total_size if total_size > 0 else None


def prewarm_model_cache(model_path: Path) -> None:
    """Pre-warm filesystem cache for all model weight files."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        return

    print(f"[INFO] Pre-warming filesystem cache for {len(weight_files)} weight file(s)...")
    for wf in weight_files:
        _prewarm_filesystem_cache(Path(wf))
    print("[INFO] Filesystem cache pre-warming complete.")


def load_optimized(
    path_or_hf_repo: str,
    tokenizer_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
    prewarm_cache: bool = True,
    mlock: bool = False,
) -> Union[
    Tuple[object, TokenizerWrapper],
    Tuple[object, TokenizerWrapper, dict],
]:
    """
    Load a model with optimizations for faster disk I/O and memory management.

    For mlock to work correctly:
    1. Set wired limit BEFORE loading (buffers are wired as they're allocated)
    2. Disable caching (prevents buffers from being unwired when freed)
    3. Keep wired limit set for model lifetime (prevents unwiring)
    4. Keep strong references to parameters (prevents deallocation)

    Args:
        path_or_hf_repo: Path to model directory or Hugging Face repo ID
        tokenizer_config: Optional tokenizer configuration
        model_config: Optional model configuration
        adapter_path: Optional path to LoRA adapters
        lazy: If False, evaluate model parameters immediately (loads into memory)
        return_config: If True, return model config as third element
        revision: Optional Hugging Face revision (branch, tag, or commit)
        prewarm_cache: If True, pre-warm filesystem cache before loading (default: True)
        mlock: If True, lock model weights in memory using MLX's wired limit (default: False)
               Note: This requires macOS 15.0+ with Metal backend

    Returns:
        Tuple of (model, tokenizer) or (model, tokenizer, config) if return_config=True
    """
    # Download model if needed
    model_path = Path(_download(path_or_hf_repo, revision=revision))

    # Pre-warm filesystem cache if requested
    if prewarm_cache:
        prewarm_model_cache(model_path)

    # Set up memory locking BEFORE loading
    old_wired_limit = None
    old_cache_limit = None

    if mlock:
        if not mx.metal.is_available():
            print("[WARNING] Wired memory requires macOS 15.0+ with Metal backend.")
            mlock = False
        else:
            try:
                max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]

                # CRITICAL STEP 1: Clear cache before setting limits
                mx.clear_cache()

                # CRITICAL STEP 2: Disable caching to prevent unwiring
                # When buffers are freed, if caching is enabled they go to cache and are unwired.
                # With cache disabled (limit=0), freed buffers are deallocated (also unwired),
                # but they won't be reused from cache (which wouldn't be wired).
                # The key is to prevent buffers from being freed in the first place.
                old_cache_limit = mx.set_cache_limit(0)

                # CRITICAL STEP 3: Set wired limit BEFORE loading
                # Buffers are only wired when newly allocated (not from cache).
                # With cache disabled, all allocations are fresh and will be wired.
                old_wired_limit = mx.set_wired_limit(max_rec_size)

                print(
                    f"[INFO] Set wired memory limit to {max_rec_size / (1024**3):.2f} GB. "
                    f"Cache disabled to prevent unwiring."
                )

                # Optional: warn about large models
                estimated_size = _get_model_size_from_files(model_path)
                if estimated_size and estimated_size > 0.9 * max_rec_size:
                    print(
                        f"[WARNING] Estimated model size ({estimated_size / (1024**3):.2f} GB) "
                        f"exceeds 90% of max recommended working set size. "
                        f"Consider: sudo sysctl iogpu.wired_limit_mb={int(estimated_size / (1024**2) * 1.1)}"
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

    # Load model - buffers will be wired as they're allocated (if mlock=True)
    if return_config:
        model, tokenizer, config = mlx_load(
            path_or_hf_repo,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=True,
            revision=revision,
        )
    else:
        model, tokenizer = mlx_load(
            path_or_hf_repo,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=False,
            revision=revision,
        )
        config = None

    # CRITICAL STEP 4: After loading, ensure parameters are allocated and stay wired
    # The model object keeps references to parameters, which should keep buffers alive.
    # But we also store explicit references to ensure they're never garbage collected.
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

                # CRITICAL: Keep parameters actively allocated by doing a dummy operation
                # This ensures buffers are fully allocated and wired, and prevents them
                # from being freed/swapped out. We'll also periodically "touch" parameters
                # after inference to keep them active.
                try:
                    # Do a small computation on parameters to ensure they're active
                    # This prevents MLX from freeing them when not in use
                    # Touch first few parameters to keep them active
                    for param in param_arrays[:10]:
                        _ = mx.sum(param)  # Small computation to keep param active
                    mx.eval(param_arrays)  # Ensure all are evaluated
                    mx.synchronize()  # Ensure all allocations complete
                except Exception:
                    pass  # Ignore errors - parameters are already allocated

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

    return (model, tokenizer, config) if return_config else (model, tokenizer)
