"""
Optimized model loading utilities with filesystem cache pre-warming and memory locking (mlock).

This module provides utilities to improve disk I/O performance when loading large models:
- Pre-warm filesystem cache by reading weight files
- Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal backend)
- Optional parallel file prefetching
"""

import glob
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.utils import TokenizerWrapper


def _prewarm_filesystem_cache(file_path: Path) -> None:
    """
    Pre-warm filesystem cache by reading a file into OS cache.

    This reads the entire file sequentially to ensure it's in the OS filesystem cache,
    which significantly speeds up subsequent reads by MLX.

    Args:
        file_path: Path to the file to pre-warm
    """
    try:
        # Read file with large buffer size for better I/O performance
        with open(file_path, "rb", buffering=1024 * 1024) as f:  # 1MB buffer
            # Read in chunks to avoid loading entire file into Python memory
            chunk_size = 1024 * 1024 * 16  # 16MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
    except (IOError, OSError):
        # Silently fail if file can't be read (e.g., permissions)
        pass


def _get_model_size_from_files(model_path: Path) -> Optional[int]:
    """
    Get model size in bytes from safetensors index file or file sizes.

    First tries to read `model.safetensors.index.json` to get the exact
    `metadata.total_size` (this is the accurate size of model weights).
    Falls back to summing safetensors file sizes if the index file doesn't exist.

    Args:
        model_path: Path to the model directory

    Returns:
        Model size in bytes, or None if files not found
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    # First, try to read the index file for exact size (MLX-LM format)
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r") as f:
                index_data = json.load(f)
                if "metadata" in index_data and "total_size" in index_data["metadata"]:
                    total_size = index_data["metadata"]["total_size"]
                    return total_size
        except (json.JSONDecodeError, IOError, KeyError):
            # If we can't read the index file, fall through to file size method
            pass

    # Fallback: sum file sizes (less accurate, includes file overhead)
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        # Try alternative pattern
        weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        return None

    # Sum file sizes
    total_size = 0
    for wf in weight_files:
        try:
            total_size += Path(wf).stat().st_size
        except (OSError, IOError):
            # If we can't stat a file, skip it
            continue

    return total_size if total_size > 0 else None


def prewarm_model_cache(model_path: Path) -> None:
    """
    Pre-warm filesystem cache for all model weight files.

    This reads all safetensors files into the OS filesystem cache,
    which significantly speeds up subsequent MLX loading.

    Args:
        model_path: Path to the model directory
    """
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    # Find all safetensors files
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        # Try alternative pattern
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
               Note: This requires macOS with Metal backend (mlock equivalent)

    Returns:
        Tuple of (model, tokenizer) or (model, tokenizer, config) if return_config=True
    """
    from mlx_lm.utils import _download

    # Download model if needed
    model_path = Path(_download(path_or_hf_repo, revision=revision))

    # Set wired memory limit BEFORE loading (if mlock requested)
    # Following MLX-LM pattern: set to max_recommended_working_set_size (the maximum allowed)
    # The wired limit is a capacity - MLX will wire buffers up to this limit as they're allocated.
    # This is more flexible than setting to model size, as it allows room for activations, KV cache, etc.
    old_wired_limit = None
    old_cache_limit = None
    estimated_model_size = None
    if mlock:
        print("[INFO] Attempting to set wired memory limit...")
        if mx.metal.is_available():
            try:
                # Get model size estimate for informational purposes
                estimated_model_size = _get_model_size_from_files(model_path)

                # Get max recommended working set size (system limit)
                max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]

                # CRITICAL: Clear cache BEFORE setting wired limit and loading
                # Buffers reused from cache are NOT wired - only newly allocated buffers are.
                # By clearing the cache first, we ensure all model buffers are freshly allocated
                # and will be wired when the wired limit is active.
                mx.clear_cache()

                # CRITICAL: Set cache limit to 0 to prevent buffers from being cached/unwired
                # When buffers are freed and recycled to cache, they're removed from the residency set (unwired).
                # By disabling caching, we ensure that once buffers are wired, they stay wired.
                # Buffers will only be freed when no longer referenced, and won't be unwired prematurely.
                old_cache_limit = mx.set_cache_limit(0)
                print("[INFO] Cleared buffer cache and disabled caching to keep buffers wired.")

                # Set wired limit to maximum allowed (following MLX-LM pattern)
                # This is a capacity limit, not an allocation - buffers are wired as they're allocated
                old_wired_limit = mx.set_wired_limit(max_rec_size)

                if estimated_model_size is not None:
                    # Warn if model exceeds 90% of recommended size
                    if estimated_model_size > 0.9 * max_rec_size:
                        print(
                            f"[WARNING] Estimated model size ({estimated_model_size / (1024**3):.2f} GB) "
                            f"exceeds 90% of max recommended working set size ({max_rec_size / (1024**3):.2f} GB). "
                            f"Performance may be degraded. Consider increasing system wired limit: "
                            f"sudo sysctl iogpu.wired_limit_mb={int(estimated_model_size / (1024**2) * 1.1)}"
                        )
                    else:
                        print(
                            f"[INFO] Estimated model size: {estimated_model_size / (1024**3):.2f} GB. "
                            f"Set wired memory limit to {max_rec_size / (1024**3):.2f} GB "
                            f"(max recommended working set size). "
                            f"Model weights will be kept in wired memory as they load."
                        )
                else:
                    print(
                        f"[INFO] Set wired memory limit to {max_rec_size / (1024**3):.2f} GB "
                        f"(max recommended working set size). "
                        f"Model weights will be kept in wired memory as they load."
                    )

                # Warn if lazy loading is enabled - weights won't be loaded until first use
                if lazy:
                    print(
                        "[WARNING] Lazy loading is enabled. Model weights will be loaded on first use. "
                        "For best wired memory effectiveness, consider using lazy=False to load weights immediately."
                    )
            except Exception as e:
                print(f"[WARNING] Failed to set wired limit: {e}")
                print("[INFO] Wired memory requires macOS 15.0+ with Metal backend.")
        else:
            print("[WARNING] Wired memory requires macOS with Metal backend.")
            print("[INFO] Current platform does not support MLX Metal backend.")

    # Pre-warm filesystem cache if requested
    if prewarm_cache:
        prewarm_model_cache(model_path)

    # Load model using standard MLX-LM loader
    # With wired limit set, MLX will keep model weights in wired memory
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

    # After loading, ensure model weights are fully allocated and wired
    # Since we cleared the cache before loading, all buffers should be freshly allocated
    # and wired. We still need to ensure all parameters are evaluated to guarantee
    # all buffers are allocated while the wired limit is active.
    # CRITICAL: After evaluation, trigger a dummy inference to ensure buffers stay wired.
    # This forces all model parameter buffers to be active and keeps them in the residency set.
    if mlock and mx.metal.is_available() and not lazy:
        try:
            from mlx.utils import tree_flatten

            # Force evaluation of all parameters to ensure they're allocated
            # This ensures all model weight buffers are allocated while wired limit is active
            params = [p for _, p in tree_flatten(model.parameters()) if isinstance(p, mx.array)]
            if params:
                mx.eval(params)
                # Synchronize to ensure all allocations are complete
                mx.synchronize()

                # CRITICAL: Force a dummy forward pass to ensure parameter buffers stay active
                # This prevents them from being freed/recycled which would unwire them.
                # We do this by creating a minimal input and running a forward pass.
                # This ensures all parameter buffers remain in active use.
                try:
                    # Get tokenizer to create a dummy input
                    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                        dummy_input = mx.array([[tokenizer.bos_token_id]])
                    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                        dummy_input = mx.array([[tokenizer.eos_token_id]])
                    else:
                        # Fallback: use token ID 1 (usually a valid token)
                        dummy_input = mx.array([[1]])

                    # Run a minimal forward pass to ensure parameter buffers stay active
                    # This prevents them from being cached/unwired
                    _ = model(dummy_input)
                    mx.eval(_)
                    mx.synchronize()
                    print("[INFO] Ran dummy forward pass to keep parameter buffers active and wired.")
                except Exception as forward_e:
                    # If forward pass fails (e.g., wrong input format), that's okay
                    # The evaluation above should have been sufficient
                    print(f"[INFO] Evaluated model parameters (forward pass not possible: {forward_e}).")

                print("[INFO] Model parameters evaluated and synchronized to ensure wired memory.")
        except Exception as e:
            print(f"[WARNING] Could not ensure model parameters are wired: {e}")

    # Verify actual model size matches estimate (informational)
    if mlock and mx.metal.is_available() and not lazy:
        try:
            from mlx.utils import tree_flatten

            actual_model_bytes = sum(
                p.nbytes
                for _, p in tree_flatten(model.parameters())
                if isinstance(p, mx.array)
            )
            max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]

            # Compare actual vs estimated (if we had an estimate)
            if estimated_model_size is not None:
                size_diff = abs(actual_model_bytes - estimated_model_size) / estimated_model_size
                if size_diff > 0.1:  # More than 10% difference
                    print(
                        f"[INFO] Actual model size ({actual_model_bytes / (1024**3):.2f} GB) "
                        f"differs from estimate ({estimated_model_size / (1024**3):.2f} GB) by "
                        f"{size_diff * 100:.1f}%."
                    )

            # Check if model size is reasonable relative to wired limit
            if actual_model_bytes > 0.9 * max_rec_size:
                print(
                    f"[WARNING] Model size ({actual_model_bytes / (1024**3):.2f} GB) "
                    f"exceeds 90% of max recommended working set size "
                    f"({max_rec_size / (1024**3):.2f} GB). "
                    f"Performance may be degraded. Consider increasing system wired limit: "
                    f"sudo sysctl iogpu.wired_limit_mb={int(actual_model_bytes / (1024**2) * 1.1)}"
                )
            else:
                print(
                    f"[INFO] Model loaded: {actual_model_bytes / (1024**3):.2f} GB. "
                    f"Wired memory limit: {max_rec_size / (1024**3):.2f} GB."
                )

            # Store old limits on model for potential restoration
            if old_wired_limit is not None:
                model._mlx_harmony_old_wired_limit = old_wired_limit
            if old_cache_limit is not None:
                model._mlx_harmony_old_cache_limit = old_cache_limit
        except Exception as e:
            print(f"[WARNING] Could not verify model size: {e}")
    elif mlock and old_wired_limit is not None:
        # Store old limit even if we can't verify size (lazy loading)
        model._mlx_harmony_old_wired_limit = old_wired_limit

    if return_config:
        return model, tokenizer, config
    else:
        return model, tokenizer
