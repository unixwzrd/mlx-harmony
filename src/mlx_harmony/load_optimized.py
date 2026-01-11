"""
Optimized model loading utilities with filesystem cache pre-warming and memory locking (mlock).

This module provides utilities to improve disk I/O performance when loading large models:
- Pre-warm filesystem cache by reading weight files
- Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal backend)
"""

import glob
from pathlib import Path
from typing import Optional, Tuple, Union

# Use our standalone loader instead of mlx_lm to avoid PyTorch dependency
from mlx_harmony.loader import load_model_standalone
from mlx_harmony.logging import get_logger
from mlx_harmony.tokenizer_native import ByteLevelBPETokenizer

logger = get_logger(__name__)


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


# Use model size estimation from loader.py (consolidated functionality)
def _get_model_size_from_files(model_path: Path) -> Optional[int]:
    """Get model size in bytes from safetensors index file or file sizes."""
    # Import from loader to avoid duplication
    from mlx_harmony.loader import _get_model_size_from_index
    return _get_model_size_from_index(model_path)


def prewarm_model_cache(model_path: Path) -> None:
    """Pre-warm filesystem cache for all model weight files."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.safetensors"))

    if not weight_files:
        return

    logger.info("Pre-warming filesystem cache for %d weight file(s)...", len(weight_files))
    for wf in weight_files:
        _prewarm_filesystem_cache(Path(wf))
    logger.info("Filesystem cache pre-warming complete.")


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
    Tuple[object, ByteLevelBPETokenizer],
    Tuple[object, ByteLevelBPETokenizer, dict],
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
    # Pre-warm filesystem cache if requested
    # Use _download_model from loader to ensure consistent behavior
    if prewarm_cache:
        from mlx_harmony.loader import _download_model
        model_path = _download_model(path_or_hf_repo, revision=revision)
        prewarm_model_cache(model_path)

    # Use our standalone loader (handles mlock internally)
    if return_config:
        model, tokenizer, config = load_model_standalone(
            path_or_hf_repo,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=True,
            revision=revision,
            mlock=mlock,
        )
        return model, tokenizer, config
    else:
        model, tokenizer = load_model_standalone(
            path_or_hf_repo,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            return_config=False,
            revision=revision,
            mlock=mlock,
        )
        return model, tokenizer
