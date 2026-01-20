from __future__ import annotations

import json
from pathlib import Path

from mlx_harmony.config import MoshiConfig
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)

_STT_SAMPLE_RATE = 24000
_STT_BLOCKSIZE = 1920
_TTS_BLOCKSIZE = 1920
_STT_WARMUP_BLOCKS = 2


def require_moshi_mlx() -> None:
    """Ensure moshi_mlx is importable when Moshi voice mode is enabled."""
    try:
        import moshi_mlx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Moshi voice mode requires the 'moshi-mlx' package. "
            "Install it before running with --moshi."
        ) from exc


def log_moshi_config(config: MoshiConfig) -> None:
    logger.info("Moshi voice mode enabled")
    if config.use_stt:
        logger.info("STT model path: %s", config.stt_model_path)
        logger.info(
            "STT silence detection: %s (threshold=%.4f, silence=%dms, min_speech=%dms)",
            "on" if config.stt_silence else "off",
            config.stt_silence_threshold,
            config.stt_silence_ms,
            config.stt_min_speech_ms,
        )
    else:
        logger.info("STT disabled")
    if config.use_tts:
        logger.info("TTS model path: %s", config.tts_model_path)
        if config.tts_voice_path:
            logger.info("TTS voice path: %s", config.tts_voice_path)
    else:
        logger.info("TTS disabled")
    if config.quantize:
        logger.info("Moshi quantize: %s", config.quantize)


def _load_json_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Moshi config.json at {config_path}")
    with open(config_path, encoding="utf-8") as fobj:
        return json.load(fobj)


def _resolve_model_file(model_dir: Path, filename: str, label: str) -> Path:
    candidate = model_dir / filename
    if not candidate.exists():
        raise FileNotFoundError(f"Missing {label} file: {candidate}")
    return candidate


def _import_stt_deps() -> tuple[object, object, object, object, object, object, object, object]:
    """Import optional STT dependencies only when Moshi STT is enabled."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import rustymimi
        import sentencepiece
        import sounddevice as sd
        from moshi_mlx import models, utils
    except ImportError as exc:
        raise RuntimeError(
            "Moshi STT requires moshi-mlx, rustymimi, sentencepiece, and sounddevice."
        ) from exc
    return mx, nn, np, rustymimi, sentencepiece, sd, models, utils


def _import_tts_deps() -> tuple[object, object, object, object, object, object, object]:
    """Import optional TTS dependencies only when Moshi TTS is enabled."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import sentencepiece
        import sounddevice as sd
        from moshi_mlx import models
        from moshi_mlx.models.tts import TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "Moshi TTS requires moshi-mlx, sentencepiece, sounddevice, and numpy."
        ) from exc
    return mx, nn, np, sentencepiece, sd, models, TTSModel


__all__ = [
    "_STT_BLOCKSIZE",
    "_STT_SAMPLE_RATE",
    "_STT_WARMUP_BLOCKS",
    "_TTS_BLOCKSIZE",
    "_import_stt_deps",
    "_import_tts_deps",
    "_load_json_config",
    "_resolve_model_file",
    "log_moshi_config",
    "require_moshi_mlx",
]
