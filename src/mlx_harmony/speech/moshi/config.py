from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mlx_harmony.config import MoshiConfig, load_moshi_config


class MoshiCliOverrides(BaseModel):
    stt_model_path: str | None = None
    stt_config_path: str | None = None
    tts_model_path: str | None = None
    tts_config_path: str | None = None
    tts_voice_path: str | None = None
    stt_max_seconds: float | None = None
    stt_vad: bool | None = None
    stt_vad_threshold: float | None = None
    stt_vad_hits: int | None = None
    stt_silence: bool | None = None
    stt_silence_threshold: float | None = None
    stt_silence_ms: int | None = None
    stt_min_speech_ms: int | None = None
    stt_block_ms: int | None = None
    stt_warmup_blocks: int | None = None
    barge_in: bool | None = None
    barge_in_window_seconds: float | None = None
    quantize: int | None = None
    tts_chunk_chars: int | None = None
    tts_chunk_sentences: bool | None = None
    tts_chunk_min_chars: int | None = None
    tts_stream: bool | None = None
    use_stt: bool | None = None
    use_tts: bool | None = None
    smoke_test: bool | None = None


def _load_base_config(args: Any) -> MoshiConfig:
    default_moshi_path = Path("configs/moshi.json")
    if args.moshi_config:
        return load_moshi_config(args.moshi_config)
    if default_moshi_path.exists():
        return load_moshi_config(str(default_moshi_path))
    return MoshiConfig(enabled=True)


def _build_overrides(args: Any) -> MoshiCliOverrides:
    return MoshiCliOverrides(
        stt_model_path=args.moshi_stt_path,
        stt_config_path=args.moshi_stt_config,
        tts_model_path=args.moshi_tts_path,
        tts_config_path=args.moshi_tts_config,
        tts_voice_path=args.moshi_voice_path,
        stt_max_seconds=args.moshi_max_seconds,
        stt_vad=bool(args.moshi_vad) if args.moshi_vad is not None else None,
        stt_vad_threshold=args.moshi_vad_threshold,
        stt_vad_hits=args.moshi_vad_hits,
        stt_silence=bool(args.moshi_silence) if args.moshi_silence is not None else None,
        stt_silence_threshold=args.moshi_silence_threshold,
        stt_silence_ms=args.moshi_silence_ms,
        stt_min_speech_ms=args.moshi_min_speech_ms,
        stt_block_ms=args.moshi_stt_block_ms,
        stt_warmup_blocks=args.moshi_stt_warmup_blocks,
        barge_in=bool(args.moshi_barge_in) if args.moshi_barge_in is not None else None,
        barge_in_window_seconds=args.moshi_barge_in_window,
        quantize=args.moshi_quantize,
        tts_chunk_chars=args.moshi_tts_chunk_chars,
        tts_chunk_sentences=bool(args.moshi_tts_chunk_sentences)
        if args.moshi_tts_chunk_sentences is not None
        else None,
        tts_chunk_min_chars=args.moshi_tts_chunk_min_chars,
        tts_stream=bool(args.moshi_tts_stream) if args.moshi_tts_stream is not None else None,
        use_stt=bool(args.moshi_stt) if args.moshi_stt is not None else None,
        use_tts=bool(args.moshi_tts) if args.moshi_tts is not None else None,
        smoke_test=True if args.moshi_smoke else None,
    )


def resolve_moshi_config(args: Any) -> MoshiConfig | None:
    if not args.moshi:
        return None
    base_config = _load_base_config(args)
    overrides = _build_overrides(args)
    config = base_config.model_copy(update=overrides.model_dump(exclude_none=True))
    return config.model_copy(update={"enabled": True})
