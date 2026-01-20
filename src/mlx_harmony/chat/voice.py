from __future__ import annotations

import select
import sys
import time
from typing import Any

from mlx_harmony.config import MoshiConfig
from mlx_harmony.conversation.conversation_io import read_user_input_from_first_line
from mlx_harmony.logging import get_logger
from mlx_harmony.speech.moshi.config import resolve_moshi_config
from mlx_harmony.speech.moshi.loader import (
    MoshiSTT,
    MoshiTTS,
    log_moshi_config,
    require_moshi_mlx,
)

logger = get_logger(__name__)


def voice_status(state: str, detail: str | None = None) -> None:
    if detail:
        print(f"[VOICE] {state}: {detail}")
    else:
        print(f"[VOICE] {state}")


def init_moshi_components(args: Any) -> tuple[MoshiConfig | None, MoshiSTT | None, MoshiTTS | None, bool]:
    moshi_config = resolve_moshi_config(args)
    if moshi_config is None:
        return None, None, None, False
    missing_paths = moshi_config.validate_paths()
    if missing_paths:
        missing_text = ", ".join(missing_paths)
        raise RuntimeError(f"Moshi voice mode is missing required paths: {missing_text}")

    require_moshi_mlx()
    log_moshi_config(moshi_config)

    moshi_stt: MoshiSTT | None = None
    moshi_tts: MoshiTTS | None = None
    if moshi_config.use_stt:
        moshi_stt = MoshiSTT(
            moshi_config.stt_model_path,
            config_path=moshi_config.stt_config_path,
            block_ms=moshi_config.stt_block_ms,
            warmup_blocks=moshi_config.stt_warmup_blocks,
        )
    if moshi_config.use_tts:
        moshi_tts = MoshiTTS(
            moshi_config.tts_model_path,
            config_path=moshi_config.tts_config_path,
            voice_path=moshi_config.tts_voice_path,
            quantize=moshi_config.quantize,
        )

    if moshi_config.smoke_test:
        print("[VOICE] Running Moshi smoke test...")
        if moshi_tts is not None:
            moshi_tts.speak("Moshi smoke test.")
        if moshi_stt is not None:
            print("[VOICE] Speak now for a short STT test.")
            transcript = moshi_stt.listen_once(
                max_seconds=moshi_config.stt_max_seconds,
                vad=moshi_config.stt_vad,
                silence=moshi_config.stt_silence,
                silence_threshold=moshi_config.stt_silence_threshold,
                silence_ms=moshi_config.stt_silence_ms,
                min_speech_ms=moshi_config.stt_min_speech_ms,
            )
            print(f"[VOICE] STT transcript: {transcript}")
        return moshi_config, moshi_stt, moshi_tts, True

    return moshi_config, moshi_stt, moshi_tts, False


def listen_for_user_input(moshi_stt: MoshiSTT | None, moshi_config: MoshiConfig | None) -> str:
    if moshi_stt is None:
        raise RuntimeError("listen_for_user_input requires an active Moshi STT instance")

    if sys.stdin.isatty():
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            line = sys.stdin.readline()
            if not line:
                return ""
            return read_user_input_from_first_line(line.rstrip("\n"))

    listen_seconds = moshi_config.stt_max_seconds if moshi_config else 8.0
    voice_status("Listening", f"up to {listen_seconds:.1f}s")
    listen_start = time.perf_counter()
    transcript = moshi_stt.listen_once(
        max_seconds=listen_seconds,
        vad=moshi_config.stt_vad if moshi_config else False,
        vad_threshold=moshi_config.stt_vad_threshold if moshi_config else 0.5,
        vad_hits_required=moshi_config.stt_vad_hits if moshi_config else 2,
        silence=moshi_config.stt_silence if moshi_config else True,
        silence_threshold=moshi_config.stt_silence_threshold if moshi_config else 0.01,
        silence_ms=moshi_config.stt_silence_ms if moshi_config else 700,
        min_speech_ms=moshi_config.stt_min_speech_ms if moshi_config else 200,
    )
    listen_elapsed = time.perf_counter() - listen_start
    logger.info("Moshi STT listen duration: %.2fs", listen_elapsed)
    return transcript
