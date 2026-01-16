from __future__ import annotations

import json
from pathlib import Path

from mlx_harmony.config import MoshiConfig, load_moshi_config
from mlx_harmony.voice.voice_moshi import chunk_text


def test_load_moshi_config_round_trip(tmp_path: Path) -> None:
    payload = {
        "enabled": True,
        "stt_model_path": "models/STT/stt-2.6b-en-mlx",
        "stt_max_seconds": 6.5,
        "stt_vad": True,
        "stt_vad_threshold": 0.6,
        "stt_vad_hits": 3,
        "tts_model_path": "models/TTS/moshiko-mlx-q8",
        "tts_voice_path": None,
        "quantize": 8,
        "tts_chunk_chars": 120,
        "tts_chunk_sentences": True,
        "use_stt": True,
        "use_tts": False,
        "smoke_test": False,
        "barge_in": True,
        "barge_in_window_seconds": 1.5,
    }
    config_path = tmp_path / "moshi.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_moshi_config(str(config_path))
    assert isinstance(config, MoshiConfig)
    assert config.stt_max_seconds == 6.5
    assert config.stt_vad_threshold == 0.6
    assert config.stt_vad_hits == 3
    assert config.tts_chunk_chars == 120
    assert config.barge_in is True
    assert config.barge_in_window_seconds == 1.5


def test_chunk_text_sentence_mode() -> None:
    text = "Hello world. This is a test! Another sentence?"
    chunks = chunk_text(text, max_chars=20, sentence_breaks=True)
    assert chunks
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_chunk_text_fixed_mode() -> None:
    text = "a" * 50
    chunks = chunk_text(text, max_chars=16, sentence_breaks=False)
    assert chunks
    assert all(len(chunk) <= 16 for chunk in chunks)


def test_moshi_smoke_flag_no_deps(monkeypatch):
    """Ensure the smoke flag can be parsed without Moshi deps present."""
    import argparse

    from mlx_harmony.cli.cli_args import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--moshi",
            "--moshi-smoke",
            "--moshi-stt-path",
            "models/STT/stt-2.6b-en-mlx",
            "--moshi-tts-path",
            "models/TTS/moshiko-mlx-q8",
        ]
    )
    assert args.moshi is True
    assert args.moshi_smoke is True
