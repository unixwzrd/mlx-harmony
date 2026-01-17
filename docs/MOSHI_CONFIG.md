# Moshi Config

**Created**: 2026-01-15
**Updated**: 2026-01-17

This document describes the Moshi voice configuration file and CLI overrides.

## Overview

The Moshi config file is optional. When provided, it supplies defaults for STT/TTS paths and options, while CLI flags override any values in the file.

If `configs/moshi.json` exists and `--moshi-config` is not supplied, it is auto-loaded when `--moshi` is set.

## Example Config

See the example file at [moshi-config.example.json](../configs/moshi-config.example.json).

```json
{
  "enabled": true,
  "stt_model_path": "models/stt-2.6b-en-mlx",
  "stt_config_path": null,
  "stt_max_seconds": 8.0,
  "stt_vad": false,
  "stt_vad_threshold": 0.5,
  "stt_vad_hits": 2,
  "stt_silence": true,
  "stt_silence_threshold": 0.01,
  "stt_silence_ms": 700,
  "stt_min_speech_ms": 200,
  "stt_block_ms": 80,
  "stt_warmup_blocks": 2,
  "tts_model_path": "models/TTS/tts-1.6b-en_fr",
  "tts_config_path": null,
  "tts_voice_path": "models/TTS/moshi-tts-voices/ears/p001/freeform_speech_01.wav.1e68beda@240.safetensors",
  "quantize": 8,
  "tts_chunk_chars": 180,
  "tts_chunk_sentences": true,
  "tts_chunk_min_chars": 60,
  "tts_stream": false,
  "use_stt": true,
  "use_tts": true,
  "smoke_test": false,
  "barge_in": false,
  "barge_in_window_seconds": 2.0
}
```

## CLI Overrides

CLI flags override any values from the config file:

- `--moshi`
- `--moshi-config <path>`
- `--moshi-stt-path <path>`
- `--moshi-max-seconds <seconds>`
- `--moshi-vad` / `--no-moshi-vad`
- `--moshi-vad-threshold <float>`
- `--moshi-vad-hits <int>`
- `--moshi-silence` / `--no-moshi-silence`
- `--moshi-silence-threshold <float>`
- `--moshi-silence-ms <int>`
- `--moshi-min-speech-ms <int>`
- `--moshi-stt-block-ms <int>`
- `--moshi-stt-warmup-blocks <int>`
- `--moshi-tts-path <path>`
- `--moshi-voice-path <path>`
- `--moshi-quantize <4|8>`
- `--moshi-tts-chunk-chars <int>`
- `--moshi-tts-chunk-sentences` / `--no-moshi-tts-chunk-sentences`
- `--moshi-tts-chunk-min-chars <int>`
- `--moshi-tts-stream` / `--no-moshi-tts-stream`
- `--moshi-stt` / `--no-moshi-stt`
- `--moshi-tts` / `--no-moshi-tts`
- `--moshi-smoke`
- `--moshi-barge-in` / `--no-moshi-barge-in`
- `--moshi-barge-in-window <seconds>`
- `--no-fs-cache` (global): Disable filesystem cache when reading model weights (macOS only, experimental).

## Notes

- Set `enabled` to true in the config when you want Moshi to run by default, or pass `--moshi` in the CLI.
- If a path is missing, Moshi voice mode fails fast with a clear error.
- If `configs/moshi.json` exists and `--moshi-config` is not supplied, it will be used automatically when `--moshi` is set.
- `--no-fs-cache` is a global loader flag (not Moshi-specific) and is intended only for load-time experiments.

## Model Directory Layout

We recommend keeping Moshi assets under `models/` to simplify configuration and portability. A typical layout looks like:

```text
models/
  stt-2.6b-en-mlx/
    config.json
    model.safetensors
    tokenizer_en_audio_4000.model
    mimi-pytorch-*.safetensors
  TTS/
    tts-1.6b-en_fr/
      config.json
      model.safetensors
      tokenizer_spm_32k_3.model
      mimi-*.safetensors
    moshi-tts-voices/
      ears/
        p001/
          freeform_speech_01.wav.1e68beda@240.safetensors
```

Paths in `moshi.json` should point to these local directories (and a voice `.wav` inside the voices repo).

## Voice Selection

The TTS model conditions on a voice embedding file (a `.safetensors` file in the voices repo). Choose a voice embedding and set `tts_voice_path` to its local path. The `.wav` files are reference clips, not the embeddings that `moshi-mlx` loads.

## Utilities

Use the `hotmic` CLI utility (installed with `mlx-harmony`) to confirm microphone permissions and live input levels before enabling `--moshi`. It only checks the microphone. For a speaker check, run a quick TTS smoke test with `--moshi-smoke` or play a system audio file.

### hotmic

`hotmic` provides a VU-style dBFS meter, a peak-hold indicator, and speech threshold markers to help tune STT silence detection. It also includes interactive controls to adjust smoothing and speech thresholds.

```bash
hotmic
hotmic --block-ms 100 --sample-rate 16000
hotmic --speech-lo-dbfs -30 --speech-release-ms 500
hotmic --h-lag-db 2
```

#### What the meter shows

- **inst dBFS**: instantaneous level (log scale).
- **avg dBFS**: smoothed/rolling level (attack/decay controls).
- **0 VU**: reference at -18 dBFS.
- **Peak-hold**: a fixed marker (|) that shows the highest recent peak, with a slow decay. The decay rate follows the meter decay setting so it stays visually consistent.
- **L/H markers**: speech low threshold (L) and a high-water reference (H).

#### Keys

- `q`: quit
- `a` / `d`: select Attack / Decay
- `l`: select low speech threshold
- `r`: select speech release delay
- `-` / `=`: decrease / increase selected value

#### Useful CLI flags

- `--block-ms`: how often the meter updates (lower = faster response).
- `--sample-rate`: audio sample rate (typically 16k).
- `--speech-lo-dbfs`: low threshold to trigger SPEAK.
- `--speech-release-ms`: delay before SPEAK turns off after crossing below L.
- `--h-lag-db`: how far the high-water marker trails peaks.

## Parameters

All parameters below are optional unless noted.

### Core

- `enabled` (bool): Enable Moshi voice mode defaults from config.
- `use_stt` (bool): Enable STT in voice mode.
- `use_tts` (bool): Enable TTS in voice mode.
- `smoke_test` (bool): Run a short STT/TTS smoke test, then exit.

### STT (Speech-to-Text)

- `stt_model_path` (string, required if `use_stt`): Local path to the STT MLX model directory.
- `stt_config_path` (string | null): Optional explicit path to STT `config.json`.
- `stt_max_seconds` (float): Maximum seconds to listen per utterance.
- `stt_vad` (bool): Enable VAD-based end-of-utterance detection when supported.
- `stt_vad_threshold` (float): VAD probability threshold (higher = stricter).
- `stt_vad_hits` (int): Consecutive VAD hits required to end an utterance.
- `stt_silence` (bool): Enable RMS-based silence detection (works on all STT models).
- `stt_silence_threshold` (float): RMS threshold below which audio is treated as silence.
- `stt_silence_ms` (int): Trailing silence required to end an utterance.
- `stt_min_speech_ms` (int): Minimum speech duration before we start decoding and before silence can end the utterance.
- `stt_block_ms` (int): STT audio block duration in milliseconds (larger reduces CPU).
- `stt_warmup_blocks` (int): Initial blocks to discard to avoid buffered audio from a previous listen.

### TTS (Text-to-Speech)

- `tts_model_path` (string, required if `use_tts`): Local path to the TTS MLX model directory.
- `tts_config_path` (string | null): Optional explicit path to TTS `config.json`.
- `tts_voice_path` (string | null): Optional voice embedding path for multi-speaker models.
- `quantize` (int | null): Quantize TTS model weights (commonly 4 or 8).
- `tts_chunk_chars` (int): Max characters per TTS chunk.
- `tts_chunk_sentences` (bool): Prefer sentence boundaries when chunking.
- `tts_chunk_min_chars` (int): Minimum characters per chunk (helps avoid tiny fragments).
- `tts_stream` (bool): Stream TTS chunks during generation (asynchronous playback).

### Barge-In

- `barge_in` (bool): Interrupt TTS if user speaks during playback.
- `barge_in_window_seconds` (float): How long to listen for barge-in during playback.

[Back to README](../README.md)
