from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Optional

from mlx_harmony.config import MoshiConfig
from mlx_harmony.logging import get_logger

logger = get_logger(__name__)

_STT_SAMPLE_RATE = 24000
_STT_BLOCKSIZE = 1920
_TTS_BLOCKSIZE = 1920


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


def _import_stt_deps() -> tuple[object, object, object, object, object, object, object]:
    """Import optional STT dependencies only when Moshi STT is enabled."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import rustymimi
        import sentencepiece
        import sounddevice as sd
        from moshi_mlx import models, utils
    except ImportError as exc:
        raise RuntimeError(
            "Moshi STT requires moshi-mlx, rustymimi, sentencepiece, and sounddevice."
        ) from exc
    return mx, nn, rustymimi, sentencepiece, sd, models, utils


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


class MoshiSTT:
    def __init__(self, model_path: str, config_path: str | None = None) -> None:
        mx, nn, rustymimi, sentencepiece, sd, models, utils = _import_stt_deps()
        self._mx = mx
        self._nn = nn
        self._rustymimi = rustymimi
        self._sentencepiece = sentencepiece
        self._sd = sd
        self._models = models
        self._utils = utils

        model_dir = Path(model_path)
        config_file = Path(config_path) if config_path else (model_dir / "config.json")
        config = _load_json_config(config_file)
        mimi_name = str(config["mimi_name"])
        moshi_name = str(config.get("moshi_name", "model.safetensors"))
        tokenizer_name = str(config["tokenizer_name"])

        mimi_weights = _resolve_model_file(model_dir, mimi_name, "mimi")
        moshi_weights = _resolve_model_file(model_dir, moshi_name, "moshi")
        tokenizer_path = _resolve_model_file(model_dir, tokenizer_name, "tokenizer")

        lm_config = models.LmConfig.from_config_dict(config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        if str(moshi_weights).endswith(".q4.safetensors"):
            nn.quantize(model, bits=4, group_size=32)
        elif str(moshi_weights).endswith(".q8.safetensors"):
            nn.quantize(model, bits=8, group_size=64)

        logger.info("Loading Moshi STT weights from %s", moshi_weights)
        model.load_weights(str(moshi_weights), strict=True)

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))  # type: ignore

        generated_codebooks = lm_config.generated_codebooks
        other_codebooks = lm_config.other_codebooks
        mimi_codebooks = max(generated_codebooks, other_codebooks)
        audio_tokenizer = rustymimi.Tokenizer(str(mimi_weights), num_codebooks=mimi_codebooks)  # type: ignore

        model.warmup()
        gen = models.LmGen(
            model=model,
            max_steps=4096,
            text_sampler=utils.Sampler(top_k=25, temp=0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

        self._text_tokenizer = text_tokenizer
        self._audio_tokenizer = audio_tokenizer
        self._gen = gen
        self._other_codebooks = other_codebooks

    def listen_once(
        self,
        max_seconds: float = 8.0,
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_hits_required: int = 2,
    ) -> str:
        block_queue: queue.Queue = queue.Queue()
        transcript_parts: list[str] = []
        vad_hits = 0

        def audio_callback(indata, _frames, _time, _status):
            block_queue.put(indata.copy())

        start_time = time.time()
        with self._sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=_STT_SAMPLE_RATE,
            blocksize=_STT_BLOCKSIZE,
            callback=audio_callback,
        ):
            while time.time() - start_time < max_seconds:
                block = block_queue.get()
                block = block[None, :, 0]
                other_audio_tokens = self._audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = self._mx.array(other_audio_tokens).transpose(0, 2, 1)[
                    :, :, : self._other_codebooks
                ]
                if vad:
                    if not hasattr(self._gen, "step_with_extra_heads"):
                        raise RuntimeError("STT model does not support VAD heads.")
                    text_token, vad_heads = self._gen.step_with_extra_heads(
                        other_audio_tokens[0]
                    )
                    if vad_heads:
                        pr_vad = float(vad_heads[2][0, 0, 0].item())
                        if pr_vad > vad_threshold:
                            vad_hits += 1
                else:
                    text_token = self._gen.step(other_audio_tokens[0])
                token_id = int(text_token[0].item())
                if token_id not in (0, 3):
                    piece = self._text_tokenizer.id_to_piece(token_id)  # type: ignore
                    piece = piece.replace("▁", " ")
                    transcript_parts.append(piece)
                    vad_hits = 0
                if vad and vad_hits >= vad_hits_required and transcript_parts:
                    break

        transcript = "".join(transcript_parts).strip()
        return transcript


class MoshiTTS:
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        voice_path: Optional[str] = None,
        quantize: Optional[int] = None,
    ) -> None:
        mx, nn, np, sentencepiece, sd, models, TTSModel = _import_tts_deps()
        self._mx = mx
        self._nn = nn
        self._np = np
        self._sd = sd
        self._models = models
        self._tts_model_cls = TTSModel
        self._voice_path = Path(voice_path) if voice_path else None
        if self._voice_path:
            if not self._voice_path.exists():
                raise FileNotFoundError(f"Missing Moshi voice embedding at {self._voice_path}")
            if self._voice_path.suffix.lower() != ".safetensors":
                raise RuntimeError(
                    "Moshi TTS expects a .safetensors voice embedding file. "
                    "Choose a voice embedding from the voices repo and set tts_voice_path "
                    "to that .safetensors file (not the .wav reference clip)."
                )

        model_dir = Path(model_path)
        config_file = Path(config_path) if config_path else (model_dir / "config.json")
        config = _load_json_config(config_file)
        mimi_name = str(config["mimi_name"])
        moshi_name = str(config.get("moshi_name", "model.safetensors"))
        tokenizer_name = str(config["tokenizer_name"])

        mimi_weights = _resolve_model_file(model_dir, mimi_name, "mimi")
        moshi_weights = _resolve_model_file(model_dir, moshi_name, "moshi")
        tokenizer_path = _resolve_model_file(model_dir, tokenizer_name, "tokenizer")

        lm_config = models.LmConfig.from_config_dict(config)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        logger.info("Loading Moshi TTS weights from %s", moshi_weights)
        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if quantize is not None:
            logger.info("Quantizing Moshi TTS model to %s bits", quantize)
            nn.quantize(model.depformer, bits=quantize)
            for layer in model.transformer.layers:
                nn.quantize(layer.self_attn, bits=quantize)
                nn.quantize(layer.gating, bits=quantize)

        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))  # type: ignore
        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            temp=0.6,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=config,
        )

        cfg_coef_conditioning = None
        if tts_model.valid_cfg_conditionings:
            cfg_coef_conditioning = tts_model.cfg_coef
            tts_model.cfg_coef = 1.0

        self._tts_model = tts_model
        self._cfg_coef_conditioning = cfg_coef_conditioning

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        if not text.strip():
            return
        stop_event = stop_event or threading.Event()
        wav_frames: queue.Queue = queue.Queue()
        frame_count = 0
        interrupted = False

        class _TTSInterrupted(RuntimeError):
            pass

        def _on_frame(frame):
            nonlocal frame_count
            if stop_event.is_set():
                raise _TTSInterrupted()
            if (frame == -1).any():
                return
            _pcm = self._tts_model.mimi.decode_step(frame[:, :, None])
            _pcm = self._np.array(self._mx.clip(_pcm[0, 0], -1, 1))
            wav_frames.put_nowait(_pcm)
            frame_count += 1

        all_entries = [self._tts_model.prepare_script([text])]
        if self._tts_model.multi_speaker and self._voice_path:
            voices = [self._voice_path]
        else:
            voices = []
        all_attributes = [
            self._tts_model.make_condition_attributes(voices, self._cfg_coef_conditioning)
        ]

        def _audio_callback(outdata, _a, _b, _c):
            try:
                pcm_data = wav_frames.get(block=False)
                outdata[:, 0] = pcm_data
            except queue.Empty:
                outdata[:] = 0

        with self._sd.OutputStream(
            samplerate=self._tts_model.mimi.sample_rate,
            blocksize=_TTS_BLOCKSIZE,
            channels=1,
            callback=_audio_callback,
        ):
            try:
                self._tts_model.generate(
                    all_entries,
                    all_attributes,
                    cfg_is_no_prefix=True,
                    cfg_is_no_text=True,
                    on_frame=_on_frame,
                )
            except _TTSInterrupted:
                interrupted = True
            time.sleep(0.5)
        if interrupted:
            logger.info("Moshi TTS interrupted by barge-in.")


def chunk_text(text: str, max_chars: int, sentence_breaks: bool) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if max_chars <= 0:
        return [stripped]
    if not sentence_breaks:
        return [stripped[i : i + max_chars].strip() for i in range(0, len(stripped), max_chars)]

    sentences = re.split(r"(?<=[.!?])\\s+", stripped)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            for idx in range(0, len(sentence), max_chars):
                chunk = sentence[idx : idx + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            continue
        if current_len + len(sentence) + (1 if current else 0) > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + (1 if current_len else 0)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks
