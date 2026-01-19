from __future__ import annotations

import json
import queue
import re
import threading
import time
import types
from pathlib import Path
from typing import Optional

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


class MoshiSTT:
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        *,
        block_ms: int = 80,
        warmup_blocks: int = _STT_WARMUP_BLOCKS,
    ) -> None:
        mx, nn, np, rustymimi, sentencepiece, sd, models, utils = _import_stt_deps()
        self._mx = mx
        self._nn = nn
        self._np = np
        self._rustymimi = rustymimi
        self._sentencepiece = sentencepiece
        self._sd = sd
        self._models = models
        self._utils = utils
        self._blocksize = max(1, int(_STT_SAMPLE_RATE * (block_ms / 1000.0)))
        self._warmup_blocks = max(0, int(warmup_blocks))

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
        gen.batch_size = 1
        gen.gen_sequence = mx.full(
            shape=(gen.batch_size, gen.num_codebooks, gen.max_steps),
            vals=gen.ungenerated_token,
            dtype=mx.int32,
        )

        self._text_tokenizer = text_tokenizer
        self._audio_tokenizer = audio_tokenizer
        self._gen = gen
        self._other_codebooks = other_codebooks
        self._stt_shape_logged: bool = False

    def _reset_stt_state(self) -> None:
        if hasattr(self._audio_tokenizer, "reset_state"):
            self._audio_tokenizer.reset_state()  # type: ignore[call-arg]
        elif hasattr(self._audio_tokenizer, "reset_all"):
            self._audio_tokenizer.reset_all()  # type: ignore[call-arg]
        gen = self._gen
        try:
            gen.step_idx = 0
            gen.batch_size = 1
            gen.gen_sequence = self._mx.full(
                shape=(gen.batch_size, gen.num_codebooks, gen.max_steps),
                vals=gen.ungenerated_token,
                dtype=self._mx.int32,
            )
            logger.debug("STT generator state reset")
        except Exception as exc:
            logger.warning("Moshi STT generator reset failed: %s", exc)

    def listen_once(
        self,
        max_seconds: float | None = 8.0,
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_hits_required: int = 2,
        silence: bool = True,
        silence_threshold: float = 0.01,
        silence_ms: int = 700,
        min_speech_ms: int = 200,
        max_speech_seconds: float | None = None,
    ) -> str:
        block_queue: queue.Queue = queue.Queue(maxsize=4)
        transcript_parts: list[str] = []
        vad_hits = 0
        speech_started = False
        speech_ms = 0.0
        silence_elapsed_ms = 0.0

        def audio_callback(indata, _frames, _time, _status):
            try:
                block_queue.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest audio when we're behind to avoid carrying it forward.
                try:
                    block_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    block_queue.put_nowait(indata.copy())
                except queue.Full:
                    logger.debug("STT block queue overflow; dropping audio block")

        start_time = time.time()
        listen_deadline: float | None
        if max_seconds is None or max_seconds <= 0:
            listen_deadline = None
        else:
            listen_deadline = start_time + max_seconds
        speech_deadline: float | None = None
        self._reset_stt_state()
        with self._sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=_STT_SAMPLE_RATE,
            blocksize=_STT_BLOCKSIZE,
            callback=audio_callback,
        ):
            # Clear any queued audio from prior cycles before warmup.
            pre_drain = 0
            while True:
                try:
                    block_queue.get_nowait()
                    pre_drain += 1
                except queue.Empty:
                    break
            if pre_drain:
                logger.debug("STT pre-drain cleared %d blocks", pre_drain)

            # Drop a couple of initial blocks to avoid stale buffered audio.
            for _ in range(_STT_WARMUP_BLOCKS):
                try:
                    block_queue.get(timeout=0.2)
                except queue.Empty:
                    break
            while True:
                now = time.time()
                if listen_deadline is not None and not speech_started and now > listen_deadline:
                    break
                if speech_deadline is not None and now > speech_deadline:
                    break
                block = block_queue.get()
                block_duration_ms = (block.shape[0] / _STT_SAMPLE_RATE) * 1000.0
                if silence:
                    rms = float((block**2).mean() ** 0.5)
                    if rms >= silence_threshold:
                        speech_ms += block_duration_ms
                        silence_elapsed_ms = 0.0
                        if not speech_started:
                            if speech_ms < min_speech_ms:
                                # Wait until we have sustained speech before tokenization.
                                continue
                            speech_started = True
                            if max_speech_seconds is not None:
                                speech_deadline = time.time() + max_speech_seconds
                    else:
                        if speech_started:
                            silence_elapsed_ms += block_duration_ms
                        else:
                            # Reset speech accumulator while idle.
                            speech_ms = 0.0
                            continue
                block = block[None, :, 0]
                block = self._np.ascontiguousarray(
                    block[:, None, :], dtype=self._np.float32
                )
                other_audio_tokens = self._audio_tokenizer.encode_step(block)
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
                if (
                    silence
                    and speech_started
                    and speech_ms >= min_speech_ms
                    and silence_elapsed_ms >= silence_ms
                    and transcript_parts
                ):
                    break

            # Drain any queued blocks so they don't bleed into the next listen cycle.
            drained_blocks = 0
            while True:
                try:
                    block_queue.get_nowait()
                    drained_blocks += 1
                except queue.Empty:
                    break
            if drained_blocks:
                logger.debug("STT drained %d queued blocks", drained_blocks)

        transcript = "".join(transcript_parts).strip()
        logger.debug("STT transcript length=%d text=%r", len(transcript), transcript)
        return transcript

    def _normalize_stt_tokens(self, tokens) -> object:
        arr = self._mx.array(tokens)
        if not self._stt_shape_logged:
            logger.info("STT raw tokens shape=%s", getattr(arr, "shape", "unknown"))
            self._stt_shape_logged = True

        if getattr(arr, "ndim", 0) < 1:
            raise RuntimeError("STT token tensor has no dimensions")

        codebook_axis = None
        for axis, size in enumerate(arr.shape):
            if size == self._other_codebooks:
                codebook_axis = axis
                break
        if codebook_axis is None:
            raise RuntimeError(
                f"STT token tensor missing codebook axis {self._other_codebooks}: {arr.shape}"
            )

        if codebook_axis != 1:
            axes = list(range(arr.ndim))
            axes.pop(codebook_axis)
            axes.insert(1, codebook_axis)
            arr = self._mx.transpose(arr, axes)

        while arr.ndim > 3 and arr.shape[-1] == 1:
            arr = self._mx.squeeze(arr, axis=-1)

        if arr.ndim == 2:
            # (frames, codebooks) -> take last frame
            arr = arr[-1:, :]
            arr = arr.reshape(1, self._other_codebooks, 1)
        else:
            if arr.ndim > 3:
                arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            arr = arr[:1, :, -1:]

        if arr.shape[1] != self._other_codebooks:
            raise RuntimeError(
                f"STT token codebook mismatch: expected {self._other_codebooks}, got {arr.shape}"
            )

        arr = arr.astype(self._mx.int32)
        return arr

    def _normalize_stt_tokens(self, tokens) -> object:
        arr = tokens
        logger.debug("STT raw token shape=%s", getattr(arr, "shape", "unknown"))

        if getattr(arr, "ndim", 0) < 1:
            raise RuntimeError("STT token tensor has no dimensions")

        codebook_axis = None
        for axis, size in enumerate(arr.shape):
            if size == self._other_codebooks:
                codebook_axis = axis
                break
        if codebook_axis is None:
            raise RuntimeError(f"STT token tensor missing codebook axis: {arr.shape}")

        if codebook_axis != 1:
            axes = list(range(arr.ndim))
            axes.pop(codebook_axis)
            axes.insert(1, codebook_axis)
            arr = self._mx.transpose(arr, axes)

        if arr.shape[1] > self._other_codebooks:
            arr = arr[:, : self._other_codebooks, ...]

        if arr.ndim == 1:
            arr = arr[None, :, None]
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim > 3:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)

        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected STT token shape after collapse: {arr.shape}")

        arr = arr[:, :, -1:]

        if arr.shape[0] != 1:
            logger.warning("Unexpected STT batch size %s; using first batch", arr.shape[0])
            arr = arr[:1]

        arr = arr.astype(self._mx.int32)
        logger.debug("STT normalized token shape=%s", getattr(arr, "shape", "unknown"))
        return arr


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
        sample_count = 0
        interrupted = False

        class _TTSInterrupted(RuntimeError):
            pass

        def _on_frame(frame):
            nonlocal frame_count, sample_count
            if stop_event.is_set():
                raise _TTSInterrupted()
            if (frame == -1).any():
                return
            _pcm = self._tts_model.mimi.decode_step(frame[:, :, None])
            _pcm = self._np.array(self._mx.clip(_pcm[0, 0], -1, 1))
            wav_frames.put_nowait(_pcm)
            frame_count += 1
            sample_count += int(_pcm.size)

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
            tts_start = time.perf_counter()
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
            time.sleep(0.1)
            tts_elapsed = time.perf_counter() - tts_start
            if sample_count:
                audio_seconds = sample_count / float(self._tts_model.mimi.sample_rate)
                rtf = audio_seconds / tts_elapsed if tts_elapsed > 0 else 0.0
                logger.info(
                    "Moshi TTS audio: %.2fs (RTF %.2fx)",
                    audio_seconds,
                    rtf,
                )
        if interrupted:
            logger.info("Moshi TTS interrupted by barge-in.")


def chunk_text(
    text: str,
    max_chars: int,
    sentence_breaks: bool,
    *,
    min_chars: int = 60,
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if max_chars <= 0:
        return [stripped]
    if not sentence_breaks:
        raw_chunks = [stripped[i : i + max_chars].strip() for i in range(0, len(stripped), max_chars)]
    else:
        sentences = re.split(r"(?<=[.!?;:])\\s+", stripped)
        raw_chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                remaining = sentence
                while remaining:
                    if len(remaining) <= max_chars:
                        raw_chunks.append(remaining.strip())
                        break
                    split_at = remaining.rfind(" ", 0, max_chars + 1)
                    if split_at <= 0:
                        split_at = max_chars
                    chunk = remaining[:split_at].strip()
                    if chunk:
                        raw_chunks.append(chunk)
                    remaining = remaining[split_at:].strip()
                continue
            projected = current_len + len(sentence) + (1 if current else 0)
            if current and projected > max_chars:
                raw_chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len = projected if current_len else len(sentence)
        if current:
            raw_chunks.append(" ".join(current).strip())

    if not raw_chunks:
        return []

    merged: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    target_min = max(0, min_chars)
    for chunk in raw_chunks:
        if not chunk:
            continue
        projected = buffer_len + len(chunk) + (1 if buffer else 0)
        if buffer and projected > max_chars:
            merged.append(" ".join(buffer).strip())
            buffer = [chunk]
            buffer_len = len(chunk)
            continue
        buffer.append(chunk)
        buffer_len = projected
        if buffer_len >= target_min:
            merged.append(" ".join(buffer).strip())
            buffer = []
            buffer_len = 0
    if buffer:
        merged.append(" ".join(buffer).strip())
    return merged
