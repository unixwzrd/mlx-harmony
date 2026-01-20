from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path

from mlx_harmony.logging import get_logger
from mlx_harmony.speech.moshi.shared import (
    _TTS_BLOCKSIZE,
    _import_tts_deps,
    _load_json_config,
    _resolve_model_file,
)

logger = get_logger(__name__)


class MoshiTTS:
    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        voice_path: str | None = None,
        quantize: int | None = None,
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
        sample_count = 0
        interrupted = False

        class _TTSInterrupted(RuntimeError):
            pass

        def _on_frame(frame):
            nonlocal sample_count
            if stop_event.is_set():
                raise _TTSInterrupted()
            if (frame == -1).any():
                return
            pcm = self._tts_model.mimi.decode_step(frame[:, :, None])
            pcm = self._np.array(self._mx.clip(pcm[0, 0], -1, 1))
            wav_frames.put_nowait(pcm)
            sample_count += int(pcm.size)

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
        raw_chunks = [
            stripped[i : i + max_chars].strip() for i in range(0, len(stripped), max_chars)
        ]
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


__all__ = ["MoshiTTS", "chunk_text"]
