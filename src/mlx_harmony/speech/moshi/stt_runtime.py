from __future__ import annotations

import queue
import time
from pathlib import Path

from mlx_harmony.logging import get_logger
from mlx_harmony.speech.moshi.shared import (
    _STT_BLOCKSIZE,
    _STT_SAMPLE_RATE,
    _STT_WARMUP_BLOCKS,
    _import_stt_deps,
    _load_json_config,
    _resolve_model_file,
)

logger = get_logger(__name__)


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
            pre_drain = 0
            while True:
                try:
                    block_queue.get_nowait()
                    pre_drain += 1
                except queue.Empty:
                    break
            if pre_drain:
                logger.debug("STT pre-drain cleared %d blocks", pre_drain)

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
                                continue
                            speech_started = True
                            if max_speech_seconds is not None:
                                speech_deadline = time.time() + max_speech_seconds
                    else:
                        if speech_started:
                            silence_elapsed_ms += block_duration_ms
                        else:
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


__all__ = ["MoshiSTT"]
