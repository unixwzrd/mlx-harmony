from __future__ import annotations

import queue
import threading
from typing import Any

from mlx_harmony.config import MoshiConfig


class TTSStreamController:
    def __init__(self, moshi_tts: Any | None, moshi_config: MoshiConfig | None) -> None:
        self._moshi_tts = moshi_tts
        self._moshi_config = moshi_config
        self._queue: queue.Queue[str | None] | None = None
        self._thread: threading.Thread | None = None
        self._buffer = ""

    def start(self) -> None:
        if not self._moshi_tts or not self._moshi_config or not self._moshi_config.tts_stream:
            return
        if self._moshi_config.barge_in:
            raise RuntimeError("TTS streaming does not support barge-in yet.")
        self._queue = queue.Queue()

        def _tts_worker() -> None:
            while True:
                chunk = self._queue.get()
                if chunk is None:
                    break
                self._moshi_tts.speak(chunk)

        self._thread = threading.Thread(target=_tts_worker, daemon=True)
        self._thread.start()

    def enqueue_chunk(self, chunk: str) -> None:
        if not self._queue:
            return
        if chunk.strip():
            self._queue.put(chunk.strip())

    def flush(self) -> None:
        if self._queue is None:
            return
        if self._buffer.strip():
            self.enqueue_chunk(self._buffer)
        self._buffer = ""
        self._queue.put(None)
        if self._thread:
            self._thread.join()

    def reset(self) -> None:
        self._buffer = ""
        if self._queue is None:
            return
        self._queue.put(None)
        if self._thread:
            self._thread.join()

    def maybe_emit(self, delta: str, channel: str | None, role: object | None) -> None:
        if not self._moshi_config or not self._moshi_config.tts_stream:
            return
        if role is not None and str(role) != "Role.ASSISTANT":
            return
        if channel not in (None, "final", "commentary"):
            return
        self._buffer += delta
        chunk_limit = self._moshi_config.tts_chunk_chars
        min_chars = self._moshi_config.tts_chunk_min_chars
        while True:
            if len(self._buffer) < min_chars:
                return
            cut_at = -1
            for mark in (".", "!", "?", ";", ":"):
                idx = self._buffer.rfind(mark, 0, chunk_limit + 1)
                if idx > cut_at:
                    cut_at = idx
            if cut_at == -1 and len(self._buffer) < chunk_limit:
                return
            if cut_at == -1:
                cut_at = chunk_limit
            chunk = self._buffer[: cut_at + 1].strip()
            self._buffer = self._buffer[cut_at + 1 :].lstrip()
            self.enqueue_chunk(chunk)
