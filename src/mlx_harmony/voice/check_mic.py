"""Quick microphone permission and input smoke test."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import sounddevice as sd


@contextmanager
def _cbreak_mode() -> Iterator[None]:
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)


def _format_bar(rms: float, width: int = 50) -> str:
    bar_len = min(width, int(rms * 500))
    return "#" * bar_len + " " * (width - bar_len)


def _callback(indata: np.ndarray, _frames: int, _time: object, status: sd.CallbackFlags) -> None:
    if status:
        print(status, flush=True)
    rms = float(np.sqrt(np.mean(indata**2)))
    bar = _format_bar(rms)
    print(f"\rMic level: {rms:0.4f} {bar}", end="", flush=True)


def main() -> None:
    print("Listening... speak into the mic.")
    if sys.stdin.isatty():
        print("Press 'q' to quit.")
        with _cbreak_mode():
            with sd.InputStream(
                channels=1,
                callback=_callback,
                samplerate=16000,
                blocksize=8000,
            ):
                while True:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                        if sys.stdin.read(1).lower() == "q":
                            print()
                            break
                    time.sleep(0.05)
        return

    print("stdin is not a TTY; use Ctrl+C to stop.")
    with sd.InputStream(
        channels=1,
        callback=_callback,
        samplerate=16000,
        blocksize=8000,
    ):
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
