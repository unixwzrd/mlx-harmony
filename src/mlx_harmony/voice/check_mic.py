"""Quick microphone permission and input smoke test."""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import time
import tty
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.text import Text

_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_BLOCK_MS = 500
_DEFAULT_WINDOW_SECONDS = 5.0
_DEFAULT_CALIBRATION_SECONDS = 3.0
_RUN_SAMPLE_WINDOW = 400
_MIN_DBFS = -60.0
_YELLOW_DBFS = -20.0
_RED_DBFS = -6.0
_VU_DBFS = -18.0
_EPS = 1e-9
_BAR_WIDTH = 50
_PEAK_HOLD_DECAY_DB_PER_SEC = 6.0
_PEAK_HOLD_DECAY_MIN = 1.5
_PEAK_HOLD_DECAY_MAX = 12.0
_BAR_FULL = "█"
_BAR_HALF = "▌"
_BAR_EMPTY_CHAR = " "
_ATTACK_DEFAULT = 0.12
_DECAY_DEFAULT = 0.30
_ATTACK_MIN = 0.01
_DECAY_MIN = 0.02
_ATTACK_MAX = 2.0
_DECAY_MAX = 3.0
_ATTACK_STEP = 0.05
_DECAY_STEP = 0.05
_SPEECH_RELEASE_MS = 300
_SPEECH_HYSTERESIS_DB = 3.0
_H_LAG_DEFAULT_DB = 2.0
_RELEASE_STEP_MS = 50


@contextmanager
def _cbreak_mode() -> Iterator[None]:
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)


def _dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(rms, _EPS))


def _normalize_dbfs(dbfs: float, min_dbfs: float = _MIN_DBFS) -> float:
    if dbfs <= min_dbfs:
        return 0.0
    return min(1.0, (dbfs - min_dbfs) / abs(min_dbfs))


def _bar_style_db(dbfs: float) -> str:
    if dbfs >= _RED_DBFS:
        return "bright_red"
    if dbfs >= _YELLOW_DBFS:
        return "yellow"
    return "green"


def _format_bar(level: float, width: int = 50) -> str:
    level = min(1.0, max(0.0, level))
    full_units = int(level * width)
    remainder = (level * width) - full_units
    half = 1 if remainder >= 0.5 and full_units < width else 0
    bar = _BAR_FULL * full_units
    if half:
        bar += _BAR_HALF
    return bar.ljust(width, _BAR_EMPTY_CHAR)


class _LevelMeter:
    def __init__(self, window_blocks: int) -> None:
        self._window = deque(maxlen=max(1, window_blocks))
        self._sum = 0.0
        self._peak = 0.0

    def update(self, indata: np.ndarray) -> tuple[float, float, float, float]:
        rms = float(np.sqrt(np.mean(indata**2)))
        clip = float(np.mean(np.abs(indata) >= 0.95) * 100.0)
        if len(self._window) == self._window.maxlen:
            oldest = self._window[0]
            self._sum -= oldest
        self._window.append(rms)
        self._sum += rms
        if rms >= self._peak:
            self._peak = rms
        elif len(self._window) == self._window.maxlen and rms < self._peak:
            self._peak = max(self._window) if self._window else 0.0
        avg = self._sum / max(1, len(self._window))
        peak = self._peak
        return rms, avg, peak, clip


class _Stats:
    def __init__(self) -> None:
        self.noise_samples: list[float] = []
        self.run_samples: deque[float] = deque(maxlen=_RUN_SAMPLE_WINDOW)
        self.max_clip = 0.0

    def record(self, rms: float, clip: float, calibrating: bool) -> None:
        if calibrating:
            self.noise_samples.append(rms)
        else:
            self.run_samples.append(rms)
        if clip > self.max_clip:
            self.max_clip = clip

    def summary(self) -> dict[str, float | None]:
        noise = float(np.median(self.noise_samples)) if self.noise_samples else None
        samples = list(self.run_samples)
        speech = float(np.percentile(samples, 90.0)) if samples else None
        speech_p99 = float(np.percentile(samples, 99.0)) if samples else None
        speech_peak = float(max(samples)) if samples else None
        peak_dbfs = _dbfs(speech_peak) if speech_peak is not None else None
        return {
            "noise_median": noise,
            "speech_p90": speech,
            "speech_p99": speech_p99,
            "speech_peak": speech_peak,
            "speech_peak_dbfs": peak_dbfs,
            "max_clip": self.max_clip,
        }


class _State:
    def __init__(self) -> None:
        self.calibrating = True
        self.threshold: float | None = None
        self.rms = 0.0
        self.dbfs = _MIN_DBFS
        self.display_dbfs = _MIN_DBFS
        self.peak_hold_dbfs = _MIN_DBFS
        self.last_update = time.monotonic()
        self.peak_hold_decay_rate = _PEAK_HOLD_DECAY_DB_PER_SEC
        self.avg = 0.0
        self.peak = 0.0
        self.clip = 0.0
        self.attack_s = _ATTACK_DEFAULT
        self.decay_s = _DECAY_DEFAULT
        self.selected: str | None = None
        self.speech_hi_dbfs = _YELLOW_DBFS
        self.speech_lo_dbfs = _YELLOW_DBFS - _SPEECH_HYSTERESIS_DB
        self.speech_release_ms = _SPEECH_RELEASE_MS
        self.speaking = False
        self.last_voice_time = 0.0
        self.h_marker_dbfs = _YELLOW_DBFS
        self.h_marker_lag_db = _H_LAG_DEFAULT_DB
        self.h_marker_last_update = time.monotonic()


def _callback(
    console: Console,
    state: _State,
    stats: _Stats,
    meter: _LevelMeter,
    indata: np.ndarray,
    _frames: int,
    _time: object,
    status: sd.CallbackFlags,
) -> None:
    if status:
        console.print(status, flush=True)
    rms, avg, peak, clip = meter.update(indata)
    stats.record(rms, clip, state.calibrating)
    state.rms = rms
    state.dbfs = _dbfs(rms)
    state.avg = avg
    state.peak = peak
    state.clip = clip


def _update_peak_hold(state: _State) -> None:
    now = time.monotonic()
    elapsed = max(0.0, now - state.last_update)
    ratio = _DECAY_DEFAULT / max(state.decay_s, _EPS)
    decay_rate = _PEAK_HOLD_DECAY_DB_PER_SEC * math.sqrt(ratio)
    decay_rate = max(_PEAK_HOLD_DECAY_MIN, min(_PEAK_HOLD_DECAY_MAX, decay_rate))
    decay = decay_rate * elapsed
    if state.dbfs >= state.peak_hold_dbfs:
        state.peak_hold_dbfs = state.dbfs
    else:
        state.peak_hold_dbfs = max(_MIN_DBFS, state.peak_hold_dbfs - decay)
    state.last_update = now
    state.peak_hold_decay_rate = decay_rate


def _update_display_dbfs(state: _State) -> None:
    now = time.monotonic()
    elapsed = max(0.0, now - state.last_update)
    if elapsed <= 0:
        return
    if state.dbfs >= state.display_dbfs:
        tau = max(_ATTACK_MIN, state.attack_s)
    else:
        tau = max(_DECAY_MIN, state.decay_s)
    alpha = 1.0 - math.exp(-elapsed / max(tau, _EPS))
    state.display_dbfs = state.display_dbfs + (state.dbfs - state.display_dbfs) * alpha


def _bar_position(dbfs: float) -> int:
    pos = int(round(_normalize_dbfs(dbfs) * _BAR_WIDTH))
    return max(0, min(_BAR_WIDTH - 1, pos))


def _render_ticks() -> tuple[str, str]:
    ticks = [-60, -40, -20, -10, -5, 0]
    line = [" "] * _BAR_WIDTH
    label = [" "] * _BAR_WIDTH
    for tick in ticks:
        idx = _bar_position(float(tick))
        line[idx] = "|"
        label_text = f"{tick}"
        start = max(0, min(_BAR_WIDTH - len(label_text), idx - len(label_text) // 2))
        for i, ch in enumerate(label_text):
            label[start + i] = ch
    return "".join(line), "".join(label)


def _speech_thresholds_from_rms(threshold: float | None) -> tuple[float, float]:
    if threshold is None:
        base = -35.0
    else:
        base = _dbfs(threshold)
    return base + _SPEECH_HYSTERESIS_DB, base


def _update_speaking(state: _State) -> None:
    now = time.monotonic()
    if state.dbfs >= state.speech_lo_dbfs:
        state.speaking = True
        state.last_voice_time = now
        return
    if state.speaking and state.dbfs < state.speech_lo_dbfs:
        if (now - state.last_voice_time) * 1000.0 >= state.speech_release_ms:
            state.speaking = False


def _update_h_marker(state: _State) -> None:
    now = time.monotonic()
    target = min(state.display_dbfs - state.h_marker_lag_db, state.peak_hold_dbfs)
    if target > state.h_marker_dbfs:
        state.h_marker_dbfs = target
    state.h_marker_last_update = now


def _render_speech_markers(state: _State) -> str:
    marker = [" "] * _BAR_WIDTH
    lo_pos = _bar_position(state.speech_lo_dbfs)
    hi_pos = _bar_position(state.h_marker_dbfs)
    marker[lo_pos] = "L"
    marker[hi_pos] = "H"
    return "".join(marker)


def _render_line(state: _State) -> Text:
    _update_display_dbfs(state)
    _update_peak_hold(state)
    _update_h_marker(state)
    _update_speaking(state)
    avg_dbfs = _dbfs(state.avg) if state.avg > 0 else _MIN_DBFS
    bar_style = _bar_style_db(state.display_dbfs)
    bar = _format_bar(_normalize_dbfs(state.display_dbfs), width=_BAR_WIDTH)
    peak_pos = _bar_position(state.peak_hold_dbfs)
    peak_style = _bar_style_db(state.peak_hold_dbfs)
    bar_chars = list(bar)
    bar_chars[peak_pos] = "│"
    bar = "".join(bar_chars)
    ticks, labels = _render_ticks()
    speech_markers = _render_speech_markers(state)
    status_label = "CALIB" if state.calibrating else "LIVE "
    status_style = "yellow" if state.calibrating else "green"
    speech_label = ""
    speech_plain = ""
    if not state.calibrating and state.threshold is not None:
        speech_label = "SPEAK" if state.speaking else "SILENT"
        speech_style = "bright_green" if state.speaking else "bright_black"
        speech_plain = "SPEAK " if state.speaking else "SILENT"
    speech_suffix = f" {speech_label}" if speech_label else ""
    speech_plain_suffix = f" {speech_plain:6s}" if speech_plain else ""
    attack_label = (
        f"Attack: {state.attack_s:0.2f}s"
        if state.selected == "attack"
        else f"Attack: {state.attack_s:0.2f}s"
    )
    decay_label = (
        f"Decay: {state.decay_s:0.2f}s"
        if state.selected == "decay"
        else f"Decay: {state.decay_s:0.2f}s"
    )
    low_label = (
        f"Low: {state.speech_lo_dbfs:5.1f}"
        if state.selected == "low"
        else f"Low: {state.speech_lo_dbfs:5.1f}"
    )
    release_label = (
        f"Release: {state.speech_release_ms:4d}ms"
        if state.selected == "release"
        else f"Release: {state.speech_release_ms:4d}ms"
    )
    prefix_plain = (
        f"Mic inst {state.dbfs:6.1f} dBFS "
        f"avg {avg_dbfs:6.1f} dBFS "
        f"0VU {_VU_DBFS:4.0f} "
        f"clip {state.clip:4.1f}% "
        f"{status_label}{speech_plain_suffix} "
    )
    prefix_padding = " " * len(prefix_plain)
    line = Text()
    line.append("Mic", style="bold cyan")
    line.append(f" inst {state.dbfs:6.1f} dBFS ", style="dim")
    line.append(f"avg {avg_dbfs:6.1f} dBFS ", style="dim")
    line.append(f"0VU {_VU_DBFS:4.0f} ", style="dim")
    line.append(f"clip {state.clip:4.1f}% ", style="dim")
    line.append(status_label, style=status_style)
    if speech_label:
        line.append(" ")
        line.append(f"{speech_label:6s}", style=speech_style)
    line.append(" | ")
    bar_text = Text(bar, style=bar_style)
    bar_text.stylize(peak_style, peak_pos, peak_pos + 1)
    line.append(bar_text)
    line.append("\n")
    line.append(prefix_padding + ticks, style="dim")
    line.append("\n")
    line.append(prefix_padding + labels, style="dim")
    line.append("\n")
    line.append(prefix_padding + speech_markers, style="dim")
    line.append("\n")
    line.append(attack_label, style="bold" if state.selected == "attack" else "dim")
    line.append("  ")
    line.append(decay_label, style="bold" if state.selected == "decay" else "dim")
    line.append("  ")
    line.append(low_label, style="bold" if state.selected == "low" else "dim")
    line.append("  ")
    line.append(
        f"Hold: auto ({state.peak_hold_decay_rate:0.1f} dB/s)", style="dim"
    )
    line.append("  ")
    line.append(f"H ref: auto (lag {state.h_marker_lag_db:0.1f} dB)", style="dim")
    line.append("  ")
    line.append(release_label, style="bold" if state.selected == "release" else "dim")
    line.append("  ")
    line.append("Keys: a/d/l/r select, +/- adjust", style="dim")
    return line


def _update_bar_thresholds(
    _state: _State,
    _noise: float | None,
    _speech: float | None,
    _speech_p99: float | None,
    _speech_peak: float | None,
) -> None:
    return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check microphone input levels and suggest Moshi STT settings."
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=_DEFAULT_CALIBRATION_SECONDS,
        help="Seconds to measure ambient noise before listening.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=_DEFAULT_WINDOW_SECONDS,
        help="Rolling window length for avg/peak display.",
    )
    parser.add_argument(
        "--block-ms",
        type=int,
        default=_DEFAULT_BLOCK_MS,
        help="Block size for audio capture in milliseconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_DEFAULT_SAMPLE_RATE,
        help="Audio sample rate used for capture.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override silence threshold used for speech indicator.",
    )
    parser.add_argument(
        "--speech-hi-dbfs",
        type=float,
        default=None,
        help="dBFS level to switch to SPEAK (hysteresis upper bound).",
    )
    parser.add_argument(
        "--speech-lo-dbfs",
        type=float,
        default=None,
        help="dBFS level to switch to SILENT (hysteresis lower bound).",
    )
    parser.add_argument(
        "--speech-release-ms",
        type=int,
        default=_SPEECH_RELEASE_MS,
        help="Hold time in ms before dropping to SILENT.",
    )
    parser.add_argument(
        "--h-lag-db",
        type=float,
        default=_H_LAG_DEFAULT_DB,
        help="dB lag for the high-water marker to trail the peak.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    return parser.parse_args()


def _window_blocks(window_seconds: float, block_ms: int) -> int:
    if window_seconds <= 0:
        return 1
    block_ms = max(1, block_ms)
    return max(1, int(round((window_seconds * 1000.0) / block_ms)))


def main() -> None:
    args = _parse_args()
    console = Console(color_system=None if args.no_color else "auto")
    block_ms = max(1, int(args.block_ms))
    sample_rate = max(1, int(args.sample_rate))
    blocksize = max(1, int(sample_rate * (block_ms / 1000.0)))
    window_blocks = _window_blocks(float(args.window_seconds), block_ms)
    calibration_seconds = max(0.0, float(args.calibration_seconds))

    console.print("Listening... speak into the mic.")
    if sys.stdin.isatty():
        console.print("Press 'q' to quit.")
        console.print(
            f"Calibrating ambient noise for {calibration_seconds:.0f}s..."
        )
        meter = _LevelMeter(window_blocks=window_blocks)
        stats = _Stats()
        state = _State()
        calibration_end = time.monotonic() + calibration_seconds
        with _cbreak_mode():
            with sd.InputStream(
                channels=1,
                callback=lambda *cb_args: _callback(
                    console, state, stats, meter, *cb_args
                ),
                samplerate=sample_rate,
                blocksize=blocksize,
            ):
                with Live(
                    _render_line(state),
                    console=console,
                    refresh_per_second=20,
                    transient=True,
                ) as live:
                    while True:
                        if state.calibrating and time.monotonic() >= calibration_end:
                            state.calibrating = False
                            state.threshold = (
                                args.threshold
                                if args.threshold is not None
                                else _suggest_threshold(
                                    stats.summary()["noise_median"], None
                                )
                            )
                            hi, lo = _speech_thresholds_from_rms(state.threshold)
                            state.speech_hi_dbfs = (
                                args.speech_hi_dbfs
                                if args.speech_hi_dbfs is not None
                                else hi
                            )
                            state.speech_lo_dbfs = (
                                args.speech_lo_dbfs
                                if args.speech_lo_dbfs is not None
                                else lo
                            )
                            state.speech_release_ms = max(
                                0, int(args.speech_release_ms)
                            )
                            state.h_marker_lag_db = max(0.0, float(args.h_lag_db))
                            _update_bar_thresholds(
                                state,
                                stats.summary()["noise_median"],
                                None,
                                None,
                                None,
                            )
                            console.print("Calibration complete. Speak normally to test.")
                        if not state.calibrating:
                            summary = stats.summary()
                            _update_bar_thresholds(
                                state,
                                summary["noise_median"],
                                summary["speech_p90"],
                                summary["speech_p99"],
                                summary["speech_peak"],
                            )
                        live.update(_render_line(state))
                        if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                            key = sys.stdin.read(1)
                            if not key:
                                continue
                            key = key.lower()
                            if key == "q":
                                console.print()
                                break
                            if key == "a":
                                state.selected = "attack"
                            elif key == "d":
                                state.selected = "decay"
                            elif key == "l":
                                state.selected = "low"
                            elif key == "r":
                                state.selected = "release"
                            elif key in ("-", "_"):
                                if state.selected == "attack":
                                    state.attack_s = max(
                                        _ATTACK_MIN, state.attack_s - _ATTACK_STEP
                                    )
                                elif state.selected == "decay":
                                    state.decay_s = max(
                                        _DECAY_MIN, state.decay_s - _DECAY_STEP
                                    )
                                elif state.selected == "low":
                                    state.speech_lo_dbfs = max(
                                        _MIN_DBFS, state.speech_lo_dbfs - 1.0
                                    )
                                    if state.speech_hi_dbfs < state.speech_lo_dbfs:
                                        state.speech_hi_dbfs = state.speech_lo_dbfs
                                elif state.selected == "release":
                                    state.speech_release_ms = max(
                                        0, state.speech_release_ms - _RELEASE_STEP_MS
                                    )
                            elif key in ("=", "+"):
                                if state.selected == "attack":
                                    state.attack_s = min(
                                        _ATTACK_MAX, state.attack_s + _ATTACK_STEP
                                    )
                                elif state.selected == "decay":
                                    state.decay_s = min(
                                        _DECAY_MAX, state.decay_s + _DECAY_STEP
                                    )
                                elif state.selected == "low":
                                    state.speech_lo_dbfs = min(
                                        0.0, state.speech_lo_dbfs + 1.0
                                    )
                                elif state.selected == "release":
                                    state.speech_release_ms = min(
                                        5000, state.speech_release_ms + _RELEASE_STEP_MS
                                    )
                        time.sleep(0.05)
        _print_summary(console, stats, args.threshold, state)
        return

    console.print("stdin is not a TTY; use Ctrl+C to stop.")
    console.print(f"Calibrating ambient noise for {calibration_seconds:.0f}s...")
    meter = _LevelMeter(window_blocks=window_blocks)
    stats = _Stats()
    state = _State()
    calibration_end = time.monotonic() + calibration_seconds
    with sd.InputStream(
        channels=1,
        callback=lambda *cb_args: _callback(console, state, stats, meter, *cb_args),
        samplerate=sample_rate,
        blocksize=blocksize,
    ):
        while True:
            if state.calibrating and time.monotonic() >= calibration_end:
                state.calibrating = False
                state.threshold = (
                    args.threshold
                    if args.threshold is not None
                    else _suggest_threshold(stats.summary()["noise_median"], None)
                )
                hi, lo = _speech_thresholds_from_rms(state.threshold)
                state.speech_hi_dbfs = (
                    args.speech_hi_dbfs if args.speech_hi_dbfs is not None else hi
                )
                state.speech_lo_dbfs = (
                    args.speech_lo_dbfs if args.speech_lo_dbfs is not None else lo
                )
                state.speech_release_ms = max(0, int(args.speech_release_ms))
                state.h_marker_lag_db = max(0.0, float(args.h_lag_db))
                _update_bar_thresholds(
                    state,
                    stats.summary()["noise_median"],
                    None,
                    None,
                    None,
                )
                console.print("Calibration complete. Speak normally to test.")
            if not state.calibrating:
                summary = stats.summary()
                _update_bar_thresholds(
                    state,
                    summary["noise_median"],
                    summary["speech_p90"],
                    summary["speech_p99"],
                    summary["speech_peak"],
                )
            time.sleep(0.1)


def _suggest_threshold(noise: float | None, speech: float | None) -> float | None:
    if noise is None:
        return None
    if speech is None:
        return max(0.002, noise * 2.5)
    midpoint = (noise + speech) / 2.0
    return max(0.002, midpoint, noise * 2.0)


def _print_summary(
    console: Console,
    stats: _Stats,
    override_threshold: float | None,
    state: _State,
) -> None:
    summary = stats.summary()
    noise = summary["noise_median"]
    speech = summary["speech_p90"]
    max_clip = summary["max_clip"]
    peak_dbfs = summary["speech_peak_dbfs"]
    suggested = _suggest_threshold(noise, speech)
    threshold = override_threshold if override_threshold is not None else suggested
    console.print("Hotmic summary:", style="bold")
    if noise is not None:
        console.print(f"- Noise floor (median RMS): {noise:0.4f}")
    else:
        console.print("- Noise floor: n/a (no calibration samples)")
    if speech is not None:
        console.print(f"- Speech level (90th pct RMS): {speech:0.4f}")
    else:
        console.print("- Speech level: n/a (no speech samples)")
    console.print(f"- Max clip percent: {max_clip:0.1f}%")
    if peak_dbfs is not None:
        console.print(f"- Absolute peak: {peak_dbfs:0.1f} dBFS")
    if threshold is not None:
        console.print(
            f"- Suggested moshi.json stt_silence_threshold: {threshold:0.4f}"
        )
    else:
        console.print("- Suggested moshi.json stt_silence_threshold: n/a")
    if threshold is not None:
        console.print(
            f"- Paste into moshi.json: \"stt_silence_threshold\": {threshold:0.4f}"
        )
    console.print(f"- Speech low (L) marker: {state.speech_lo_dbfs:0.1f} dBFS")
    console.print(f"- Speech release delay: {state.speech_release_ms} ms")
    console.print(
        f"- Suggested moshi.json \"stt_silence_ms:\" {state.speech_release_ms}"
    )


if __name__ == "__main__":
    main()
