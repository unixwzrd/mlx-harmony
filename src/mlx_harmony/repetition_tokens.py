from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class TokenRepetitionConfig:
    mode: str = "cheap"  # "off" | "cheap" | "full"
    min_tokens: int = 96
    single_token_run: int = 24
    span_sizes: tuple[int, ...] = (8, 16, 32, 64)
    low_var_window: int = 128
    low_var_unique_max: int = 8
    ngram: int = 16
    ngram_threshold: int = 3
    ngram_window: int = 256
    check_every: int = 8


class TokenRepetitionDetector:
    def __init__(self, cfg: TokenRepetitionConfig = TokenRepetitionConfig()) -> None:
        self.cfg = cfg
        max_span = max(cfg.span_sizes, default=0) * 2
        max_window = max(cfg.low_var_window, cfg.ngram_window * cfg.ngram, max_span)
        self.tokens: Deque[int] = deque(maxlen=max_window)
        self._ngram_q: Deque[int] = deque(maxlen=cfg.ngram)
        self._ngram_hist: Deque[tuple[int, ...]] = deque()
        self._ngram_counts: Counter[tuple[int, ...]] = Counter()
        self._i = 0

    def update(self, tok: int) -> Optional[str]:
        cfg = self.cfg
        self._i += 1
        t = int(tok)
        self.tokens.append(t)
        self._ngram_q.append(t)

        if cfg.mode == "off":
            return None
        if self._i % max(1, cfg.check_every) != 0:
            return None
        if len(self.tokens) < cfg.min_tokens:
            return None

        tlist = list(self.tokens)

        if len(tlist) >= cfg.single_token_run:
            recent = tlist[-cfg.single_token_run:]
            if all(x == recent[0] for x in recent):
                return "loop_detected:single_token_run"

        if cfg.mode != "full":
            return None

        for size in cfg.span_sizes:
            if len(tlist) >= 2 * size and tlist[-size:] == tlist[-2 * size:-size]:
                return f"loop_detected:repeat_span:{size}"

        if len(self._ngram_q) == cfg.ngram:
            gram = tuple(self._ngram_q)
            self._ngram_hist.append(gram)
            self._ngram_counts[gram] += 1
            if self._ngram_counts[gram] >= cfg.ngram_threshold:
                return "loop_detected:ngram_repeat"
            if len(self._ngram_hist) > cfg.ngram_window:
                old = self._ngram_hist.popleft()
                self._ngram_counts[old] -= 1
                if self._ngram_counts[old] <= 0:
                    del self._ngram_counts[old]

        if cfg.low_var_window and len(tlist) >= cfg.low_var_window:
            tail = tlist[-cfg.low_var_window:]
            if len(set(tail)) <= cfg.low_var_unique_max:
                return "loop_detected:low_variance"

        return None
