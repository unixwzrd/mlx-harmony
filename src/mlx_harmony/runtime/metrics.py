from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, float]:
        return dict(self.totals)


@contextmanager
def timer(stats: TimingStats, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - start)
