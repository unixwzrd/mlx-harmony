#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Iterable


def iter_instructions(path: Path) -> Iterable[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of objects in JSON file")
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction")
        if instruction:
            yield str(instruction)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: build_prompt_stream.py <english.json> [limit]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] JSON file not found: {path}", file=sys.stderr)
        return 1
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    count = 0
    try:
        for instruction in iter_instructions(path):
            print("\\")
            print(instruction)
            print("\\")
            count += 1
            if limit is not None and count >= limit:
                break
        print("q")
    except BrokenPipeError:
        # Downstream consumer closed stdin; treat as normal termination.
        return 0
    return 0


if __name__ == "__main__":
    # Exit cleanly when the downstream consumer closes the pipe.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Avoid interpreter finalization attempting to flush a broken stdout pipe.
        os._exit(0)
