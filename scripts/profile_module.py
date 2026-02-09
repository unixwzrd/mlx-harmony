#!/usr/bin/env python3
"""Generic cProfile wrapper for Python modules or scripts."""

from __future__ import annotations

import argparse
import cProfile
import runpy
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Profile a Python module or script.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--module", help="Module to execute (as with python -m).")
    group.add_argument("--script", help="Script path to execute.")
    parser.add_argument(
        "--profile-output",
        required=True,
        help="Output path for cProfile stats (pstats format).",
    )
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main() -> int:
    args, passthrough = _parse_args(sys.argv[1:])
    output_path = Path(args.profile_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_argv = sys.argv[:]
    profiler = cProfile.Profile()
    try:
        if args.module:
            sys.argv = [args.module, *passthrough]
            profiler.enable()
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
            profiler.disable()
        else:
            script_path = Path(args.script).resolve()
            if not script_path.exists():
                print(f"[ERROR] Script not found: {script_path}", file=sys.stderr)
                return 2
            sys.argv = [str(script_path), *passthrough]
            profiler.enable()
            runpy.run_path(str(script_path), run_name="__main__")
            profiler.disable()
    except SystemExit as exc:
        profiler.disable()
        code = exc.code if isinstance(exc.code, int) else 0
    finally:
        sys.argv = original_argv

    profiler.dump_stats(str(output_path))
    return code if "code" in locals() else 0


if __name__ == "__main__":
    raise SystemExit(main())
