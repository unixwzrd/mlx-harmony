#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pstats
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate profile reports from pstats output.")
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--profile-text", required=True)
    parser.add_argument("--profile-metrics-json", required=True)
    parser.add_argument("--profile-static-txt", required=True)
    args = parser.parse_args()

    stats_path = Path(args.profile_output)
    report_path = Path(args.profile_text)
    metrics_path = Path(args.profile_metrics_json)
    static_path = Path(args.profile_static_txt)

    try:
        stats = pstats.Stats(str(stats_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to read pstats: {stats_path} ({exc})", flush=True)
        return 0

    stats.sort_stats("cumulative")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as out:
        stats.stream = out
        stats.print_stats(50)

    metrics: dict[str, object] = {}
    static_txt = ""
    try:
        import scripts.profile_chat as profile_chat  # type: ignore

        metrics = profile_chat.derive_runtime_metrics(stats_path)
        static_txt, static_structured = profile_chat.build_static_reports(Path("src"))
        if isinstance(metrics, dict):
            metrics["static"] = static_structured
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to derive metrics/static from stats: {exc}", flush=True)
    if metrics_path and metrics:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if static_path and static_txt:
        static_path.write_text(static_txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
