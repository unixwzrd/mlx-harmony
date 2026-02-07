#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pstats
import subprocess as sp
import sys
from pathlib import Path


def _load_profile_reporting_module():
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "profile_reporting.py"
    spec = importlib.util.spec_from_file_location("profile_reporting", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load profile_reporting from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("profile_reporting", module)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate profile reports from pstats output.")
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--profile-text", required=True)
    parser.add_argument("--profile-metrics-json", required=True)
    parser.add_argument("--profile-static-txt", required=True)
    parser.add_argument("--profile-dot", default="")
    parser.add_argument("--profile-svg", default="")
    parser.add_argument("--node-thres", type=float, default=None)
    parser.add_argument("--edge-thres", type=float, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    stats_path = Path(args.profile_output)
    report_path = Path(args.profile_text)
    metrics_path = Path(args.profile_metrics_json)
    static_path = Path(args.profile_static_txt)
    dot_path = Path(args.profile_dot) if args.profile_dot else None
    svg_path = Path(args.profile_svg) if args.profile_svg else None

    try:
        stats = pstats.Stats(str(stats_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to read pstats: {stats_path} ({exc})", flush=True)
        return 0

    stats.sort_stats("cumulative")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as out:
        stats.stream = out
        stats.print_stats(args.top)

    metrics: dict[str, object] = {}
    static_txt = ""
    try:
        profile_reporting = _load_profile_reporting_module()
        metrics = profile_reporting.derive_runtime_metrics(stats_path)
        if not args.no_static:
            static_txt, static_structured = profile_reporting.build_static_reports(Path("src"))
            if isinstance(metrics, dict):
                metrics["static"] = static_structured
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to derive metrics/static from stats: {exc}", flush=True)
    if metrics_path and metrics:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if static_path and static_txt:
        static_path.write_text(static_txt, encoding="utf-8")

    if args.text_only:
        return 0

    graph_dot = dot_path
    graph_svg = svg_path
    if graph_svg and not graph_dot:
        graph_dot = graph_svg.with_suffix(".dot")

    if graph_dot:
        try:
            gprof_cmd = ["gprof2dot", "-f", "pstats", str(stats_path), "-o", str(graph_dot)]
            if args.node_thres is not None:
                gprof_cmd.extend(["--node-thres", str(args.node_thres)])
            if args.edge_thres is not None:
                gprof_cmd.extend(["--edge-thres", str(args.edge_thres)])
            sp.run(gprof_cmd, capture_output=True, text=True, check=True)
        except (sp.CalledProcessError, FileNotFoundError):
            print("[WARNING] gprof2dot not available for graph generation.", flush=True)
            return 0
        dot_filter = os.environ.get("DOT_FILTER", "1")
        dot_filter_substring = os.environ.get("DOT_FILTER_SUBSTRING", "mlx_harmony")
        if dot_filter == "1" and dot_filter_substring:
            filtered_dot = graph_dot.with_suffix(".filtered.dot")
            filter_cmd = [
                sys.executable,
                "scripts/filter_profile_dot.py",
                "--input",
                str(graph_dot),
                "--output",
                str(filtered_dot),
                "--keep-substring",
                dot_filter_substring,
            ]
            sp.run(filter_cmd, capture_output=True, text=True, check=False)
            if filtered_dot.exists() and filtered_dot.stat().st_size > 0:
                keep_full = os.environ.get("DOT_FILTER_KEEP_FULL", "0")
                if keep_full == "1":
                    full_dot = graph_dot.with_suffix(".full.dot")
                    graph_dot.rename(full_dot)
                filtered_dot.replace(graph_dot)

    if graph_svg and graph_dot:
        try:
            sp.run(["dot", "-Tsvg", str(graph_dot), "-o", str(graph_svg)], capture_output=True, text=True, check=True)
        except (sp.CalledProcessError, FileNotFoundError):
            print("[WARNING] graphviz 'dot' not available for graph generation.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
