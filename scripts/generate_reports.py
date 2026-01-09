#!/usr/bin/env python3
"""
Generate text and SVG reports from an existing profile.stats file.

Usage:
    python scripts/generate_reports.py profile.stats
"""

import argparse
import pstats
import subprocess
import sys
from pathlib import Path


def generate_text_report(stats_file: str, output_file: str = None):
    """Generate a text report from profile stats."""
    if output_file is None:
        output_file = stats_file + ".txt"

    stats = pstats.Stats(stats_file)
    stats.sort_stats("cumulative")

    with open(output_file, "w") as f:
        # Redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = f
        stats.print_stats(50)
        sys.stdout = old_stdout

    print(f"[REPORT] Text report saved to: {output_file}")

    # Also print top 20 to console
    print("\n" + "=" * 80)
    print("PROFILING REPORT (Top 20 functions by cumulative time)")
    print("=" * 80)
    stats.print_stats(20)


def generate_graphviz_report(stats_file: str, output_file: str = None):
    """Generate a Graphviz SVG from profile stats."""
    if output_file is None:
        output_file = stats_file.replace(".stats", ".svg")

    out_path = Path(output_file)

    try:
        # Step 1: Generate DOT file from stats using gprof2dot
        dot_output = str(out_path.with_suffix(".dot"))
        subprocess.run(
            ["gprof2dot", "-f", "pstats", stats_file, "-o", dot_output],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[REPORT] DOT file generated: {dot_output}")

        # Step 2: Convert DOT to SVG using graphviz dot
        if out_path.suffix.lower() == ".svg":
            subprocess.run(
                ["dot", "-Tsvg", dot_output, "-o", str(out_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"[REPORT] SVG visualization saved to: {output_file}")
            print(f"[REPORT] View with: open {output_file} (macOS) or xdg-open {output_file} (Linux)")
        else:
            print(f"[REPORT] Graph output saved to: {dot_output}")

    except FileNotFoundError as e:
        print(f"[ERROR] Required tool not found: {e}")
        print("[INFO] Install: pip install gprof2dot && brew install graphviz")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to generate graphviz visualization: {e}")
        print(f"[INFO] Stats file: {stats_file}")
        print(f"[INFO] You can view with: python -m pstats {stats_file}")
        print(f"[INFO] Or install snakeviz: pip install snakeviz && snakeviz {stats_file}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from existing profile.stats file"
    )
    parser.add_argument(
        "stats_file",
        help="Path to profile.stats file",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Output file for text report (default: <stats_file>.txt)",
    )
    parser.add_argument(
        "--graph-output",
        default=None,
        help="Output file for SVG visualization (default: <stats_file>.svg)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only generate text report, skip graphviz",
    )

    args = parser.parse_args()

    stats_path = Path(args.stats_file)
    if not stats_path.exists():
        print(f"[ERROR] Stats file not found: {args.stats_file}")
        sys.exit(1)

    print(f"[REPORT] Generating reports from: {args.stats_file}")

    # Generate text report
    generate_text_report(args.stats_file, args.text_output)

    # Generate graphviz visualization
    if not args.text_only:
        generate_graphviz_report(args.stats_file, args.graph_output)

    print("\n[REPORT] Done!")


if __name__ == "__main__":
    main()
