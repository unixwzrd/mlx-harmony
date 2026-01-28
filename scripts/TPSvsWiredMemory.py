#!/usr/bin/env python3
import argparse
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

GB: int = 1024 ** 3


def _parse_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _filter_existing(columns: Iterable[str], available: set[str]) -> list[str]:
    return [col for col in columns if col in available]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="Path to merged-timings-vm_stat.tsv")
    ap.add_argument("--out", default="tokens_per_second_vs_wired_memory.png")
    ap.add_argument("--title", default="Tokens/sec vs Wired Memory")
    ap.add_argument("--x-col", default="datetime")
    ap.add_argument(
        "--left-cols",
        default="tokens_per_second",
        help="Comma-separated columns for left axis (default: tokens_per_second)",
    )
    ap.add_argument(
        "--right-cols",
        default="wired_bytes",
        help="Comma-separated columns for right axis (default: wired_bytes)",
    )
    args = ap.parse_args()

    # Read TSV
    df = pd.read_csv(args.tsv, sep="\t")

    available_cols = set(df.columns)
    left_cols = _filter_existing(_parse_columns(args.left_cols), available_cols)
    right_cols = _filter_existing(_parse_columns(args.right_cols), available_cols)
    if not left_cols:
        raise SystemExit("No valid columns found for --left-cols.")
    if not right_cols:
        raise SystemExit("No valid columns found for --right-cols.")

    x_col = args.x_col
    if x_col not in df.columns:
        raise SystemExit(f"Missing x column: {x_col}")

    if x_col == "datetime":
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col]).sort_values(x_col)
    else:
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col]).sort_values(x_col)

    for col in left_cols + right_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "wired_bytes" in right_cols:
        df["wired_gb"] = df["wired_bytes"] / GB
        right_cols = [col if col != "wired_bytes" else "wired_gb" for col in right_cols]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for col in left_cols:
        series = df.dropna(subset=[col])
        ax1.plot(series[x_col], series[col], linewidth=1.5, label=col)
    ax1.set_ylabel(", ".join(left_cols))
    ax1.set_xlabel(x_col)

    ax2 = ax1.twinx()
    for col in right_cols:
        series = df.dropna(subset=[col])
        ax2.plot(series[x_col], series[col], linewidth=1.5, label=col)
    ax2.set_ylabel(", ".join(right_cols))

    # Optional: nicer tick labels for wired axis (2 decimals)
    if "wired_gb" in right_cols:
        ax2.yaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")

    # Title + grid
    ax1.set_title(args.title)
    ax1.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5)

    # Combined legend (matplotlib keeps them separate per-axis)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
