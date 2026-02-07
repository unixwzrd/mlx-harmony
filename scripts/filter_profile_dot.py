#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

NODE_RE = re.compile(r'^\s*(\d+)\s+\[.*tooltip="([^"]+)"')
EDGE_RE = re.compile(r"^\s*(\d+)\s+->\s+(\d+)\b")


def filter_dot(*, input_path: Path, output_path: Path, keep_substring: str) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    keep_nodes: set[str] = set()

    for line in lines:
        match = NODE_RE.match(line)
        if not match:
            continue
        node_id, tooltip = match.group(1), match.group(2)
        if keep_substring in tooltip:
            keep_nodes.add(node_id)

    filtered: list[str] = []
    for line in lines:
        if line.strip() in {"digraph {", "}", ""}:
            filtered.append(line)
            continue
        if line.lstrip().startswith(("graph ", "node ", "edge ")):
            filtered.append(line)
            continue
        match = NODE_RE.match(line)
        if match and match.group(1) in keep_nodes:
            filtered.append(line)
            continue
        match = EDGE_RE.match(line)
        if match and match.group(1) in keep_nodes and match.group(2) in keep_nodes:
            filtered.append(line)
            continue

    header = [
        f"// source: {input_path}",
        f"// filter: keep tooltip contains '{keep_substring}'",
    ]
    output_path.write_text(
        "\n".join(header + filtered) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter Graphviz dot nodes by tooltip substring.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-substring", default="mlx_harmony")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    filter_dot(
        input_path=input_path,
        output_path=output_path,
        keep_substring=args.keep_substring,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
