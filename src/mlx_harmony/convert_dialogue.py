#!/usr/bin/env python3
"""
Utility to convert dialogue text format to JSON format for example_dialogues.

Usage:
    python -m mlx_harmony.convert_dialogue input.txt output.json
    python -m mlx_harmony.convert_dialogue input.txt --stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import parse_dialogue_file, parse_dialogue_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dialogue text format to JSON format for example_dialogues."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input dialogue text file (or '-' for stdin)",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output JSON file (or --stdout to write to stdout)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write output to stdout instead of a file",
    )
    parser.add_argument(
        "--as-example-dialogues",
        action="store_true",
        help="Output as example_dialogues format (array of arrays) instead of flat messages array",
    )
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        text = sys.stdin.read()
        messages = parse_dialogue_text(text)
    else:
        messages = parse_dialogue_file(args.input)

    # Convert to output format
    if args.as_example_dialogues:
        # Group messages into example dialogues (each example is a list of turns)
        # For now, treat all messages as one example dialogue
        output = [messages]
    else:
        output = messages

    # Write output
    if args.stdout or args.output is None:
        json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Converted {len(messages)} messages to {output_path}")


if __name__ == "__main__":
    main()
