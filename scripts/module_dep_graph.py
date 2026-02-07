#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.name == "__pycache__":
            continue
        yield path


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module = ".".join(["mlx_harmony", *parts]) if parts else "mlx_harmony"
    return module


def _resolve_relative(module: str, level: int, name: str | None) -> str | None:
    parts = module.split(".")
    if level > len(parts):
        return None
    base = parts[:-level]
    if name:
        base.append(name)
    return ".".join(base)


def _collect_edges(root: Path) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for path in _iter_python_files(root):
        module = _module_name(root, path)
        edges.setdefault(module, set())
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("mlx_harmony"):
                        edges[module].add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.level and module:
                    resolved = _resolve_relative(module, node.level, node.module)
                else:
                    resolved = node.module
                if resolved and resolved.startswith("mlx_harmony"):
                    edges[module].add(resolved)
    return edges


def _normalize_entry(root: Path, entry: str) -> str:
    path = Path(entry)
    if path.exists():
        return _module_name(root, path)
    if entry.endswith(".py"):
        return _module_name(root, root / entry)
    if entry.startswith("mlx_harmony"):
        return entry
    return f"mlx_harmony.{entry}"


def _reachable(edges: dict[str, set[str]], entries: list[str]) -> dict[str, set[str]]:
    if not entries:
        return edges
    seen: set[str] = set()
    stack = list(entries)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in edges.get(current, set()):
            if dep not in seen:
                stack.append(dep)
    return {node: deps for node, deps in edges.items() if node in seen}


def _emit_dot(edges: dict[str, set[str]]) -> str:
    lines = ["digraph mlx_harmony {", "  rankdir=LR;"]
    for node in sorted(edges):
        lines.append(f'  "{node}";')
    for node in sorted(edges):
        for dep in sorted(edges[node]):
            lines.append(f'  "{node}" -> "{dep}";')
    lines.append("}")
    return "\n".join(lines)


def _emit_tsv(edges: dict[str, set[str]]) -> str:
    lines = ["source\ttarget"]
    for node in sorted(edges):
        for dep in sorted(edges[node]):
            lines.append(f"{node}\t{dep}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a module dependency graph for mlx_harmony.")
    parser.add_argument(
        "--root",
        default="src/mlx_harmony",
        help="Root directory to scan (default: src/mlx_harmony).",
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Entry module or file path to filter reachable nodes (repeatable).",
    )
    parser.add_argument(
        "--format",
        choices=["dot", "tsv"],
        default="dot",
        help="Output format (dot or tsv).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path (default: stdout).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        return 2
    edges = _collect_edges(root)
    entries = [_normalize_entry(root, entry) for entry in args.entry]
    filtered = _reachable(edges, entries)
    output = _emit_dot(filtered) if args.format == "dot" else _emit_tsv(filtered)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
