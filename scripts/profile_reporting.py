#!/usr/bin/env python3
"""
Shared profile reporting helpers.

Provides:
  - derive_runtime_metrics: compute runtime metrics from pstats output
  - build_static_reports: AST-derived static metrics (complexity, fan-in/out)
"""

from __future__ import annotations

import ast
import os
import pstats
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

EXCLUDE_DIRS_DEFAULT = {
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def iter_py_files(root: Path, *, exclude_dirs: Set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py") and not fn.startswith("."):
                yield Path(dirpath) / fn


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


@dataclass(frozen=True)
class FuncKey:
    file: str
    qualname: str
    lineno: int


class StaticAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.file_path = str(file_path)
        self.class_stack: List[str] = []
        self.func_stack: List[FuncKey] = []
        self.defined: Dict[FuncKey, Dict[str, object]] = {}
        self.calls: Dict[FuncKey, List[str]] = {}
        self.complexity: Dict[FuncKey, int] = {}

    def current_qual_prefix(self) -> str:
        return ".".join(self.class_stack) if self.class_stack else ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _enter_function(self, node: ast.AST, name: str) -> FuncKey:
        prefix = self.current_qual_prefix()
        qual = f"{prefix}.{name}" if prefix else name
        fk = FuncKey(self.file_path, qual, int(getattr(node, "lineno", 1)))

        self.func_stack.append(fk)
        self.defined[fk] = {
            "kind": "method" if prefix else "function",
            "name": name,
            "qualname": qual,
            "lineno": fk.lineno,
            "end_lineno": int(getattr(node, "end_lineno", fk.lineno)),
        }
        self.calls.setdefault(fk, [])
        self.complexity[fk] = 1
        return fk

    def _leave_function(self) -> None:
        self.func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node, node.name)
        self.generic_visit(node)
        self._leave_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_function(node, node.name)
        self.generic_visit(node)
        self._leave_function()

    def _bump_complexity(self, n: int = 1) -> None:
        if not self.func_stack:
            return
        fk = self.func_stack[-1]
        self.complexity[fk] = self.complexity.get(fk, 1) + n

    def visit_If(self, node: ast.If) -> None:
        self._bump_complexity(1)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._bump_complexity(1)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._bump_complexity(1)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._bump_complexity(1)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._bump_complexity(len(node.handlers) + (1 if node.finalbody else 0))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._bump_complexity(max(0, len(getattr(node, "values", [])) - 1))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._bump_complexity(max(0, len(getattr(node, "ops", [])) - 1))
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._bump_complexity(1 + len(getattr(node, "ifs", [])))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.func_stack:
            dn = dotted_name(node.func)
            if dn:
                self.calls[self.func_stack[-1]].append(dn)
        self.generic_visit(node)


def build_static_reports(src_root: Path) -> Tuple[str, Dict[str, object]]:
    exclude_dirs = set(EXCLUDE_DIRS_DEFAULT)
    files = sorted(iter_py_files(src_root, exclude_dirs=exclude_dirs))

    analyzers: List[StaticAnalyzer] = []
    for p in files:
        try:
            tree = ast.parse(safe_read_text(p), filename=str(p))
        except SyntaxError:
            continue
        sa = StaticAnalyzer(p)
        sa.visit(tree)
        analyzers.append(sa)

    funcs: List[FuncKey] = []
    complexity: Dict[FuncKey, int] = {}
    calls: Dict[FuncKey, List[str]] = {}

    for sa in analyzers:
        funcs.extend(sa.defined.keys())
        complexity.update(sa.complexity)
        calls.update(sa.calls)

    fan_out: Dict[FuncKey, int] = {fk: len(set(calls.get(fk, []))) for fk in funcs}

    qual_to_fk: Dict[str, List[FuncKey]] = {}
    for fk in funcs:
        qual_to_fk.setdefault(fk.qualname, []).append(fk)

    fan_in_count: Dict[FuncKey, int] = {fk: 0 for fk in funcs}
    for caller, callees in calls.items():
        for dn in set(callees):
            for target in qual_to_fk.get(dn, []):
                fan_in_count[target] += 1
            if "." not in dn:
                suffix = f".{dn}"
                for qual, targets in qual_to_fk.items():
                    if qual.endswith(suffix):
                        for target in targets:
                            fan_in_count[target] += 1

    top_complex = sorted(funcs, key=lambda fk: complexity.get(fk, 1), reverse=True)[:40]
    top_fanout = sorted(funcs, key=lambda fk: fan_out.get(fk, 0), reverse=True)[:40]
    top_fanin = sorted(funcs, key=lambda fk: fan_in_count.get(fk, 0), reverse=True)[:40]

    def fmt_fk(fk: FuncKey) -> str:
        return f"{fk.file}:{fk.lineno}  {fk.qualname}"

    lines: List[str] = []
    lines.append("STATIC METRICS (AST-derived)")
    lines.append("")
    lines.append("Top cyclomatic-ish complexity (higher = more branching):")
    for fk in top_complex:
        lines.append(f"  {complexity.get(fk, 1):4d}  {fmt_fk(fk)}")
    lines.append("")
    lines.append("Top fan-out (unique callees; candidates for splitting/decoupling):")
    for fk in top_fanout:
        lines.append(f"  {fan_out.get(fk, 0):4d}  {fmt_fk(fk)}")
    lines.append("")
    lines.append("Top fan-in (many callers; ‘central’ APIs worth stabilizing):")
    for fk in top_fanin:
        lines.append(f"  {fan_in_count.get(fk, 0):4d}  {fmt_fk(fk)}")
    lines.append("")

    structured = {
        "top_complexity": [
            {
                "file": fk.file,
                "lineno": fk.lineno,
                "qualname": fk.qualname,
                "complexity": complexity.get(fk, 1),
            }
            for fk in top_complex
        ],
        "top_fan_out": [
            {
                "file": fk.file,
                "lineno": fk.lineno,
                "qualname": fk.qualname,
                "fan_out": fan_out.get(fk, 0),
            }
            for fk in top_fanout
        ],
        "top_fan_in": [
            {
                "file": fk.file,
                "lineno": fk.lineno,
                "qualname": fk.qualname,
                "fan_in": fan_in_count.get(fk, 0),
            }
            for fk in top_fanin
        ],
    }
    return "\n".join(lines), structured


def _stats_to_dict(
    stats: pstats.Stats,
) -> Dict[Tuple[str, int, str], Tuple[int, int, float, float, Dict]]:
    return stats.stats  # type: ignore[return-value]


def derive_runtime_metrics(profile_stats_path: Path) -> Dict[str, object]:
    ps = pstats.Stats(str(profile_stats_path))
    st = _stats_to_dict(ps)

    total_calls = getattr(ps, "total_calls", None)
    prim_calls = getattr(ps, "prim_calls", None)
    total_tt = getattr(ps, "total_tt", None)

    decode_key = None
    for k in st.keys():
        _, _, funcname = k
        if funcname == "_decode_next_token":
            decode_key = k
            break

    tokens = None
    if decode_key:
        _, nc, _, ct, _ = st[decode_key]
        tokens = int(nc)

    calls_per_token = None
    if tokens and total_calls:
        calls_per_token = float(total_calls) / float(tokens) if tokens > 0 else None

    suspects = [
        ("dict.get", "method 'get' of 'dict' objects"),
        ("str.join", "method 'join' of 'str' objects"),
        ("str.translate", "method 'translate' of 'str' objects"),
        ("re.escape", "re.py:255(escape)"),
        ("numpy.array", "numpy.array"),
        ("mlx.array", "mlx"),
        ("mx.array", "mx.array"),
    ]

    def label_for(k: Tuple[str, int, str]) -> str:
        filename, lineno, funcname = k
        return f"{filename}:{lineno}({funcname})"

    hot: Dict[str, Dict[str, object]] = {}
    for name, pattern in suspects:
        best = None
        best_ct = 0.0
        best_nc = 0
        for k, v in st.items():
            _, nc, _, ct, _ = v
            lab = label_for(k)
            if pattern in lab and ct > best_ct:
                best_ct = ct
                best_nc = int(nc)
                best = lab
        if best:
            hot[name] = {"where": best, "cumtime": best_ct, "ncalls": best_nc}

    return {
        "profile_stats": str(profile_stats_path),
        "total_calls": total_calls,
        "primitive_calls": prim_calls,
        "total_time_seconds": total_tt,
        "token_steps_estimate_from__decode_next_token": tokens,
        "calls_per_token_step_estimate": calls_per_token,
        "hot_suspects": hot,
    }
