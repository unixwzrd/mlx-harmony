#!/usr/bin/env python3
"""
Profile mlx-harmony-chat (real-world usage profiling) + extra metrics.

Outputs:
  - cProfile stats:            stats/profile_chat.stats
  - text summary (top N):      stats/profile_chat.stats.txt
  - graphviz (optional):       stats/profile_chat.svg + .dot
  - derived runtime metrics:   stats/profile_chat.metrics.json
  - static metrics (AST):      stats/profile_chat.static.txt

Extra metrics include:
  - total calls, primitive calls, total time
  - approximate "calls per token step" using callcount of _decode_next_token
  - hotspot counters for suspected overhead (dict.get, str.join, re.escape, numpy.array, mlx array conversions)
  - static fan-in/fan-out and a simple cyclomatic-ish complexity estimate per function
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import json
import os
import pstats
import subprocess as sp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ----------------------------
# Utilities: filesystem & AST
# ----------------------------

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
    """
    Best-effort dotted name for calls:
      - foo()
      - mod.foo()
      - obj.method()
    """
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
    qualname: str  # e.g. "Class.method" or "func" (module-level)
    lineno: int


class StaticAnalyzer(ast.NodeVisitor):
    """
    Builds:
      - function/method table
      - call edges (caller -> callee dotted name)
      - cyclomatic-ish complexity per function
    """
    def __init__(self, file_path: Path) -> None:
        self.file_path = str(file_path)
        self.class_stack: List[str] = []
        self.func_stack: List[FuncKey] = []

        self.defined: Dict[FuncKey, Dict[str, object]] = {}  # metadata
        self.calls: Dict[FuncKey, List[str]] = {}            # dotted names
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
        self.complexity[fk] = 1  # start at 1 (McCabe-style baseline)
        return fk

    def _leave_function(self) -> None:
        self.func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fk = self._enter_function(node, node.name)
        self.generic_visit(node)
        self._leave_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        fk = self._enter_function(node, node.name)
        self.generic_visit(node)
        self._leave_function()

    # Complexity increments: a pragmatic subset of McCabe contributors
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
        # each except/finally increases branching
        self._bump_complexity(len(node.handlers) + (1 if node.finalbody else 0))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # a and b and c => +2
        self._bump_complexity(max(0, len(getattr(node, "values", [])) - 1))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # a < b < c => +1 (extra comparator)
        self._bump_complexity(max(0, len(getattr(node, "ops", [])) - 1))
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # each "for" in a comprehension adds a branch; each if adds more
        self._bump_complexity(1 + len(getattr(node, "ifs", [])))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.func_stack:
            dn = dotted_name(node.func)
            if dn:
                self.calls[self.func_stack[-1]].append(dn)
        self.generic_visit(node)


def build_static_reports(src_root: Path) -> Tuple[str, Dict[str, object]]:
    """
    Returns:
      - human readable report
      - structured dict with complexity + fan-in/out
    """
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

    # Flatten definitions
    funcs: List[FuncKey] = []
    complexity: Dict[FuncKey, int] = {}
    calls: Dict[FuncKey, List[str]] = {}

    for sa in analyzers:
        funcs.extend(sa.defined.keys())
        complexity.update(sa.complexity)
        calls.update(sa.calls)

    # Fan-out: unique callees per function (by dotted name)
    fan_out: Dict[FuncKey, int] = {fk: len(set(calls.get(fk, []))) for fk in funcs}

    # Fan-in: count how many functions reference a dotted name that matches a defined qualname
    qual_to_fk: Dict[str, List[FuncKey]] = {}
    for fk in funcs:
        qual_to_fk.setdefault(fk.qualname, []).append(fk)

    fan_in_count: Dict[FuncKey, int] = {fk: 0 for fk in funcs}
    # Build reverse edges by matching callee dotted names to known qualnames (best effort)
    for caller, callees in calls.items():
        for dn in set(callees):
            # exact match
            for target in qual_to_fk.get(dn, []):
                fan_in_count[target] += 1
            # also allow matching "Class.method" when call is "method" (very common)
            if "." not in dn:
                # if any qualname ends with ".<dn>", count it (approx)
                suffix = f".{dn}"
                for qual, targets in qual_to_fk.items():
                    if qual.endswith(suffix):
                        for target in targets:
                            fan_in_count[target] += 1

    # Prepare top lists
    top_complex = sorted(funcs, key=lambda fk: complexity.get(fk, 1), reverse=True)[:40]
    top_fanout = sorted(funcs, key=lambda fk: fan_out.get(fk, 0), reverse=True)[:40]
    top_fanin  = sorted(funcs, key=lambda fk: fan_in_count.get(fk, 0), reverse=True)[:40]

    def fmt_fk(fk: FuncKey) -> str:
        rel = fk.file
        return f"{rel}:{fk.lineno}  {fk.qualname}"

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
            {"file": fk.file, "lineno": fk.lineno, "qualname": fk.qualname, "complexity": complexity.get(fk, 1)}
            for fk in top_complex
        ],
        "top_fan_out": [
            {"file": fk.file, "lineno": fk.lineno, "qualname": fk.qualname, "fan_out": fan_out.get(fk, 0)}
            for fk in top_fanout
        ],
        "top_fan_in": [
            {"file": fk.file, "lineno": fk.lineno, "qualname": fk.qualname, "fan_in": fan_in_count.get(fk, 0)}
            for fk in top_fanin
        ],
    }
    return "\n".join(lines), structured


# ----------------------------
# Runtime metrics from pstats
# ----------------------------

def _normalize_stats_path(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute() and not str(pp).startswith("stats/"):
        pp = Path("stats") / pp
    pp.parent.mkdir(parents=True, exist_ok=True)
    return pp


def _stats_to_dict(stats: pstats.Stats) -> Dict[Tuple[str, int, str], Tuple[int, int, float, float, Dict]]:
    # pstats.Stats.stats maps (filename, lineno, funcname) -> (cc, nc, tt, ct, callers)
    return stats.stats  # type: ignore[return-value]


def derive_runtime_metrics(profile_stats_path: Path) -> Dict[str, object]:
    ps = pstats.Stats(str(profile_stats_path))
    st = _stats_to_dict(ps)

    # Total call counts and time (pstats stores these)
    total_calls = getattr(ps, "total_calls", None)
    prim_calls = getattr(ps, "prim_calls", None)
    total_tt = getattr(ps, "total_tt", None)

    # Find likely "token step" counter: _decode_next_token in your tree
    decode_key = None
    for k in st.keys():
        filename, lineno, funcname = k
        if funcname == "_decode_next_token":
            decode_key = k
            break

    tokens = None
    if decode_key:
        cc, nc, tt, ct, callers = st[decode_key]
        # nc is total calls to the function (often = tokens generated)
        tokens = int(nc)

    calls_per_token = None
    if tokens and total_calls:
        calls_per_token = float(total_calls) / float(tokens) if tokens > 0 else None

    # “Hot allocation/conversion suspects” by substring matching
    suspects = [
        ("dict.get", "method 'get' of 'dict' objects"),
        ("str.join", "method 'join' of 'str' objects"),
        ("str.translate", "method 'translate' of 'str' objects"),
        ("re.escape", "re.py:255(escape)"),
        ("numpy.array", "numpy.array"),
        ("mlx.array", "mlx"),
        ("mx.array", "mx.array"),
    ]

    # Build a searchable label per entry
    def label_for(k: Tuple[str, int, str]) -> str:
        filename, lineno, funcname = k
        return f"{filename}:{lineno}({funcname})"

    hot: Dict[str, Dict[str, object]] = {}
    for name, pattern in suspects:
        best = None
        best_ct = 0.0
        best_nc = 0
        for k, v in st.items():
            cc, nc, tt, ct, callers = v
            lab = label_for(k)
            if pattern in lab:
                if ct > best_ct:
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


# ----------------------------
# Main profiler runner
# ----------------------------

def profile_chat_command(
    model_path: str,
    chat_args: List[str],
    profile_output: str = "stats/profile_chat.stats",
    graph_output: str = "stats/profile_chat.svg",
    text_only: bool = False,
    static_metrics: bool = True,
    top_n: int = 50,
    node_thres: float | None = None,
    edge_thres: float | None = None,
) -> int:
    print("[PROFILE] Starting profiling of mlx-harmony-chat...")
    print(f"[PROFILE] Model: {model_path}")
    print(f"[PROFILE] Chat args: {chat_args}")
    print("[PROFILE] Type 'q' to quit and finish profiling.\n")

    # Ensure src in sys.path so we can import mlx_harmony
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from mlx_harmony.chat import main as chat_main

    original_argv = sys.argv
    sys.argv = ["mlx-harmony-chat", "--model", model_path] + chat_args

    prof = cProfile.Profile()
    try:
        prof.enable()
        chat_main()
        prof.disable()
    except (KeyboardInterrupt, EOFError, SystemExit):
        prof.disable()
        print("\n[PROFILE] Profiling stopped.")
    finally:
        sys.argv = original_argv

    profile_path = _normalize_stats_path(profile_output)
    prof.dump_stats(str(profile_path))
    print(f"[PROFILE] Stats saved to: {profile_path}")

    # Text report
    txt_path = Path(str(profile_path) + ".txt")
    with txt_path.open("w") as f:
        s = pstats.Stats(str(profile_path), stream=f)
        s.sort_stats("cumulative")
        s.print_stats(top_n)
    print(f"[PROFILE] Text report saved to: {txt_path}")

    # Derived runtime metrics (JSON)
    metrics = derive_runtime_metrics(profile_path)

    # Static metrics (AST) from src/
    static_structured: Dict[str, object] = {}
    static_txt = ""
    if static_metrics:
        static_txt, static_structured = build_static_reports(src_path)
        static_txt_path = profile_path.with_suffix(".static.txt")
        static_txt_path.write_text(static_txt, encoding="utf-8")
        print(f"[PROFILE] Static metrics saved to: {static_txt_path}")

    metrics["static"] = static_structured

    metrics_path = profile_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[PROFILE] Derived metrics saved to: {metrics_path}")

    # Graphviz
    if not text_only:
        try:
            graph_path = _normalize_stats_path(graph_output)
            dot_path = graph_path.with_suffix(".dot")

            gprof_cmd = ["gprof2dot", "-f", "pstats", str(profile_path), "-o", str(dot_path)]
            if node_thres is not None:
                gprof_cmd.extend(["--node-thres", str(node_thres)])
            if edge_thres is not None:
                gprof_cmd.extend(["--edge-thres", str(edge_thres)])
            sp.run(
                gprof_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            if graph_path.suffix.lower() == ".svg":
                sp.run(
                    ["dot", "-Tsvg", str(dot_path), "-o", str(graph_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            print(f"[PROFILE] Graphviz visualization saved to: {graph_path}")
        except (sp.CalledProcessError, FileNotFoundError):
            print("[WARNING] gprof2dot and/or graphviz 'dot' not found.")
            print("[INFO] Install: pip install gprof2dot && (brew install graphviz | apt install graphviz)")
        except Exception as e:
            print(f"[WARNING] Failed to generate graphviz visualization: {e}")

    print("[PROFILE] Done.")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Profile mlx-harmony-chat as it runs (cProfile + extra metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model path to use")
    parser.add_argument(
        "--profile-output",
        default="stats/profile_chat.stats",
        help="Output file for cProfile stats (default: stats/profile_chat.stats)",
    )
    parser.add_argument(
        "--graph",
        default="stats/profile_chat.svg",
        help="Output file for graphviz visualization (default: stats/profile_chat.svg)",
    )
    parser.add_argument("--text-only", action="store_true", help="Only generate text report, skip graphviz")
    parser.add_argument("--no-static", action="store_true", help="Skip AST static metrics (complexity + fan-in/out)")
    parser.add_argument("--top", type=int, default=50, help="Top N functions for the pstats text report (default: 50)")
    parser.add_argument(
        "--node-thres",
        type=float,
        default=None,
        help="gprof2dot node threshold (percentage, e.g. 0.1).",
    )
    parser.add_argument(
        "--edge-thres",
        type=float,
        default=None,
        help="gprof2dot edge threshold (percentage, e.g. 0.1).",
    )

    args, passthrough_args = parser.parse_known_args(argv[1:])
    passthrough_args = [a for a in passthrough_args if a != "--"]

    return profile_chat_command(
        model_path=args.model,
        chat_args=passthrough_args,
        profile_output=args.profile_output,
        graph_output=args.graph,
        text_only=args.text_only,
        static_metrics=not args.no_static,
        top_n=args.top,
        node_thres=args.node_thres,
        edge_thres=args.edge_thres,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
