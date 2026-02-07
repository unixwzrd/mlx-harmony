#!/usr/bin/env python3
"""
Unified profiling entrypoint for CLI, client, server, and startup.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional


def _normalize_stats_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Profile mlx-harmony-chat (CLI).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-output", default="stats/profile_chat.stats")
    parser.add_argument("--graph", default="stats/profile_chat.svg")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--node-thres", type=float, default=None)
    parser.add_argument("--edge-thres", type=float, default=None)
    args, passthrough_args = parser.parse_known_args(argv)
    passthrough_args = [a for a in passthrough_args if a != "--"]

    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import subprocess as sp

    from mlx_harmony.chat import main as chat_main

    original_argv = sys.argv
    sys.argv = ["mlx-harmony-chat", "--model", args.model] + passthrough_args
    prof = cProfile.Profile()
    try:
        prof.enable()
        chat_main()
        prof.disable()
    except (KeyboardInterrupt, EOFError, SystemExit):
        prof.disable()
    finally:
        sys.argv = original_argv

    profile_path = _normalize_stats_path(args.profile_output)
    prof.dump_stats(str(profile_path))

    text_path = profile_path.with_suffix(".stats.txt")
    metrics_path = profile_path.with_suffix(".metrics.json")
    static_path = profile_path.with_suffix(".static.txt")
    graph_path = _normalize_stats_path(args.graph)
    dot_path = graph_path.with_suffix(".dot")

    report_args = [
        sys.executable,
        "scripts/process_profile_artifacts.py",
        "--profile-output",
        str(profile_path),
        "--profile-text",
        str(text_path),
        "--profile-metrics-json",
        str(metrics_path),
        "--profile-static-txt",
        str(static_path),
        "--top",
        str(args.top)
    ]
    if args.no_static:
        report_args.append("--no-static")
    if args.text_only:
        report_args.append("--text-only")
    else:
        report_args.extend(["--profile-dot", str(dot_path), "--profile-svg", str(graph_path)])
        if args.node_thres is not None:
            report_args.extend(["--node-thres", str(args.node_thres)])
        if args.edge_thres is not None:
            report_args.extend(["--edge-thres", str(args.edge_thres)])
    sp.run(report_args, check=False)
    return 0


@dataclass
class _ClientConfig:
    host: str
    port: int
    model: Optional[str]
    profile: Optional[str]
    prompt_config: Optional[str]
    max_tokens: int | None
    timeout: int
    health_retries: int
    health_sleep: float
    report_file: Optional[Path]
    requests_log: Optional[Path]
    profile_output: Optional[Path]
    graph_output: Optional[Path]
    node_thres: Optional[float]
    edge_thres: Optional[float]
    text_only: bool


def _iter_prompt_blocks() -> Iterator[str]:
    collecting = False
    buffer: list[str] = []
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if line.strip() == "q":
            break
        if line.strip() == "\\":
            if collecting:
                yield "\n".join(buffer)
                buffer.clear()
                collecting = False
            else:
                collecting = True
            continue
        if collecting:
            buffer.append(line)


def _build_frontend_context(cfg: _ClientConfig) -> SimpleNamespace:
    from mlx_harmony.chat_utils import get_assistant_name, get_truncate_limits
    from mlx_harmony.config import load_prompt_config

    prompt_config = load_prompt_config(cfg.prompt_config) if cfg.prompt_config else None
    assistant_name = get_assistant_name(prompt_config)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
    return SimpleNamespace(
        generator=None,
        tools=[],
        prompt_config=prompt_config,
        profile_data=None,
        chats_dir="logs",
        logs_dir="logs",
        chat_file_path=None,
        chat_input_path=None,
        debug_path="logs/server-debug.log",
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=True,
    )


def _build_frontend_args(cfg: _ClientConfig) -> SimpleNamespace:
    return SimpleNamespace(
        debug=False,
        debug_file=None,
        debug_tokens=None,
        no_markdown=False,
        performance_mode=None,
        perf_max_tokens=None,
        perf_max_context_tokens=None,
        perf_max_kv_size=None,
        browser=False,
        use_python=False,
        apply_patch=False,
        mlock=None,
        lazy=None,
        seed=None,
        reseed_each_turn=None,
        max_context_tokens=None,
        max_tokens=cfg.max_tokens,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=None,
        repetition_penalty=None,
        repetition_context_size=None,
        loop_detection=None,
        profile=None,
        profiles_file="configs/profiles.example.json",
        prompt_config=cfg.prompt_config,
        model=cfg.model,
        chat=None,
    )


def _run_client(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Send prompt stream to the MLX Harmony API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--prompt-config", default="")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--health-retries", type=int, default=50)
    parser.add_argument("--health-sleep", type=float, default=0.2)
    parser.add_argument("--report-file", default="")
    parser.add_argument("--requests-log", default="")
    parser.add_argument("--profile-output", default="")
    parser.add_argument("--graph", dest="graph_output", default="")
    parser.add_argument("--node-thres", type=float, default=None)
    parser.add_argument("--edge-thres", type=float, default=None)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args(argv)

    cfg = _ClientConfig(
        host=args.host,
        port=args.port,
        model=args.model or None,
        profile=args.profile or None,
        prompt_config=args.prompt_config or None,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        health_retries=args.health_retries,
        health_sleep=args.health_sleep,
        report_file=Path(args.report_file) if args.report_file else None,
        requests_log=Path(args.requests_log) if args.requests_log else None,
        profile_output=Path(args.profile_output) if args.profile_output else None,
        graph_output=Path(args.graph_output) if args.graph_output else None,
        node_thres=args.node_thres,
        edge_thres=args.edge_thres,
        text_only=args.text_only,
    )

    import subprocess as sp

    from mlx_harmony.api_client import ApiClient, ApiClientConfig
    from mlx_harmony.chat_backend import ServerBackend
    from mlx_harmony.chat_frontend import run_cli_frontend, run_prompt_frontend
    from mlx_harmony.generation.client import GenerationClient, ServerGenerationClient

    profiler = None
    if cfg.profile_output is not None:
        cfg.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()

    results = {"status": "ok", "tests": []}
    health_client = ApiClient(
        ApiClientConfig(
            host=cfg.host,
            port=cfg.port,
            model=cfg.model,
            profile=cfg.profile,
            prompt_config=cfg.prompt_config,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            return_analysis=True,
            requests_log=cfg.requests_log,
        )
    )
    if not health_client.health_check(cfg.health_retries, cfg.health_sleep):
        msg = f"Server health check failed: http://{cfg.host}:{cfg.port}/v1/health"
        results["status"] = "fail"
        results["tests"].append({"name": "health_check", "status": "fail", "details": msg})
        if cfg.report_file:
            cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
            cfg.report_file.write_text(json.dumps(results), encoding="utf-8")
        print(msg, file=sys.stderr)
        return 1

    generation_client: GenerationClient = ServerGenerationClient(
        host=cfg.host,
        port=cfg.port,
        model=cfg.model,
        profile=cfg.profile,
        prompt_config=cfg.prompt_config,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        requests_log=None if cfg.requests_log is None else str(cfg.requests_log),
        return_analysis=True,
    )
    backend = ServerBackend(generation_client)
    frontend_context = _build_frontend_context(cfg)
    frontend_args = _build_frontend_args(cfg)

    try:
        if sys.stdin.isatty():
            run_cli_frontend(
                args=frontend_args,
                context=frontend_context,
                conversation=[],
                model_path=cfg.model or cfg.profile or "server",
                prompt_config_path=cfg.prompt_config,
                loaded_hyperparameters={},
                loaded_max_context_tokens=None,
                loaded_model_path=None,
                loaded_chat_id=None,
                backend=backend,
            )
        else:
            prompts = list(_iter_prompt_blocks())
            run_prompt_frontend(
                prompts=prompts,
                args=frontend_args,
                context=frontend_context,
                conversation=[],
                model_path=cfg.model or cfg.profile or "server",
                prompt_config_path=cfg.prompt_config,
                loaded_hyperparameters={},
                loaded_max_context_tokens=None,
                loaded_model_path=None,
                loaded_chat_id=None,
                backend=backend,
            )
            for idx, _ in enumerate(prompts, start=1):
                results["tests"].append({"name": f"chat_completions_{idx}", "status": "pass"})
    except Exception as exc:  # noqa: BLE001
        if results["status"] != "fail":
            results["status"] = "fail"
            results["tests"].append(
                {"name": "chat_completions", "status": "fail", "details": str(exc)}
            )
            print(f"[ERROR] Request failed: {exc}", file=sys.stderr)

    if cfg.report_file:
        cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_file.write_text(json.dumps(results), encoding="utf-8")

    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(str(cfg.profile_output))
        text_path = cfg.profile_output.with_suffix(".stats.txt")
        metrics_path = cfg.profile_output.with_suffix(".metrics.json")
        static_path = cfg.profile_output.with_suffix(".static.txt")

        report_args = [
            sys.executable,
            "scripts/process_profile_artifacts.py",
            "--profile-output",
            str(cfg.profile_output),
            "--profile-text",
            str(text_path),
            "--profile-metrics-json",
            str(metrics_path),
            "--profile-static-txt",
            str(static_path),
        ]
        if cfg.text_only:
            report_args.append("--text-only")
        else:
            if cfg.graph_output is not None:
                dot_path = cfg.graph_output.with_suffix(".dot")
                report_args.extend(["--profile-dot", str(dot_path), "--profile-svg", str(cfg.graph_output)])
            if cfg.node_thres is not None:
                report_args.extend(["--node-thres", str(cfg.node_thres)])
            if cfg.edge_thres is not None:
                report_args.extend(["--edge-thres", str(cfg.edge_thres)])
        sp.run(report_args, check=False)

    return 0 if results["status"] == "ok" else 1


def _sanitize_server_args(argv: List[str]) -> List[str]:
    if not argv:
        return []
    cleaned: List[str] = []
    skip_next = False
    for idx, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == "--":
            continue
        if item in {"--profile-output", "--graph"}:
            skip_next = True
            continue
        cleaned.append(item)
    return cleaned


def _run_server(argv: List[str], profile_output: str | None, graph_output: str | None) -> int:
    import mlx_harmony.server as server_module

    sys.argv = ["server", *_sanitize_server_args(argv)]
    if not profile_output:
        server_module.main()
        return 0

    profile_path = Path(profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    # Use cProfile module directly so the server process is profiled end-to-end.
    # We exec so the PID remains the server process (signals from the harness work).
    cmd = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(profile_path),
        "-m",
        "mlx_harmony.server",
        *sys.argv[1:],
    ]
    os.execvp(sys.executable, cmd)
    return 0


def _run_startup(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Profile startup performance.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--output", default="stats/profile_startup.stats")
    parser.add_argument("--graph", default="stats/profile_startup.svg")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-static", action="store_true")
    parser.add_argument("--full-startup", action="store_true")
    parser.add_argument("--node-thres", type=float, default=None)
    parser.add_argument("--edge-thres", type=float, default=None)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args(argv)

    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import subprocess as sp

    from mlx_harmony.config import load_prompt_config
    from mlx_harmony.generator import TokenGenerator

    profiler = cProfile.Profile()
    profiler.enable()
    _ = TokenGenerator(
        args.model,
        prompt_config=load_prompt_config(args.prompt_config) if args.prompt_config else None,
    )
    profiler.disable()

    output_path = _normalize_stats_path(args.output)
    profiler.dump_stats(str(output_path))

    text_path = output_path.with_suffix(".stats.txt")
    metrics_path = output_path.with_suffix(".metrics.json")
    static_path = output_path.with_suffix(".static.txt")
    graph_path = _normalize_stats_path(args.graph)
    dot_path = graph_path.with_suffix(".dot")

    report_args = [
        sys.executable,
        "scripts/process_profile_artifacts.py",
        "--profile-output",
        str(output_path),
        "--profile-text",
        str(text_path),
        "--profile-metrics-json",
        str(metrics_path),
        "--profile-static-txt",
        str(static_path),
        "--top",
        str(args.top),
    ]
    if args.no_static:
        report_args.append("--no-static")
    if args.text_only:
        report_args.append("--text-only")
    else:
        report_args.extend(["--profile-dot", str(dot_path), "--profile-svg", str(graph_path)])
        if args.node_thres is not None:
            report_args.extend(["--node-thres", str(args.node_thres)])
        if args.edge_thres is not None:
            report_args.extend(["--edge-thres", str(args.edge_thres)])
    sp.run(report_args, check=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified profiling entrypoint.")
    parser.add_argument("mode", choices=["cli", "client", "server", "startup"])
    parser.add_argument("--profile-output", default="")
    parser.add_argument("--graph", default="")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    passthrough = args.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.mode == "cli":
        return _run_cli(passthrough)
    if args.mode == "client":
        return _run_client(passthrough)
    if args.mode == "startup":
        return _run_startup(passthrough)
    return _run_server(passthrough, args.profile_output or None, args.graph or None)


if __name__ == "__main__":
    raise SystemExit(main())
