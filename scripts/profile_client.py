#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

from mlx_harmony.api_client import ApiClient, ApiClientConfig
from mlx_harmony.chat_backend import ServerBackend
from mlx_harmony.chat_frontend import run_cli_frontend, run_prompt_frontend
from mlx_harmony.chat_utils import get_assistant_name, get_truncate_limits
from mlx_harmony.config import load_prompt_config
from mlx_harmony.generation.client import GenerationClient, ServerGenerationClient


@dataclass
class RequestConfig:
    host: str
    port: int
    model: Optional[str]
    profile: Optional[str]
    prompt_config: Optional[str]
    max_tokens: int
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


def iter_prompt_blocks() -> Iterator[str]:
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


def build_api_client(cfg: RequestConfig) -> ApiClient:
    client_cfg = ApiClientConfig(
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
    return ApiClient(client_cfg)

def build_generation_client(cfg: RequestConfig) -> GenerationClient:
    return ServerGenerationClient(
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


def build_frontend_context(cfg: RequestConfig) -> SimpleNamespace:
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


def build_frontend_args(cfg: RequestConfig) -> SimpleNamespace:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Send prompt stream to the MLX Harmony API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--prompt-config", default="")
    parser.add_argument("--max-tokens", type=int, default=512)
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
    args = parser.parse_args()

    cfg = RequestConfig(
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

    profiler = None
    if cfg.profile_output is not None:
        import cProfile

        cfg.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()

    results = {"status": "ok", "tests": []}
    health_client = build_api_client(cfg)
    if not health_client.health_check(cfg.health_retries, cfg.health_sleep):
        msg = f"Server health check failed: http://{cfg.host}:{cfg.port}/v1/health"
        results["status"] = "fail"
        results["tests"].append({"name": "health_check", "status": "fail", "details": msg})
        if cfg.report_file:
            cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
            cfg.report_file.write_text(json.dumps(results), encoding="utf-8")
        print(msg, file=sys.stderr)
        return 1

    generation_client = build_generation_client(cfg)
    backend = ServerBackend(generation_client)
    frontend_context = build_frontend_context(cfg)
    frontend_args = build_frontend_args(cfg)

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
            prompts = list(iter_prompt_blocks())
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
                results["tests"].append(
                    {"name": f"chat_completions_{idx}", "status": "pass"}
                )
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
        args_list = [
            sys.executable,
            "scripts/generate_reports.py",
            str(cfg.profile_output),
        ]
        if cfg.graph_output is not None:
            args_list.extend(["--graph-output", str(cfg.graph_output)])
        if cfg.node_thres is not None:
            args_list.extend(["--node-thres", str(cfg.node_thres)])
        if cfg.edge_thres is not None:
            args_list.extend(["--edge-thres", str(cfg.edge_thres)])
        if cfg.text_only:
            args_list.append("--text-only")
        else:
            args_list.extend(
                ["--text-output", f"{cfg.profile_output}.txt"]
            )
        subprocess.run(args_list, check=False)

    return 0 if results["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
