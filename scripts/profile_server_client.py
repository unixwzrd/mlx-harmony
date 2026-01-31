#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Iterator, Optional


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


def append_request_log(path: Path, prompt: str, response: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"[{timestamp}] Prompt\n")
        handle.write("-" * 80 + "\n")
        handle.write(prompt)
        handle.write("\n" + "-" * 80 + "\n")
        handle.write(f"[{timestamp}] Response (raw JSON)\n")
        handle.write("-" * 80 + "\n")
        handle.write(json.dumps(response, ensure_ascii=False, indent=2))
        handle.write("\n")


def send_prompt(cfg: RequestConfig, prompt: str) -> dict:
    url = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.max_tokens,
    }
    if cfg.model:
        payload["model"] = cfg.model
    if cfg.profile:
        payload["profile"] = cfg.profile
    if cfg.prompt_config:
        payload["prompt_config"] = cfg.prompt_config
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if cfg.requests_log:
        append_request_log(cfg.requests_log, prompt, body)
    return body


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
    )

    results = {"status": "ok", "tests": []}
    health_url = f"http://{cfg.host}:{cfg.port}/v1/health"
    health_ok = False
    for _ in range(cfg.health_retries):
        try:
            with urllib.request.urlopen(health_url, timeout=cfg.timeout) as response:
                if response.status == 200:
                    health_ok = True
                    break
        except Exception:
            sleep(cfg.health_sleep)
    if not health_ok:
        msg = f"Server health check failed: {health_url}"
        results["status"] = "fail"
        results["tests"].append({"name": "health_check", "status": "fail", "details": msg})
        if cfg.report_file:
            cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
            cfg.report_file.write_text(json.dumps(results), encoding="utf-8")
        print(msg, file=sys.stderr)
        return 1
    idx = 0
    for prompt in iter_prompt_blocks():
        idx += 1
        try:
            body = send_prompt(cfg, prompt)
        except Exception as exc:  # noqa: BLE001
            results["status"] = "fail"
            results["tests"].append(
                {"name": f"chat_completions_{idx}", "status": "fail", "details": str(exc)}
            )
            break
        if "choices" not in body:
            results["status"] = "fail"
            results["tests"].append(
                {"name": f"chat_completions_{idx}", "status": "fail", "details": "missing choices"}
            )
            break
        results["tests"].append({"name": f"chat_completions_{idx}", "status": "pass"})

    if cfg.report_file:
        cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_file.write_text(json.dumps(results), encoding="utf-8")

    return 0 if results["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
