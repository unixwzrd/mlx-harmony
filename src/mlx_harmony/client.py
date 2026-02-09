from __future__ import annotations

"""Thin API client entrypoint for server-backed chat."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from mlx_harmony.api_client import ApiClient, ApiClientConfig


def _iter_prompt_blocks() -> Iterable[str]:
    collecting = False
    buffer: list[str] = []
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped == "q":
            break
        if stripped == "\\":
            if collecting:
                yield "\n".join(buffer)
                buffer.clear()
                collecting = False
            else:
                collecting = True
            continue
        if collecting:
            buffer.append(line)
        elif stripped:
            yield line


def _extract_message_fields(body: dict) -> tuple[str, str]:
    """Extract analysis and content fields from an OpenAI-style response body.

    Args:
        body: Response payload from the chat completions endpoint.

    Returns:
        Tuple of (analysis_text, content_text). Missing fields are returned as
        empty strings.
    """
    choices = body.get("choices", [])
    if not choices:
        return "", ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return "", ""
    analysis = message.get("analysis", "")
    content = message.get("content", "")
    return str(analysis) if analysis else "", str(content) if content else ""


def _print_response(body: dict) -> str:
    """Render a response body to stdout and return assistant content text.

    Args:
        body: Response payload from the chat completions endpoint.

    Returns:
        Assistant content text, or an empty string when unavailable.
    """
    choices = body.get("choices", [])
    if not choices:
        print("[ERROR] Invalid response: missing choices", file=sys.stderr)
        return ""
    analysis, content = _extract_message_fields(body)
    if analysis:
        print(f"[THINKING - {analysis}]")
        print()
    if content:
        print(f"Assistant: {content}")
        print()
    return content


def _interactive_loop(client: ApiClient) -> int:
    """Run interactive stdin/stdout chat mode against the API server.

    Args:
        client: API client used to submit chat completion requests.

    Returns:
        Process exit code.
    """
    print("[INFO] API client connected. Type 'q' to quit.")
    print("[INFO] Use '\\\\' on a line by itself to start/end multiline input.")
    conversation: list[dict[str, str]] = []
    while True:
        try:
            prompt = input(">> ")
        except EOFError:
            print()
            return 0
        if prompt.strip() == "q":
            return 0
        if prompt.strip() == "\\":
            lines: list[str] = []
            while True:
                try:
                    line = input("... ")
                except EOFError:
                    print()
                    return 0
                if line.strip() == "\\":
                    break
                lines.append(line)
            prompt = "\n".join(lines)
            if not prompt.strip():
                continue
        if not prompt.strip():
            continue
        conversation.append({"role": "user", "content": prompt})
        body = client.send_messages(conversation, return_analysis=True)
        assistant_text = _print_response(body)
        if assistant_text:
            conversation.append({"role": "assistant", "content": assistant_text})


def main() -> int:
    parser = argparse.ArgumentParser(description="Thin HTTP client for mlx_harmony.server.")
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
    args = parser.parse_args()

    report_file = Path(args.report_file) if args.report_file else None
    requests_log = Path(args.requests_log) if args.requests_log else None
    results = {"status": "ok", "tests": []}

    client = ApiClient(
        ApiClientConfig(
            host=args.host,
            port=args.port,
            model=args.model or None,
            profile=args.profile or None,
            prompt_config=args.prompt_config or None,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            return_analysis=True,
            requests_log=requests_log,
        )
    )

    if not client.health_check(args.health_retries, args.health_sleep):
        msg = f"Server health check failed: http://{args.host}:{args.port}/v1/health"
        results["status"] = "fail"
        results["tests"].append({"name": "health_check", "status": "fail", "details": msg})
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(results), encoding="utf-8")
        print(msg, file=sys.stderr)
        return 1

    try:
        if sys.stdin.isatty():
            return _interactive_loop(client)

        prompts = list(_iter_prompt_blocks())
        conversation: list[dict[str, str]] = []
        for idx, prompt in enumerate(prompts, start=1):
            conversation.append({"role": "user", "content": prompt})
            body = client.send_messages(conversation, return_analysis=True)
            assistant_text = _print_response(body)
            if assistant_text:
                conversation.append({"role": "assistant", "content": assistant_text})
            results["tests"].append({"name": f"chat_completions_{idx}", "status": "pass"})
    except Exception as exc:  # noqa: BLE001
        results["status"] = "fail"
        results["tests"].append({"name": "chat_completions", "status": "fail", "details": str(exc)})
        print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(results), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
