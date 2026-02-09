from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Optional


@dataclass(frozen=True)
class ApiClientConfig:
    host: str
    port: int
    model: Optional[str]
    profile: Optional[str]
    prompt_config: Optional[str]
    max_tokens: Optional[int]
    timeout: int
    return_analysis: bool
    requests_log: Optional[Path]


class ApiClient:
    def __init__(self, config: ApiClientConfig) -> None:
        self._config = config

    def health_check(self, retries: int, delay_seconds: float) -> bool:
        url = f"http://{self._config.host}:{self._config.port}/v1/health"
        for _ in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=self._config.timeout) as response:
                    if response.status == 200:
                        return True
            except Exception:
                sleep(delay_seconds)
        return False

    def send_prompt(self, prompt: str) -> dict:
        return self.send_messages([{"role": "user", "content": prompt}])

    def send_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        return_analysis: Optional[bool] = None,
    ) -> dict:
        url = f"http://{self._config.host}:{self._config.port}/v1/chat/completions"
        payload: dict = {
            "messages": messages,
        }
        resolved_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        if resolved_max_tokens is not None:
            payload["max_tokens"] = resolved_max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if min_p is not None:
            payload["min_p"] = min_p
        if top_k is not None:
            payload["top_k"] = top_k
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            payload["repetition_context_size"] = repetition_context_size
        if self._config.model:
            payload["model"] = self._config.model
        if self._config.profile:
            payload["profile"] = self._config.profile
        if self._config.prompt_config:
            payload["prompt_config"] = self._config.prompt_config
        if return_analysis is not None:
            payload["return_analysis"] = return_analysis
        elif self._config.return_analysis:
            payload["return_analysis"] = True
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8") if exc.fp is not None else ""
            raise RuntimeError(f"Server HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Server connection failed: {exc}") from exc
        self._append_request_log(json.dumps(messages, ensure_ascii=False, indent=2), body)
        return body

    def _append_request_log(self, prompt: str, response: dict) -> None:
        if not self._config.requests_log:
            return
        path = self._config.requests_log
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
