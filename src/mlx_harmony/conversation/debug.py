from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _write_debug_block(debug_path: Path, header: str, payload: str) -> None:
    """Write a labeled debug section to the debug log."""
    with open(debug_path, "a", encoding="utf-8") as df:
        df.write(f"\n{header}\n")
        df.write("-" * 80 + "\n")
        df.write(payload + "\n")
        df.write("-" * 80 + "\n")


def write_debug_prompt(
    *,
    debug_path: Path,
    raw_prompt: str,
    show_console: bool,
) -> None:
    """Write a raw prompt block to the debug log and optionally console."""
    if show_console:
        print("\n[DEBUG] Raw prompt sent to LLM:")
        print("-" * 80)
        print(raw_prompt)
        print("-" * 80)
    _write_debug_block(debug_path, "[DEBUG] Raw prompt sent to LLM:", raw_prompt)


def write_debug_response(
    *,
    debug_path: Path,
    raw_response: str,
    cleaned_response: str,
    show_console: bool,
) -> None:
    """Write a raw response block to the debug log and optionally console."""
    if show_console:
        print("\n[DEBUG] Raw response from LLM:")
        print("-" * 80)
        print(cleaned_response)
        print("-" * 80)
    _write_debug_block(debug_path, "[DEBUG] Raw response from LLM:", raw_response)


def write_debug_tokens(
    *,
    debug_path: Path,
    token_ids: list[int],
    decode_tokens: Callable[[list[int]], str] | None = None,
    label: str = "response",
    enabled: bool = True,
) -> None:
    """Write token IDs and decoded text to the debug log."""
    if not enabled or not token_ids:
        return
    _write_debug_block(
        debug_path,
        f"[DEBUG] {label} tokens ({len(token_ids)} IDs):",
        str(token_ids),
    )
    if decode_tokens is not None:
        decoded_all = decode_tokens(token_ids)
        _write_debug_block(
            debug_path,
            f"[DEBUG] {label} tokens decoded (raw):",
            decoded_all,
        )


def write_debug_metrics(
    *,
    debug_path: Path,
    metrics: dict[str, Any],
) -> None:
    """Write generation metrics to the debug log."""
    _write_debug_block(
        debug_path,
        "[DEBUG] Generation metrics:",
        json.dumps(metrics, indent=2, ensure_ascii=False),
    )
    _write_debug_block(
        debug_path,
        "[DEBUG] Generation metrics (TSV):",
        _format_metrics_tsv(metrics),
    )


def _format_metrics_tsv(metrics: dict[str, Any]) -> str:
    """Format metrics as a TSV line for easy extraction."""
    keys = [
        "prompt_tokens",
        "generated_tokens",
        "elapsed_seconds",
        "tokens_per_second",
        "prompt_start_to_prompt_start_seconds",
        "max_context_tokens",
    ]
    values = [metrics.get(key, "") for key in keys]
    return "TIMING_STATS\t" + "\t".join(str(v) for v in values)
