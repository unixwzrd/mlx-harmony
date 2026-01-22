#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai_harmony import Role, StreamableParser

from mlx_harmony.config import load_prompt_config
from mlx_harmony.generator import TokenGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harmony repro harness")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--prompt-config",
        default="configs/prompt-config.deterministic.json",
        help="Prompt config JSON path",
    )
    parser.add_argument(
        "--prompt",
        default="How do I solve a crossword puzzle?",
        help="Prompt text to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens for generation",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    prompt_config = load_prompt_config(args.prompt_config)
    generator = TokenGenerator(
        args.model,
        prompt_config=prompt_config,
        mlock=prompt_config.mlock if prompt_config else False,
    )

    messages = [{"role": "user", "content": args.prompt}]
    prompt_max_tokens = prompt_config.max_tokens if prompt_config else None
    max_tokens = args.max_tokens if args.max_tokens is not None else prompt_max_tokens

    parser_state = None
    final_boundary_detected = False
    last_message_count = 0
    stream_parser = None
    if generator.use_harmony and generator.encoding is not None:
        stream_parser = StreamableParser(generator.encoding, Role.ASSISTANT, strict=False)

    tokens: list[int] = []
    start_time = time.perf_counter()
    for token_id in generator.generate(
        messages=messages,
        max_tokens=max_tokens,
        system_message=None,
    ):
        token_int = int(token_id)
        tokens.append(token_int)
        if stream_parser is not None:
            try:
                stream_parser.process(token_int)
            except Exception:
                pass
            if stream_parser.messages:
                current_count = len(stream_parser.messages)
                if current_count > last_message_count:
                    last_message_count = current_count
                    last_msg = stream_parser.messages[-1]
                    channel = getattr(last_msg, "channel", None)
                    if channel == "final":
                        final_boundary_detected = True
                        break

    end_time = time.perf_counter()
    decoded = (
        generator.encoding.decode(tokens)
        if generator.encoding is not None
        else generator.tokenizer.decode(tokens)
    )
    finish_reason = generator.last_finish_reason or ("length" if max_tokens else "unknown")
    elapsed = end_time - start_time
    tokens_per_second = len(tokens) / elapsed if elapsed > 0 else 0.0

    print(decoded[:200])
    print(
        f"finish_reason={finish_reason} final_boundary={final_boundary_detected} stop_token_id={generator.last_stop_token_id}"
    )
    print(f"tokens={len(tokens)} elapsed_seconds={elapsed:.4f} tokens_per_second={tokens_per_second:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
