from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for chat."""
    parser = argparse.ArgumentParser(description="Chat with any MLX-LM model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or Hugging Face repo (or set via --profile).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Enable browser tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--python",
        dest="use_python",
        action="store_true",
        help="Enable Python tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--apply-patch",
        action="store_true",
        help="Enable apply_patch tool (GPT-OSS only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config/default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum prompt context tokens (overrides config/metadata).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=None,
        help="Number of previous tokens used for repetition penalty.",
    )
    parser.add_argument(
        "--loop-detection",
        choices=["off", "cheap", "full"],
        default=None,
        help="Loop detection mode: off, cheap, or full (overrides config).",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to JSON file with Harmony prompt configuration (GPT-OSS).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Named profile from a profiles JSON (see --profiles-file).",
    )
    parser.add_argument(
        "--profiles-file",
        type=str,
        default="configs/profiles.example.json",
        help="Path to profiles JSON (default: configs/profiles.example.json)",
    )
    parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Chat name (loads from chats_dir/<name>.json if exists, otherwise creates new chat).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: output raw prompts and responses as text.",
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        default=None,
        help="Path to write debug prompts/responses when debug is enabled (default: logs_dir/prompt-debug.log).",
    )
    parser.add_argument(
        "--debug-tokens",
        nargs="?",
        const="out",
        choices=["in", "out", "both"],
        default=None,
        help="Write token IDs and decoded tokens to the debug log. "
        "Use 'in' for prompt tokens, 'out' for response tokens, or 'both'. "
        "If set with no value, defaults to 'out'.",
    )
    parser.add_argument(
        "--mlock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Lock model weights in memory using MLX's wired limit (mlock equivalent, macOS Metal only). "
        "Can also be set in prompt config JSON. Use --mlock to enable or --no-mlock to disable. Default: None (use config or False)",
    )
    parser.add_argument(
        "--lazy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable lazy loading of model weights. Use --lazy to enable or --no-lazy to disable. "
        "Default: None (use library default, typically False).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (-1 for random each run).",
    )
    parser.add_argument(
        "--reseed-each-turn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reseed before each generation when seed >= 0.",
    )
    parser.add_argument(
        "--performance-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable performance mode overrides from prompt config.",
    )
    parser.add_argument(
        "--perf-max-tokens",
        type=int,
        default=None,
        help="Performance mode override for max_tokens.",
    )
    parser.add_argument(
        "--perf-max-context-tokens",
        type=int,
        default=None,
        help="Performance mode override for max_context_tokens.",
    )
    parser.add_argument(
        "--perf-max-kv-size",
        type=int,
        default=None,
        help="Performance mode override for max_kv_size.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        default=False,
        help="Disable markdown rendering for assistant responses (display as plain text).",
    )
    return parser
