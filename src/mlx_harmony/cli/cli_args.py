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
        "--no-fs-cache",
        action="store_true",
        default=False,
        help="Disable filesystem cache when reading model weights (macOS only, experimental).",
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
        "--no-markdown",
        action="store_true",
        default=False,
        help="Disable markdown rendering for assistant responses (display as plain text).",
    )
    parser.add_argument(
        "--moshi",
        action="store_true",
        default=False,
        help="Enable Moshi STT/TTS voice mode (requires moshi-mlx).",
    )
    parser.add_argument(
        "--moshi-config",
        type=str,
        default=None,
        help="Path to Moshi JSON config (CLI options override the file).",
    )
    parser.add_argument(
        "--moshi-stt-path",
        type=str,
        default=None,
        help="Local path to Moshi STT MLX model weights.",
    )
    parser.add_argument(
        "--moshi-stt-config",
        type=str,
        default=None,
        help="Path to Moshi STT config.json (override model directory default).",
    )
    parser.add_argument(
        "--moshi-max-seconds",
        type=float,
        default=None,
        help="Max seconds to listen for STT (voice mode).",
    )
    parser.add_argument(
        "--moshi-vad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Moshi VAD when supported by the STT model.",
    )
    parser.add_argument(
        "--moshi-vad-threshold",
        type=float,
        default=None,
        help="VAD probability threshold to detect end of utterance.",
    )
    parser.add_argument(
        "--moshi-vad-hits",
        type=int,
        default=None,
        help="Consecutive VAD hits required to end an utterance.",
    )
    parser.add_argument(
        "--moshi-silence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable RMS-based silence detection for STT end-of-utterance.",
    )
    parser.add_argument(
        "--moshi-silence-threshold",
        type=float,
        default=None,
        help="RMS threshold used to detect silence for STT end-of-utterance.",
    )
    parser.add_argument(
        "--moshi-silence-ms",
        type=int,
        default=None,
        help="Milliseconds of trailing silence required to end an utterance.",
    )
    parser.add_argument(
        "--moshi-min-speech-ms",
        type=int,
        default=None,
        help="Minimum milliseconds of speech before silence can end an utterance.",
    )
    parser.add_argument(
        "--moshi-stt-block-ms",
        type=int,
        default=None,
        help="STT audio block duration in milliseconds (larger reduces CPU).",
    )
    parser.add_argument(
        "--moshi-stt-warmup-blocks",
        type=int,
        default=None,
        help="Number of initial audio blocks to discard when starting STT listening.",
    )
    parser.add_argument(
        "--moshi-barge-in",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable barge-in (interrupt TTS on user speech).",
    )
    parser.add_argument(
        "--moshi-barge-in-window",
        type=float,
        default=None,
        help="Seconds to listen for barge-in during TTS playback.",
    )
    parser.add_argument(
        "--moshi-tts-path",
        type=str,
        default=None,
        help="Local path to Moshi TTS MLX model weights.",
    )
    parser.add_argument(
        "--moshi-tts-config",
        type=str,
        default=None,
        help="Path to Moshi TTS config.json (override model directory default).",
    )
    parser.add_argument(
        "--moshi-voice-path",
        type=str,
        default=None,
        help="Local path to Moshi TTS voice embedding file.",
    )
    parser.add_argument(
        "--moshi-quantize",
        type=int,
        default=None,
        help="Quantize Moshi model weights (e.g., 4 or 8).",
    )
    parser.add_argument(
        "--moshi-tts-chunk-chars",
        type=int,
        default=None,
        help="Chunk assistant text for TTS by max character count.",
    )
    parser.add_argument(
        "--moshi-tts-chunk-sentences",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Chunk assistant text for TTS on sentence boundaries when possible.",
    )
    parser.add_argument(
        "--moshi-tts-chunk-min-chars",
        type=int,
        default=None,
        help="Minimum character length for each TTS chunk (helps avoid tiny fragments).",
    )
    parser.add_argument(
        "--moshi-tts-stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stream TTS chunks during generation (asynchronous playback).",
    )
    parser.add_argument(
        "--moshi-stt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Moshi STT (default: enabled).",
    )
    parser.add_argument(
        "--moshi-tts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Moshi TTS (default: enabled).",
    )
    parser.add_argument(
        "--moshi-smoke",
        action="store_true",
        default=False,
        help="Run a Moshi smoke test and exit (uses configured STT/TTS).",
    )
    return parser
