from __future__ import annotations

import argparse

from mlx_harmony.config import (
    apply_performance_overrides,
    load_profiles,
    load_prompt_config,
)
from mlx_harmony.generator import TokenGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot text generation with MLX Harmony.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or Hugging Face repo (or set via --profile).",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config/default).",
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
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to JSON file with Harmony prompt configuration (GPT-OSS).",
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
    args = parser.parse_args()

    # Resolve profile/model/prompt_config
    profile_model = None
    profile_prompt_cfg = None
    if args.profile:
        profiles = load_profiles(args.profiles_file)
        if args.profile not in profiles:
            raise SystemExit(
                f"Profile '{args.profile}' not found in {args.profiles_file}"
            )
        profile = profiles[args.profile]
        profile_model = profile.get("model")
        profile_prompt_cfg = profile.get("prompt_config")

    model_path = args.model or profile_model
    if not model_path:
        raise SystemExit("Model must be provided via --model or --profile")

    prompt_config_path = args.prompt_config or profile_prompt_cfg
    prompt_config = (
        load_prompt_config(prompt_config_path) if prompt_config_path else None
    )
    prompt_config = apply_performance_overrides(
        prompt_config,
        performance_mode=args.performance_mode,
        perf_max_tokens=args.perf_max_tokens,
        perf_max_context_tokens=args.perf_max_context_tokens,
        perf_max_kv_size=args.perf_max_kv_size,
    )

    generator = TokenGenerator(model_path, prompt_config=prompt_config)
    tokens = list(
        generator.generate(
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
            loop_detection=args.loop_detection,
        )
    )
    text = generator.tokenizer.decode([int(t) for t in tokens])
    print(text)


if __name__ == "__main__":
    main()
