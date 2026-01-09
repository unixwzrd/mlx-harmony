#!/usr/bin/env python3
"""
Profile mlx-harmony chat startup to identify performance bottlenecks.

Usage:
    python scripts/profile_startup.py --model <model_path> [--output profile.stats] [--graph profile.svg]
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

SRC_PATH = Path(__file__).parent.parent / "src"


def _ensure_src_on_path() -> None:
    """Ensure we can import `mlx_harmony` from the local `src/` tree."""
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def profile_token_generator_init(model_path: str, prompt_config_path: str = None):
    """Profile just the TokenGenerator initialization."""
    print(f"[PROFILE] Profiling TokenGenerator initialization for: {model_path}")

    _ensure_src_on_path()
    from mlx_harmony.config import load_prompt_config
    from mlx_harmony.generator import TokenGenerator

    profiler = cProfile.Profile()
    profiler.enable()

    prompt_config = None
    if prompt_config_path:
        prompt_config = load_prompt_config(prompt_config_path)

    generator = TokenGenerator(model_path, prompt_config=prompt_config)

    profiler.disable()
    return profiler, generator


def profile_chat_startup(model_path: str, prompt_config_path: str = None):
    """Profile the entire chat startup (up to model loading)."""
    print(f"[PROFILE] Profiling chat startup for: {model_path}")

    _ensure_src_on_path()
    from mlx_harmony.config import load_prompt_config
    from mlx_harmony.generator import TokenGenerator

    # Mock sys.argv to simulate CLI args
    original_argv = sys.argv
    sys.argv = [
        "mlx-harmony-chat",
        "--model", model_path,
    ]
    if prompt_config_path:
        sys.argv.extend(["--prompt-config", prompt_config_path])

    profiler = cProfile.Profile()

    try:
        # We'll need to intercept before the interactive loop starts
        # For now, let's just profile the generator init
        profiler.enable()
        _ = TokenGenerator(
            model_path,
            prompt_config=load_prompt_config(prompt_config_path) if prompt_config_path else None,
        )
        profiler.disable()
    except KeyboardInterrupt:
        profiler.disable()
    finally:
        sys.argv = original_argv

    return profiler


def generate_text_report(profiler: cProfile.Profile, output_file: str = None):
    """Generate a text-based profiling report."""
    if output_file:
        with open(output_file, "w") as f:
            stats_file = pstats.Stats(profiler, stream=f)
            stats_file.sort_stats("cumulative")
            stats_file.print_stats(50)
        print(f"[PROFILE] Text report saved to: {output_file}")
    else:
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\n" + "=" * 80)
        print("PROFILING REPORT (Top 50 functions by cumulative time)")
        print("=" * 80)
        stats.print_stats(50)

    # Also print by total time
    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    print("\n" + "=" * 80)
    print("PROFILING REPORT (Top 50 functions by total time)")
    print("=" * 80)
    if output_file:
        with open(output_file, "a") as f:
            stats_file = pstats.Stats(profiler, stream=f)
            stats_file.sort_stats("tottime")
            stats_file.print_stats(50)
    else:
        stats.print_stats(50)


def generate_graphviz_report(profiler: cProfile.Profile, output_file: str):
    """Generate a graphviz visualization using gprof2dot."""
    try:
        import subprocess
        import tempfile

        # Ensure stats directory exists for graph output
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write cProfile stats to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stats", delete=False) as tmp:
            stats_file = tmp.name
            profiler.dump_stats(stats_file)

        # Try to use gprof2dot if available. It outputs DOT, so we run `dot` to make SVG.
        try:
            dot_output = str(out_path.with_suffix(".dot")) if out_path.suffix.lower() == ".svg" else str(out_path)

            subprocess.run(
                ["gprof2dot", "-f", "pstats", stats_file, "-o", dot_output],
                capture_output=True,
                text=True,
                check=True,
            )

            if out_path.suffix.lower() == ".svg":
                subprocess.run(
                    ["dot", "-Tsvg", dot_output, "-o", str(out_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            print(f"[PROFILE] Graphviz visualization saved to: {output_file}")
            print(f"[PROFILE] View with: xdg-open {output_file} (Linux) or open {output_file} (macOS)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[WARNING] gprof2dot and/or graphviz 'dot' not found.")
            print("[INFO] Install: pip install gprof2dot && brew install graphviz  # macOS")
            print("[INFO] You can also use: python -m pstats <stats_file>")
            # Fallback: try snakeviz
            try:
                print(f"[INFO] Stats file saved to: {stats_file}")
                print(f"[INFO] View with: snakeviz {stats_file}")
            except Exception:
                pass

        # Clean up temp file (optional - user might want to keep it)
        # Path(stats_file).unlink()

    except ImportError:
        print("[WARNING] Could not generate graphviz visualization.")
        print("[INFO] Install: pip install gprof2dot && brew install graphviz  # macOS")
        print("[INFO] Or use snakeviz: pip install snakeviz && snakeviz <stats_file>")


def main():
    parser = argparse.ArgumentParser(
        description="Profile mlx-harmony startup performance"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path to profile",
    )
    parser.add_argument(
        "--prompt-config",
        default=None,
        help="Optional prompt config path",
    )
    parser.add_argument(
        "--output",
        default="stats/profile_startup.stats",
        help="Output file for cProfile stats (default: stats/profile_startup.stats)",
    )
    parser.add_argument(
        "--graph",
        default="stats/profile_startup.svg",
        help="Output file for graphviz visualization (default: stats/profile_startup.svg)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only generate text report, skip graphviz",
    )
    parser.add_argument(
        "--full-startup",
        action="store_true",
        help="Profile full chat startup (not just TokenGenerator init)",
    )

    args = parser.parse_args()

    print("[PROFILE] Starting profiling...")
    print(f"[PROFILE] Model: {args.model}")
    if args.prompt_config:
        print(f"[PROFILE] Prompt config: {args.prompt_config}")

    # Profile the startup
    if args.full_startup:
        profiler = profile_chat_startup(args.model, args.prompt_config)
    else:
        profiler, _ = profile_token_generator_init(args.model, args.prompt_config)

    # Ensure stats directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save stats
    profiler.dump_stats(args.output)
    print(f"[PROFILE] Stats saved to: {args.output}")

    # Generate text report
    generate_text_report(profiler, args.output + ".txt")

    # Generate graphviz visualization
    if not args.text_only:
        generate_graphviz_report(profiler, args.graph)

    print("\n[PROFILE] Profiling complete!")
    print(f"[PROFILE] View stats interactively: python -m pstats {args.output}")


if __name__ == "__main__":
    main()
