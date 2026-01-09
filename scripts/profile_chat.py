#!/usr/bin/env python3
"""
Profile mlx-harmony-chat as it runs (real-world usage profiling).

This script runs mlx-harmony-chat with profiling enabled, capturing
performance data during actual usage including:
- Model loading
- Token generation
- Message parsing
- Full chat loop

Usage:
    python scripts/profile_chat.py --model <model_path> [chat_args...] --profile-output profile.stats
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path


def profile_chat_command(
    model_path: str,
    chat_args: list,
    profile_output: str = "profile_chat.stats",
    graph_output: str = "profile_chat.svg",
    text_only: bool = False,
):
    """
    Run mlx-harmony-chat with profiling enabled.

    Args:
        model_path: Model path to use
        chat_args: Additional arguments to pass to mlx-harmony-chat
        profile_output: Output file for cProfile stats
        graph_output: Output file for graphviz visualization
        text_only: Only generate text report, skip graphviz
    """
    print("[PROFILE] Starting profiling of mlx-harmony-chat...")
    print(f"[PROFILE] Model: {model_path}")
    print(f"[PROFILE] Chat args: {chat_args}")

    # Import the chat module
    # Add src to path so we can import mlx_harmony
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from mlx_harmony.chat import main as chat_main

    # Set up sys.argv to simulate command line arguments
    original_argv = sys.argv
    sys.argv = ["mlx-harmony-chat", "--model", model_path] + chat_args

    print("[PROFILE] Note: This will start the chat interface. Type 'q' to quit and finish profiling.")
    print("[PROFILE] Profiling is active - all operations are being recorded.\n")

    # Profile the chat main function
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        chat_main()
        profiler.disable()
    except (KeyboardInterrupt, EOFError, SystemExit):
        profiler.disable()
        print("\n[PROFILE] Profiling stopped.")
    except Exception as e:
        profiler.disable()
        print(f"\n[ERROR] Chat failed: {e}")
        raise
    finally:
        sys.argv = original_argv

    # Ensure stats directory exists and resolve relative paths to stats/
    profile_output_path = Path(profile_output)
    # If path is relative and doesn't start with stats/, put it in stats/ directory
    if not profile_output_path.is_absolute() and not str(profile_output_path).startswith("stats/"):
        profile_output_path = Path("stats") / profile_output_path
    profile_output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_output = str(profile_output_path)

    # Save the profile
    profiler.dump_stats(profile_output)

    if not Path(profile_output).exists():
        print(f"[ERROR] Profile output file not created: {profile_output}")
        return

    print(f"\n[PROFILE] Stats saved to: {profile_output}")

    # Generate text report
    try:
        text_output = profile_output + ".txt"
        with open(text_output, "w") as f:
            stats_file = pstats.Stats(profile_output, stream=f)
            stats_file.sort_stats("cumulative")
            stats_file.print_stats(50)
        print(f"[PROFILE] Text report saved to: {text_output}")

        # Also print top 20 to console
        print("\n" + "=" * 80)
        print("PROFILING REPORT (Top 20 functions by cumulative time)")
        print("=" * 80)
        stats_console = pstats.Stats(profile_output)
        stats_console.sort_stats("cumulative")
        stats_console.print_stats(20)
    except Exception as e:
        print(f"[WARNING] Failed to generate text report: {e}")

    # Generate graphviz visualization
    if not text_only:
        try:
            import subprocess as sp

            # Ensure stats directory exists for graph output and resolve relative paths
            graph_path = Path(graph_output)
            # If path is relative and doesn't start with stats/, put it in stats/ directory
            if not graph_path.is_absolute() and not str(graph_path).startswith("stats/"):
                graph_path = Path("stats") / graph_path
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_output = str(graph_path)
            dot_output = str(graph_path.with_suffix(".dot")) if graph_path.suffix.lower() == ".svg" else str(graph_path)

            sp.run(
                ["gprof2dot", "-f", "pstats", profile_output, "-o", dot_output],
                capture_output=True,
                text=True,
                check=True,
            )

            if graph_path.suffix.lower() == ".svg":
                sp.run(
                    ["dot", "-Tsvg", dot_output, "-o", str(graph_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            print(f"[PROFILE] Graphviz visualization saved to: {graph_output}")
            print(f"[PROFILE] View with: open {graph_output} (macOS) or xdg-open {graph_output} (Linux)")
        except (sp.CalledProcessError, FileNotFoundError):
            print("[WARNING] gprof2dot and/or graphviz 'dot' not found.")
            print("[INFO] Install: pip install gprof2dot && brew install graphviz  # macOS")
            print(f"[INFO] You can also use: python -m pstats {profile_output}")
            print(f"[INFO] Or use snakeviz: pip install snakeviz && snakeviz {profile_output}")
        except Exception as e:
            print(f"[WARNING] Failed to generate graphviz visualization: {e}")

    print("\n[PROFILE] Profiling complete!")
    print(f"[PROFILE] View stats interactively: python -m pstats {profile_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile mlx-harmony-chat as it runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile chat startup and one interaction
  python scripts/profile_chat.py --model models/my-model --prompt-config configs/Mia.json

  # Profile with specific chat arguments
  python scripts/profile_chat.py --model models/my-model --temperature 0.8 --max-tokens 100

  # Profile and save to custom files
  python scripts/profile_chat.py --model models/my-model --profile-output my_profile.stats --graph my_profile.svg
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path to use",
    )
    parser.add_argument(
        "--profile-output",
        default="stats/profile_chat.stats",
        help="Output file for cProfile stats (default: stats/profile_chat.stats)",
    )
    parser.add_argument(
        "--graph",
        default="stats/profile_chat.svg",
        help="Output file for graphviz visualization (default: stats/profile_chat.svg)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only generate text report, skip graphviz",
    )
    # Parse known args and pass anything else through to mlx-harmony-chat.
    # This lets you run the profiler exactly like mlx-harmony-chat, e.g.:
    #   scripts/profile_chat.py --model ... --prompt-config ... --debug-file ...
    args, passthrough_args = parser.parse_known_args()
    # If the user included `--` separator, drop it.
    passthrough_args = [a for a in passthrough_args if a != "--"]

    profile_chat_command(
        model_path=args.model,
        chat_args=passthrough_args,
        profile_output=args.profile_output,
        graph_output=args.graph,
        text_only=args.text_only,
    )


if __name__ == "__main__":
    main()
