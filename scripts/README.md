# Profiling Scripts

**Created**: 2026-01-07
**Updated**: 2026-01-28

## Profile Running Chat (Real-World Usage) ⭐ Recommended

Use `profile_chat.py` to profile the actual `mlx-harmony-chat` command as it runs. This captures **real-world performance** including:

- Model loading
- Token generation
- Message parsing
- Full chat loop interactions
- User input handling

This is more useful than `profile_startup.py` because it shows performance during actual usage.

### Quick Start

```bash
# Profile chat with one interaction (type 'q' to quit and finish profiling)
python scripts/profile_chat.py \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Profile with specific chat arguments
python scripts/profile_chat.py \
  --model models/my-model \
  --temperature 0.8 \
  --max-tokens 100 \
  --profile-output chat_profile.stats \
  --graph chat_profile.svg

# Text-only report (faster, no graphviz needed)
python scripts/profile_chat.py \
  --model models/my-model \
  --prompt-config configs/Mia.json \
  --text-only
```

**Note**: This will start the actual chat interface. Interact with it normally (ask questions, have a conversation), then type 'q' to quit. The profiling data will be saved when you exit. All operations during the chat session are profiled.

### Deterministic Dataset Runs (No Typing)

If you have a JSON file with an `instruction` field (array of objects), you can stream it to the profiler:

```bash
scripts/profile_chat_dataset.sh path/to/english.json models/your-model 200
```

The optional last argument (`LIMIT`) caps the number of prompts. This uses [build_prompt_stream.py](./build_prompt_stream.py)
to emit `\\` blocks and a final `q` for the chat loop.

### Schema Migration Utility

Use `migrate_chat_schema.py` to convert chat logs to the latest schema:

```bash
python scripts/migrate_chat_schema.py logs/profiling-chat.json --in-place
```

### Output Files

All profiling output files are saved to the `stats/` directory by default:

- `stats/profile_chat.stats` - cProfile stats file (can be viewed with `python -m pstats`)
- `stats/profile_chat.stats.txt` - Text report with top functions
- `stats/profile_chat.svg` - Graphviz call graph (if `gprof2dot` + Graphviz `dot` are installed)

### Viewing Results

Same as `profile_startup.py` - see the "Viewing Results" section below.

---

## Profile Startup Performance

Use `profile_startup.py` to identify performance bottlenecks during model loading and initialization only (doesn't run the full chat loop).

### Quick Start

```bash
# Profile TokenGenerator initialization (model loading)
python scripts/profile_startup.py \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Generate both text and graphviz visualization
python scripts/profile_startup.py \
  --model <model_path> \
  --output profile.stats \
  --graph profile.svg

# Text-only report (faster)
python scripts/profile_startup.py \
  --model <model_path> \
  --text-only
```

### Output Files

All profiling output files are saved to the `stats/` directory by default:

- `stats/profile_startup.stats` - cProfile stats file (can be viewed with `python -m pstats`)
- `stats/profile_startup.stats.txt` - Text report with top functions
- `stats/profile_startup.svg` - Graphviz call graph (if `gprof2dot` + Graphviz `dot` are installed)

### Viewing Results

**Text Report:**

```bash
# View the text report
cat profile_startup.stats.txt

# Or use pstats interactively
python -m pstats profile_startup.stats
```

**Graphviz Visualization:**

```bash
# Install gprof2dot (if not already installed)
pip install gprof2dot

# Install Graphviz (provides the `dot` binary)
brew install graphviz  # macOS

# Generate SVG
python scripts/profile_startup.py --model <model_path> --graph stats/profile.svg

# View on macOS
open stats/profile.svg

# View on Linux
xdg-open stats/profile.svg
```

**Interactive Visualization (SnakeViz):**

```bash
# Install snakeviz
pip install snakeviz

# View stats interactively in browser
snakeviz stats/profile_startup.stats
```

### Understanding the Output

The profiling report shows:

- **Cumulative time**: Total time including subcalls
- **Total time**: Time spent in the function itself (excluding subcalls)

Look for:

- Functions with high cumulative time that call many subfunctions
- Functions with high total time that are doing expensive work
- Model loading operations (`mlx_lm.load`)
- Tokenizer initialization
- Harmony encoding loading

### Example Output

```text
PROFILING REPORT (Top 50 functions by cumulative time)
============================================================
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   45.234   45.234 generator.py:33(__init__)
        1    0.000    0.000   44.891   44.891 mlx_lm/__init__.py:123(load)
        1    0.000    0.000   38.234   38.234 mlx_lm/models/llama.py:45(load_model)
    ...
```

### Tips

1. **Model Loading is Usually the Bottleneck**: Most startup time is spent loading model weights from disk. This is expected and hard to optimize.

2. **Check for Redundant Imports**: Look for modules being imported multiple times.

3. **Lazy Loading**: Consider using `lazy=True` in `TokenGenerator` if you want to defer model loading.

4. **Profile Specific Operations**: Modify the script to profile specific parts of the startup if needed.

---

## Benchmark Harness

Use [bench_run.sh](./bench_run.sh) to run the end-to-end benchmark harness (dataset run + vm_stat capture + metrics merge).

**Typical usage:**

```bash
bash scripts/bench_run.sh
```

**Optional modes:**

```bash
# Preserve TTY formatting while logging
BENCH_TTY=1 bash scripts/bench_run.sh

# Stream output with tee (no TTY)
BENCH_TEE=1 bash scripts/bench_run.sh
```

### Plot TPS vs Wired Memory

Use [TPSvsWiredMemory.py](./TPSvsWiredMemory.py) to plot metrics from a merged TSV. By default it plots
`tokens_per_second` on the left axis and `wired_bytes` on the right axis (converted to GB).

```bash
python scripts/TPSvsWiredMemory.py stats/merged-timings-vm_stat.tsv \
  --out stats/tps_vs_wired.png
```

To plot additional columns (example: `elapsed_seconds`, `tokens_per_second`,
`generated_tokens`, `prompt_tokens`) with wired memory on the right axis:

```bash
python scripts/TPSvsWiredMemory.py stats/merged-timings-vm_stat.tsv \
  --x-col elapsed_seconds \
  --left-cols tokens_per_second,generated_tokens,prompt_tokens \
  --right-cols wired_bytes \
  --out stats/tps_vs_wired.png
```

This script expects:

- [filter-vm_stat.py](./filter-vm_stat.py) for vm_stat capture.
- [merge_timing_metrics.py](./merge_timing_metrics.py) for joining timing + vm_stat TSVs.

---

## Utility Scripts

- [filter-vm_stat.py](./filter-vm_stat.py): Convert `vm_stat` output to JSON or TSV (with optional byte conversion).
- [merge_timing_metrics.py](./merge_timing_metrics.py): Merge `timings-debug.csv` with `vm_stat` TSV output.
- [build_prompt_stream.py](./build_prompt_stream.py): Stream dataset prompts into the chat loop.
- [profile_chat_dataset.sh](./profile_chat_dataset.sh): Run deterministic dataset prompts for profiling.
