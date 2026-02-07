# Profiling Scripts

**Created**: 2026-01-07
**Updated**: 2026-02-04

- [Profiling Scripts](#profiling-scripts)
  - [Profile Running Chat (Real-World Usage) ⭐ Recommended](#profile-running-chat-real-world-usage--recommended)
  - [Script Call Tree](#script-call-tree)
    - [Quick Start](#quick-start)
    - [Deterministic Dataset Runs (No Typing)](#deterministic-dataset-runs-no-typing)
    - [Server Dataset Runs](#server-dataset-runs)
    - [Schema Migration Utility](#schema-migration-utility)
    - [Output Files](#output-files)
    - [Viewing Results](#viewing-results)
  - [Profile Startup Performance](#profile-startup-performance)
    - [Quick Start](#quick-start-1)
    - [Output Files](#output-files-1)
    - [Viewing Results](#viewing-results-1)
    - [Understanding the Output](#understanding-the-output)
    - [Example Output](#example-output)
    - [Tips](#tips)
  - [Benchmark Harness](#benchmark-harness)
  - [API Server Dataset Profile](#api-server-dataset-profile)
    - [Metrics Layout](#metrics-layout)
    - [Logs and Meta Layout](#logs-and-meta-layout)
    - [Plot TPS vs Wired Memory](#plot-tps-vs-wired-memory)
  - [Utility Scripts](#utility-scripts)
    - [Module Dependency Graph](#module-dependency-graph)
    - [API Docs (pydoc-markdown)](#api-docs-pydoc-markdown)

## Profile Running Chat (Real-World Usage) ⭐ Recommended

Use `profile_module.py` to profile the actual `mlx-harmony-chat` command as it runs. This captures **real-world performance** including:

- Model loading
- Token generation
- Message parsing
- Full chat loop interactions
- User input handling

This is more useful than startup-only profiling because it shows performance during actual usage.

## Script Call Tree

Call tree (bulleted hierarchy with links):

- [Benchmark Harness](#benchmark-harness) → [bench_run.sh](./bench_run.sh)
  - [Dataset Harness](#api-server-dataset-profile) → [run_dataset_harness.sh](./run_dataset_harness.sh)
    - Cleanup → [clean_logs.sh](./clean_logs.sh)
    - vm_stat → [filter-vm_stat.py](./filter-vm_stat.py)
    - Dataset/Profile → [run_profile.sh](./run_profile.sh)
      - CLI → [profile_module.py](./profile_module.py) → `mlx_harmony.chat.main`
      - Server + Client → [profile_module.py](./profile_module.py) → `mlx_harmony.server.main`
    - Reports/Plots
      - [process_stats.sh](./process_stats.sh)
      - [process_profile_artifacts.py](./process_profile_artifacts.py)
      - [merge_timing_metrics.py](./merge_timing_metrics.py)
      - [TPSvsWiredMemory.py](./TPSvsWiredMemory.py)
    - Logs → [preserve_logs.sh](./preserve_logs.sh) → runs/<run-id>/logs/<mode>/
    - Finalize → runs/<run-id>/...
    - Repeat if mode=all

Notes:

- `bench_run.sh` is the top-level entry for full benchmark runs.
- `run_dataset_harness.sh` runs one component at a time (CLI or server), completing the full artifact cycle before the next run.
 - Set `DOT_FILTER_SUBSTRING=mlx_harmony` (default) to limit graph nodes to project code. Set `DOT_FILTER=0` to disable filtering.

### Quick Start

```bash
# Profile chat with one interaction (type 'q' to quit and finish profiling)
scripts/profile_module.py cli -- \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Profile with specific chat arguments
scripts/profile_module.py cli -- \
  --model models/my-model \
  --temperature 0.8 \
  --max-tokens 100 \
  --profile-output chat_profile.stats \
  --graph chat_profile.svg

# Include more nodes/edges in the call graph (lower thresholds)
scripts/profile_module.py cli -- \
  --model models/my-model \
  --prompt-config configs/Mia.json \
  --graph chat_profile.svg \
  --node-thres 0.001 \
  --edge-thres 0.001

# Text-only report (faster, no graphviz needed)
scripts/profile_module.py cli -- \
  --model models/my-model \
  --prompt-config configs/Mia.json \
  --text-only
```

**Note**: This will start the actual chat interface. Interact with it normally (ask questions, have a conversation), then type 'q' to quit. The profiling data will be saved when you exit. All operations during the chat session are profiled.

### Deterministic Dataset Runs (No Typing)

If you have a JSON file with an `instruction` field (array of objects), you can stream it to the profiler:

```bash
scripts/build_prompt_stream.py path/to/english.json 200 | \
  scripts/profile_module.py cli -- \
    --model models/your-model \
    --prompt-config configs/prompt-config.deterministic.json
```

The optional last argument (`LIMIT`) caps the number of prompts. This uses [build_prompt_stream.py](./build_prompt_stream.py)
to emit `\\` blocks and a final `q` for the chat loop.

### Server Dataset Runs

Use `profile_module.py` to run the same dataset prompts against the API server workflow and capture
the same metrics under `runs/<run-id>/`.

```bash
scripts/build_prompt_stream.py tests/data/english.json 20 | \
  scripts/profile_module.py client -- \
    --host 127.0.0.1 \
    --port 8000 \
    --model models/your-model
```

### Schema Migration Utility

Use `migrate_chat_schema.py` to convert chat logs to the latest schema:

```bash
scripts/migrate_chat_schema.py logs/profiling-chat.json --in-place
```

### Output Files

All profiling output files are saved to the `stats/` directory by default:

- `stats/profile_chat.stats` - cProfile stats file (can be viewed with `python -m pstats`)
- `stats/profile_chat.stats.txt` - Text report with top functions
- `stats/profile_chat.svg` - Graphviz call graph (if `gprof2dot` + Graphviz `dot` are installed)

### Viewing Results

Same as `profile_module.py startup` - see the "Viewing Results" section below.

---

## Profile Startup Performance

Use `profile_module.py startup` to identify performance bottlenecks during model loading and initialization only (doesn't run the full chat loop).

### Quick Start

```bash
# Profile TokenGenerator initialization (model loading)
scripts/profile_module.py startup -- \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Generate both text and graphviz visualization
scripts/profile_module.py startup -- \
  --model <model_path> \
  --output profile.stats \
  --graph profile.svg

# Text-only report (faster)
scripts/profile_module.py startup -- \
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
scripts/profile_module.py startup -- --model <model_path> --graph stats/profile.svg

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

You can also add the API server dataset run to the benchmark run:

```bash
RUN_SERVER=1 bash scripts/bench_run.sh
```

To profile during the server dataset run (server-side profiling is enabled by default in the bench harness):

```bash
RUN_SERVER=1 INTEGRATION_PROFILE=1 bash scripts/bench_run.sh
```

To include more functions in the call tree, lower the gprof2dot thresholds for the harness run:

```bash
PROFILE_NODE_THRES=0.0001 PROFILE_EDGE_THRES=0.0001 bash scripts/bench_run.sh
```

To include the full call tree (no threshold filtering), set `PROFILE_FULL=1`:

```bash
PROFILE_FULL=1 bash scripts/bench_run.sh
```

For server + client profiling, you can also force a full client call tree:

```bash
RUN_SERVER=1 CLIENT_PROFILE_FULL=1 bash scripts/bench_run.sh
```

To include more functions in Graphviz output, pass thresholds through the report generator:

```bash
scripts/process_profile_artifacts.py \
  --profile-output runs/<run-id>/metrics/cli/profile.stats \
  --profile-text runs/<run-id>/metrics/cli/profile.stats.txt \
  --profile-metrics-json runs/<run-id>/metrics/cli/profile.metrics.json \
  --profile-static-txt runs/<run-id>/metrics/cli/profile.static.txt \
  --profile-dot runs/<run-id>/metrics/cli/profile.dot \
  --profile-svg runs/<run-id>/metrics/cli/profile.svg \
  --node-thres 0.001 \
  --edge-thres 0.001
```

---

## API Server Dataset Profile

Use `profile_module.py` to profile the client while the server runs the dataset prompts.

```bash
scripts/build_prompt_stream.py tests/data/english.json 3 | \
  scripts/profile_module.py client -- \
    --host 127.0.0.1 \
    --port 8000 \
    --model models/your-model
```

Optional overrides (apply to server dataset runs, including the bench harness):

- `INTEGRATION_TURNS`: Number of prompts to send (default: `3`).
- `INTEGRATION_PROMPTS_FILE`: JSON prompt file (default: `tests/data/english.json`).
- `INTEGRATION_REPORT_FILE`: JSON report path (default: `runs/<run-id>/meta/server/server-dataset-report.json`).
- `INTEGRATION_PROFILE`: Set to `1` to capture profile stats/graph for the dataset run.
- `INTEGRATION_REQUEST_TIMEOUT`: Per-request timeout in seconds (default: `300`).
- `INTEGRATION_HEALTH_RETRIES`: Number of health checks before failing (default: `100`).

Server integration runs also capture timing/vm_stat outputs in the same directory:

- `timings-debug.tsv`
- `vm_stat-timing.tsv`
- `merged-timings-vm_stat.tsv`
- `tps_vs_wired.png` (under `plots/server/`)

Server request/response payloads are logged to:

- `logs/server/server-requests.log`

Graphviz filtering is enabled by default to keep only `mlx_harmony` nodes in the SVG output:

- `profile.dot` remains the raw `gprof2dot` output.
- `profile.filtered.dot` is the filtered graph (used for the SVG).

You can override the filter substring or disable filtering:

```bash
DOT_FILTER=0 RUN_SERVER=1 bash scripts/bench_run.sh
DOT_FILTER_SUBSTRING=my_package RUN_SERVER=1 bash scripts/bench_run.sh
```

To profile the STDIO client against the API server (for CLI parity), run:

```bash
scripts/profile_module.py client -- \
  --host 127.0.0.1 \
  --port 8000 \
  --model models/your-model
```

To capture a client-side profile:

```bash
scripts/profile_module.py client -- \
  --host 127.0.0.1 \
  --port 8000 \
  --model models/your-model \
  --profile-output runs/<run-id>/metrics/server/client-profile.stats \
  --graph runs/<run-id>/metrics/server/client-profile.svg
```

### Metrics Layout

`bench_run.sh` separates CLI and server metrics under the run directory:

- `runs/<run-id>/metrics/cli/`: CLI benchmark metrics (timings, vm_stat, merged TSVs, profiles).
- `runs/<run-id>/metrics/server/`: Server-side metrics and client profiles.

### Logs and Meta Layout

Each run also keeps logs and metadata separated by workflow:

- `runs/<run-id>/logs/cli/`: CLI logs and debug artifacts.
- `runs/<run-id>/logs/server/`: Server logs.
- `runs/<run-id>/meta/cli/`: CLI metadata (`run.env`, prompt config copy, vm_stat stderr).
- `runs/<run-id>/meta/server/`: Server metadata (`run.env`, integration report, vm_stat stderr).

The integration and benchmark scripts call [clean_run_artifacts.sh](./clean_run_artifacts.sh) to clear
debug/log artifacts that can skew results.

Shared helpers used by the CLI and server dataset runs live in
[dataset_run_common.sh](./dataset_run_common.sh).

Both CLI and server runs now go through the unified runner:
[run_dataset_harness.sh](./run_dataset_harness.sh).

### Plot TPS vs Wired Memory

Use [TPSvsWiredMemory.py](./TPSvsWiredMemory.py) to plot metrics from a merged TSV. By default it plots
`tokens_per_second` on the left axis and `wired_bytes` on the right axis (converted to GB).

```bash
scripts/TPSvsWiredMemory.py stats/merged-timings-vm_stat.tsv \
  --out stats/tps_vs_wired.png
```

To plot additional columns (example: `elapsed_seconds`, `tokens_per_second`,
`generated_tokens`, `prompt_tokens`) with wired memory on the right axis:

```bash
scripts/TPSvsWiredMemory.py stats/merged-timings-vm_stat.tsv \
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

### Module Dependency Graph

Generate a module dependency graph to compare CLI versus server code paths.

```bash
# Full graph (Graphviz DOT)
scripts/module_dep_graph.py --output stats/module-deps.dot

# CLI-only subgraph
scripts/module_dep_graph.py --entry src/mlx_harmony/chat.py --output stats/cli-deps.dot

# Server-only subgraph
scripts/module_dep_graph.py --entry src/mlx_harmony/server.py --output stats/server-deps.dot

# TSV format for analysis
scripts/module_dep_graph.py --format tsv --output stats/module-deps.tsv
```

If Graphviz is installed, render the DOT output:

```bash
dot -Tsvg stats/cli-deps.dot -o stats/cli-deps.svg
dot -Tsvg stats/server-deps.dot -o stats/server-deps.svg
```

### API Docs (pydoc-markdown)

Generate Markdown API docs from code using `pydoc-markdown`.

```bash
pip install pydoc-markdown
pydoc-markdown -I src -m mlx_harmony > docs/API_REFERENCE.md
```

Notes:

- This is optional and external; the repo does not bundle a custom doc generator.
- Regenerate after refactors that move or rename modules.

- [filter-vm_stat.py](./filter-vm_stat.py): Convert `vm_stat` output to JSON or TSV (with optional byte conversion).
- [merge_timing_metrics.py](./merge_timing_metrics.py): Merge `timings-debug.csv` with `vm_stat` TSV output.
- [build_prompt_stream.py](./build_prompt_stream.py): Stream dataset prompts into the chat loop.
- [profile_module.py](./profile_module.py): Profile CLI/server/client via a single entry point.
