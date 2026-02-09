# Profiling Scripts

**Created**: 2026-01-07  
**Updated**: 2026-02-08

This directory contains the benchmark/profiling harness for CLI and server/client parity runs.

## Top-Level Flow

- `scripts/bench_run.sh`
  - Creates `runs/<run-id>/...` bundle directories.
  - Runs one or both components (`cli`, `server`, or `all`).
  - Collects one analysis snapshot (`function-analysis`, LOC stats).
- `scripts/run_dataset_harness.sh`
  - Runs exactly one component (`cli` or `server`) per invocation.
  - Cleans staged `logs/` artifacts before run.
  - Starts/stops `vm_stat` capture for the component run.
  - Executes profile runner (`profile_cli.sh` or `profile_server.sh`).
  - Preserves staged logs into `runs/<run-id>/logs/<component>/`.
  - Generates merged metrics and plots.

## Quick Start

```bash
# CLI + server/client
bash scripts/bench_run.sh all

# CLI only
bash scripts/bench_run.sh cli

# Server/client only
bash scripts/bench_run.sh server
```

Optional positional arguments:

```text
bash scripts/bench_run.sh [mode] [dataset] [model_path] [prompt_config] [turn_limit]
```

Defaults:

- `mode`: `all`
- `dataset`: `tests/data/english.json`
- `model_path`: `models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx`
- `prompt_config`: `configs/prompt-config.deterministic.json`
- `turn_limit`: `20`

## Script Responsibilities

- `bench_run.sh`
  - Orchestrator only.
  - Decides which components to run.
  - Creates run bundle directory structure.

- `run_dataset_harness.sh`
  - Component runner only.
  - Uses absolute paths internally to avoid cwd/path drift.

- `profile_cli.sh`
  - Profiles `mlx_harmony.chat` using dataset prompts from STDIN.
  - Produces one cProfile stats file for CLI.

- `profile_server.sh`
  - Starts profiled `mlx_harmony.server`.
  - Runs profiled `mlx_harmony.client` against it via HTTP.
  - Produces two cProfile stats files in the server metrics directory:
    - `profile.stats` (server)
    - `client-profile.stats` (client)

- `profile_module.py`
  - Generic cProfile wrapper around a Python `--module` or `--script`.
  - Does not perform benchmark artifact processing.

- `clean_logs.sh`
  - Removes staged log artifacts from `logs/`.

- `preserve_logs.sh`
  - Moves staged log artifacts from `logs/` into run bundle `logs/<component>/`.

- `process_stats.sh`
  - Extracts timings from debug log.
  - Merges timings with vm_stat.
  - Generates TPS vs wired-memory plot.
  - Converts `.stats` profile output into `.txt`, `.dot`, `.svg`, metrics JSON.

- `process_profile_artifacts.py`
  - Standalone profile artifact conversion for any `.stats` file.

- `dataset_run_common.sh`
  - Shared helper functions (vm_stat lifecycle, paths, merge/plot helpers).

## Run Bundle Layout

For `runs/<run-id>/`:

- `meta/cli/`, `meta/server/`
- `metrics/cli/`, `metrics/server/`
- `logs/cli/`, `logs/server/`
- `plots/cli/`, `plots/server/`
- `analysis/`

Server component run stores both server and client profile artifacts under:

- `metrics/server/profile.*`
- `metrics/server/client-profile.*`

## DOT Filtering

`process_stats.sh` uses these environment variables when creating profile graphs:

- `DOT_FILTER` (default `1`)
- `DOT_FILTER_SUBSTRING` (default `mlx_harmony`)
- `DOT_FILTER_KEEP_FULL` (default `0`)

Examples:

```bash
# Disable filtering
DOT_FILTER=0 bash scripts/bench_run.sh all

# Filter to a different package name
DOT_FILTER_SUBSTRING=my_package bash scripts/bench_run.sh all
```

## Direct Utilities

- `build_prompt_stream.py`: emits deterministic prompt stream for STDIN-driven runs.
- `filter-vm_stat.py`: converts `vm_stat` output into structured TSV.
- `merge_timing_metrics.py`: joins timing TSV and vm_stat TSV.
- `TPSvsWiredMemory.py`: generates plot from merged TSV.
- `module_dep_graph.py`: emits module dependency DOT/TSV.

