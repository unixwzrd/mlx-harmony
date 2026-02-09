#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

MODEL_PATH=""
PROMPT_CONFIG=""
DATASET=""
TURNS=""
PROFILE_OUTPUT=""
BENCH_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --prompt-config) PROMPT_CONFIG="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --profile-output) PROFILE_OUTPUT="$2"; shift 2 ;;
    --bench-log) BENCH_LOG="$2"; shift 2 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$MODEL_PATH" && -n "$PROMPT_CONFIG" && -n "$DATASET" && -n "$TURNS" && -n "$PROFILE_OUTPUT" && -n "$BENCH_LOG" ]] \
  || die "Missing required arguments"

set +e
scripts/build_prompt_stream.py "$DATASET" "$TURNS" | \
  PYTHONFAULTHANDLER=1 python3 scripts/profile_module.py \
    --module mlx_harmony.chat \
    --profile-output "$PROFILE_OUTPUT" \
    -- \
    --model "$MODEL_PATH" \
    --prompt-config "$PROMPT_CONFIG" \
    --debug-file "debug.log" \
    > "$BENCH_LOG" 2>&1
rc=$?
if [[ "$rc" -eq 134 ]]; then
  echo "[WARNING] CLI profiling aborted with exit 134; retrying once..." >&2
  scripts/build_prompt_stream.py "$DATASET" "$TURNS" | \
    PYTHONFAULTHANDLER=1 python3 scripts/profile_module.py \
      --module mlx_harmony.chat \
      --profile-output "$PROFILE_OUTPUT" \
      -- \
      --model "$MODEL_PATH" \
      --prompt-config "$PROMPT_CONFIG" \
      --debug-file "debug.log" \
      > "$BENCH_LOG" 2>&1
  rc=$?
fi
set -e
exit "$rc"
