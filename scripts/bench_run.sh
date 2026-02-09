#!/usr/bin/env bash
set -Eeuo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

RUN_MODE="${1:-all}"
case "${RUN_MODE}" in
  cli|server|all) shift ;;
  *) echo "ERROR: Unknown run mode: ${RUN_MODE}" >&2; exit 1 ;;
esac

DATASET="${1:-tests/data/english.json}"
MODEL_PATH="${2:-models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx}"
PROMPT_CONFIG="${3:-configs/prompt-config.deterministic.json}"
TURN_LIMIT="${4:-20}"
RUN_BASE="${RUN_BASE:-runs}"
RUN_LABEL="${RUN_LABEL:-q8-20prompt}"
RUN_ID="${RUN_ID:-save-$(date +%Y%m%d-%H%M%S)-${RUN_LABEL}}"
RUN_DIR="${RUN_BASE}/${RUN_ID}"

META_DIR="${RUN_DIR}/meta"
LOGS_DIR="${RUN_DIR}/logs"
METRICS_DIR="${RUN_DIR}/metrics"
PLOTS_DIR="${RUN_DIR}/plots"
ANALYSIS_DIR="${RUN_DIR}/analysis"
mkdir -p "$META_DIR" "$LOGS_DIR" "$METRICS_DIR" "$PLOTS_DIR" "$ANALYSIS_DIR"

log() { printf '%s\n' "$*" >&2; }

run_component() {
  local mode="$1"
  log "running ${mode} dataset harness..."
  scripts/run_dataset_harness.sh \
    --mode "$mode" \
    --run-id "$RUN_ID" \
    --run-root "$RUN_BASE" \
    --dataset "$DATASET" \
    --model "$MODEL_PATH" \
    --prompt-config "$PROMPT_CONFIG" \
    --turns "$TURN_LIMIT"
}

collect_analysis() {
  if command -v function-analysis >/dev/null 2>&1; then
    function-analysis src > "${ANALYSIS_DIR}/function-analysis.txt" || true
  fi
  find src -name "*.py" -print0 | xargs -0 wc -l | sort -rn > "${ANALYSIS_DIR}/LOC-stats.txt" || true
}

case "${RUN_MODE}" in
  cli)
    run_component "cli"
    ;;
  server)
    run_component "server"
    ;;
  all)
    run_component "cli"
    run_component "server"
    ;;
esac

collect_analysis
log "run complete -> ${RUN_DIR}"
