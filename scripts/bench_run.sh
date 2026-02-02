#!/usr/bin/env bash
# scripts/bench_run.sh
set -euo pipefail
unset TIMEFORMAT

# -------- Defaults (override via env or CLI args) --------
DATASET_DEFAULT="tests/data/english.json"
MODEL_DEFAULT="models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx"
LIMIT_DEFAULT="20"
PROMPT_CONFIG_DEFAULT="configs/prompt-config.deterministic.json"

RUNS_ROOT_DEFAULT="runs"
QUANT_LABEL_DEFAULT="q8"
RUN_LABEL_DEFAULT="20prompt"

# -------- CLI (positional) --------
MODE_ARG=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    cli|server|all)
      MODE_ARG="$1"
      shift
      ;;
  esac
fi

DATASET="${1:-$DATASET_DEFAULT}"
MODEL_PATH="${2:-$MODEL_DEFAULT}"
LIMIT="${3:-$LIMIT_DEFAULT}"

# -------- Env overrides --------
RUNS_ROOT="${RUNS_ROOT:-$RUNS_ROOT_DEFAULT}"
PROMPT_CONFIG="${PROMPT_CONFIG:-$PROMPT_CONFIG_DEFAULT}"
QUANT_LABEL="${QUANT_LABEL:-$QUANT_LABEL_DEFAULT}"
RUN_LABEL="${RUN_LABEL:-$RUN_LABEL_DEFAULT}"
INTEGRATION_PROFILE="${INTEGRATION_PROFILE:-1}"
RUN_SERVER="${RUN_SERVER:-0}"
RUN_MODE="${RUN_MODE:-}"
if [[ -n "$MODE_ARG" ]]; then
  RUN_MODE="$MODE_ARG"
fi

# -------- Resolve repo root --------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# -------- Run directory --------
TS="$(date +%Y%m%d-%H%M)"
RUN_ID="save-${TS}-${QUANT_LABEL}-${RUN_LABEL}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"

META_DIR="${RUN_DIR}/meta"
LOGS_DIR="${RUN_DIR}/logs"
METRICS_DIR="${RUN_DIR}/metrics"
ANALYSIS_DIR="${RUN_DIR}/analysis"
PLOTS_DIR="${RUN_DIR}/plots"
CLI_META_DIR="${META_DIR}/cli"
SERVER_META_DIR="${META_DIR}/server"
CLI_LOGS_DIR="${LOGS_DIR}/cli"
SERVER_LOGS_DIR="${LOGS_DIR}/server"
CLI_METRICS_DIR="${METRICS_DIR}/cli"
SERVER_METRICS_DIR="${METRICS_DIR}/server"
CLI_PLOTS_DIR="${PLOTS_DIR}/cli"
SERVER_PLOTS_DIR="${PLOTS_DIR}/server"

mkdir -p "${META_DIR}" "${LOGS_DIR}" "${METRICS_DIR}" "${ANALYSIS_DIR}" "${PLOTS_DIR}" \
  "${CLI_META_DIR}" "${SERVER_META_DIR}" "${CLI_LOGS_DIR}" "${SERVER_LOGS_DIR}" \
  "${CLI_METRICS_DIR}" "${SERVER_METRICS_DIR}" "${CLI_PLOTS_DIR}" "${SERVER_PLOTS_DIR}"

log() { printf '%s\n' "$*" >&2; }

# -------- CLI harness --------
if [[ -z "${RUN_MODE}" || "${RUN_MODE}" == "cli" || "${RUN_MODE}" == "all" ]]; then
  log "running CLI dataset harness..."
  RUN_ID="${RUN_ID}" RUN_ROOT="${RUNS_ROOT}" META_DIR="${CLI_META_DIR}" METRICS_DIR="${CLI_METRICS_DIR}" \
    LOGS_DIR="${CLI_LOGS_DIR}" PLOTS_DIR="${CLI_PLOTS_DIR}" \
    DEBUG_LOG_PATH="logs/debug.log" CHAT_LOG_PATH="logs/profiling-chat.json" \
    PROMPT_CONFIG="${PROMPT_CONFIG}" INTEGRATION_PROMPTS_FILE="${DATASET}" INTEGRATION_TURNS="${LIMIT}" \
    MLX_HARMONY_MODEL_PATH="${MODEL_PATH}" \
    scripts/run_dataset_harness.sh cli
fi

# -------- Collect analysis artifacts --------
if command -v function-analysis >/dev/null 2>&1; then
  function-analysis src > "${ANALYSIS_DIR}/function-analysis.txt" || true
fi
find src -name "*.py" -print0 | xargs -0 wc -l | sort -rn > "${ANALYSIS_DIR}/LOC-stats.txt" || true

# -------- Server harness --------
if [[ -z "${RUN_MODE}" ]]; then
  SERVER_OK="${RUN_SERVER}"
else
  if [[ "${RUN_MODE}" == "server" || "${RUN_MODE}" == "all" ]]; then
    SERVER_OK="1"
  else
    SERVER_OK="0"
  fi
fi

if [[ "${SERVER_OK}" == "1" ]]; then
  log "running server dataset harness..."
  RUN_ID="${RUN_ID}" RUN_ROOT="${RUNS_ROOT}" META_DIR="${SERVER_META_DIR}" METRICS_DIR="${SERVER_METRICS_DIR}" \
    LOGS_DIR="${SERVER_LOGS_DIR}" PLOTS_DIR="${SERVER_PLOTS_DIR}" \
    PROMPT_CONFIG="${PROMPT_CONFIG}" INTEGRATION_PROMPTS_FILE="${DATASET}" INTEGRATION_TURNS="${LIMIT}" \
    MLX_HARMONY_MODEL_PATH="${MODEL_PATH}" SERVER_PROFILE=1 \
    REPORT_FILE="${SERVER_META_DIR}/server-dataset-report.json" \
    scripts/run_dataset_harness.sh server
fi

log "run complete -> ${RUN_DIR}"
