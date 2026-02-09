#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${BASE_DIR}"
source "${SCRIPT_DIR}/dataset_run_common.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

MODE=""
RUN_ID=""
RUN_ROOT=""
DATASET=""
MODEL_PATH=""
PROMPT_CONFIG=""
TURN_LIMIT=""
HOST="${MLX_HARMONY_HOST:-127.0.0.1}"
PORT="${MLX_HARMONY_PORT:-8000}"
PROFILES_FILE="${MLX_HARMONY_PROFILES_FILE:-configs/profiles.example.json}"
VM_FILTER="${VM_FILTER:-scripts/filter-vm_stat.py}"
VM_INTERVAL="${VM_INTERVAL:-1}"
MERGE_SCRIPT="${MERGE_SCRIPT:-scripts/merge_timing_metrics.py}"
TPS_WIRED_PLOT="${TPS_WIRED_PLOT:-scripts/TPSvsWiredMemory.py}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --prompt-config) PROMPT_CONFIG="$2"; shift 2 ;;
    --turns) TURN_LIMIT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --profiles-file) PROFILES_FILE="$2"; shift 2 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$MODE" && -n "$RUN_ID" && -n "$RUN_ROOT" && -n "$DATASET" && -n "$MODEL_PATH" && -n "$PROMPT_CONFIG" && -n "$TURN_LIMIT" ]] \
  || die "Required args missing"
case "$MODE" in
  cli|server) ;;
  *) die "Unknown mode: $MODE" ;;
esac

RUN_ROOT="$(abs_path "${RUN_ROOT}")"
LOG_SOURCE_DIR="${BASE_DIR}/logs"
META_DIR="${RUN_ROOT}/${RUN_ID}/meta/${MODE}"
METRICS_DIR="${RUN_ROOT}/${RUN_ID}/metrics/${MODE}"
LOGS_DIR="${RUN_ROOT}/${RUN_ID}/logs/${MODE}"
PLOTS_DIR="${RUN_ROOT}/${RUN_ID}/plots/${MODE}"
mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

PROFILE_OUTPUT="${METRICS_DIR}/profile.stats"
PROFILE_SVG="${METRICS_DIR}/profile.svg"
VMSTAT_OUT="${METRICS_DIR}/vm_stat-timing.tsv"
VMSTAT_ERR="${META_DIR}/vm_stat.stderr"
BENCH_LOG="${META_DIR}/bench.time.log"
FINALIZED=0

finalize_run() {
  if [[ "$FINALIZED" == "1" ]]; then
    return
  fi
  FINALIZED=1
  stop_vmstat || true
  scripts/preserve_logs.sh "$LOG_SOURCE_DIR" "$LOGS_DIR" || true
  if [[ -f "${LOGS_DIR}/debug.log" ]]; then
    cp -a "${LOGS_DIR}/debug.log" "${METRICS_DIR}/debug.log" || true
  fi
}

trap finalize_run EXIT

write_run_env "$META_DIR" \
  "run_id=${RUN_ID}" \
  "timestamp=$(date +%Y%m%d-%H%M%S)" \
  "mode=${MODE}" \
  "dataset=${DATASET}" \
  "model=${MODEL_PATH}" \
  "limit=${TURN_LIMIT}" \
  "prompt_config=${PROMPT_CONFIG}" \
  "metrics_dir=${METRICS_DIR}" \
  "logs_dir=${LOGS_DIR}" \
  "plots_dir=${PLOTS_DIR}"
[[ -f "$PROMPT_CONFIG" ]] && cp -a "$PROMPT_CONFIG" "$META_DIR/prompt-config.json"

scripts/clean_logs.sh "$LOG_SOURCE_DIR" || true
cleanup_vmstat_processes "logs/vm_stat-timing.tsv"
cleanup_vmstat_processes "$VMSTAT_OUT"

start_vmstat "$VM_INTERVAL" "$VM_FILTER" "$VMSTAT_OUT" "$VMSTAT_ERR" -t -b -s
if [[ "$MODE" == "cli" ]]; then
  scripts/profile_cli.sh \
    --model "$MODEL_PATH" \
    --prompt-config "$PROMPT_CONFIG" \
    --dataset "$DATASET" \
    --turns "$TURN_LIMIT" \
    --profile-output "$PROFILE_OUTPUT" \
    --bench-log "$BENCH_LOG"
else
  CLIENT_PROFILE_OUTPUT="${METRICS_DIR}/client-profile.stats"
  # Server-side logs are staged in ./logs and moved by preserve_logs.sh.
  SERVER_LOG="logs/server-run.log"
  SERVER_DEBUG_LOG="logs/debug.log"
  SERVER_REQUESTS_LOG="logs/server-requests.log"
  REPORT_FILE="${META_DIR}/server-dataset-report.json"
  scripts/profile_server.sh \
    --model "$MODEL_PATH" \
    --prompt-config "$PROMPT_CONFIG" \
    --profiles-file "$PROFILES_FILE" \
    --dataset "$DATASET" \
    --turns "$TURN_LIMIT" \
    --host "$HOST" \
    --port "$PORT" \
    --server-profile-output "$PROFILE_OUTPUT" \
    --client-profile-output "$CLIENT_PROFILE_OUTPUT" \
    --server-log "$SERVER_LOG" \
    --server-debug-log "$SERVER_DEBUG_LOG" \
    --server-requests-log "$SERVER_REQUESTS_LOG" \
    --report-file "$REPORT_FILE" \
    --bench-log "$BENCH_LOG"
fi
finalize_run

scripts/process_stats.sh \
  "${LOGS_DIR}/debug.log" \
  "$METRICS_DIR" \
  "$PLOTS_DIR" \
  "$VMSTAT_OUT" \
  "$MERGE_SCRIPT" \
  "$TPS_WIRED_PLOT" \
  "$PROFILE_OUTPUT" \
  "$PROFILE_SVG"

if [[ "$MODE" == "server" && -f "${METRICS_DIR}/client-profile.stats" ]]; then
  scripts/process_profile_artifacts.py \
    --profile-output "${METRICS_DIR}/client-profile.stats" \
    --profile-text "${METRICS_DIR}/client-profile.stats.txt" \
    --profile-metrics-json "$(abs_path "${METRICS_DIR}/client-profile.metrics.json")" \
    --profile-static-txt "$(abs_path "${METRICS_DIR}/client-profile.static.txt")" \
    --profile-dot "${METRICS_DIR}/client-profile.dot" \
    --profile-svg "${METRICS_DIR}/client-profile.svg" \
    --top 50 || true
fi
