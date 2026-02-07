#!/usr/bin/env bash
set -Eeuo pipefail

# -------- bootstrap --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/dataset_run_common.sh"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -ge 1 ]] || die "Usage: $0 <cli|server|all>"
TOP_MODE="$1"

# -------- inputs --------
HOST="${MLX_HARMONY_HOST:-127.0.0.1}"
PORT="${MLX_HARMONY_PORT:-8000}"
PROFILES_FILE="${MLX_HARMONY_PROFILES_FILE:-configs/profiles.example.json}"
PROMPT_FILE="${INTEGRATION_PROMPTS_FILE:-tests/data/english.json}"
TURN_LIMIT="${INTEGRATION_TURNS:-3}"
MODEL_PATH="${MLX_HARMONY_MODEL_PATH:-}"
PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompt-config.deterministic.json}"
MAX_TOKENS="${INTEGRATION_MAX_TOKENS:-}"

RUN_ID="${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs}"

VM_FILTER="${VM_FILTER:-scripts/filter-vm_stat.py}"
VM_INTERVAL="${VM_INTERVAL:-1}"
MERGE_SCRIPT="${MERGE_SCRIPT:-scripts/merge_timing_metrics.py}"
TPS_WIRED_PLOT="${TPS_WIRED_PLOT:-scripts/TPSvsWiredMemory.py}"
VM_FILTER_ARGS_DEFAULT=( -t -b -s )

init_mode_paths() {
  local mode="$1"
  META_DIR="${RUN_ROOT}/${RUN_ID}/meta/${mode}"
  METRICS_DIR="${RUN_ROOT}/${RUN_ID}/metrics/${mode}"
  LOGS_DIR="${RUN_ROOT}/${RUN_ID}/logs/${mode}"
  PLOTS_DIR="${RUN_ROOT}/${RUN_ID}/plots/${mode}"
  mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

  PROFILE_OUTPUT="${METRICS_DIR}/profile.stats"
  PROFILE_SVG="${METRICS_DIR}/profile.svg"
  CLIENT_PROFILE_OUTPUT="${METRICS_DIR}/client-profile.stats"
  VMSTAT_OUT="${METRICS_DIR}/vm_stat-timing.tsv"
  VMSTAT_ERR="${META_DIR}/vm_stat.stderr"
}

write_metadata() {
  local mode="$1"
  write_run_env "$META_DIR" \
    "run_id=${RUN_ID}" \
    "timestamp=$(date +%Y%m%d-%H%M%S)" \
    "mode=${mode}" \
    "dataset=${PROMPT_FILE}" \
    "model=${MODEL_PATH}" \
    "limit=${TURN_LIMIT}" \
    "prompt_config=${PROMPT_CONFIG}" \
    "metrics_dir=${METRICS_DIR}" \
    "logs_dir=${LOGS_DIR}" \
    "plots_dir=${PLOTS_DIR}"
  [[ -f "$PROMPT_CONFIG" ]] && cp -a "$PROMPT_CONFIG" "$META_DIR/prompt-config.json"
}

run_cli_mode() {
  local bench_log="${META_DIR}/bench.time.log"
  local debug_log="logs/debug.log"
  local chat_log="logs/profiling-chat.json"

  scripts/clean_logs.sh "logs" || true
  start_vmstat "$VM_INTERVAL" "$VM_FILTER" "$VMSTAT_OUT" "$VMSTAT_ERR" "${VM_FILTER_ARGS_DEFAULT[@]}"

  cli_args=(
    --mode cli
    --model "$MODEL_PATH"
    --prompt-config "$PROMPT_CONFIG"
    --dataset "$PROMPT_FILE"
    --turns "$TURN_LIMIT"
    --profile-out "$PROFILE_OUTPUT"
    --bench-log "$bench_log"
  )
  if [[ -n "${MAX_TOKENS}" ]]; then
    cli_args+=( --max-tokens "$MAX_TOKENS" )
  fi
  scripts/run_profile.sh "${cli_args[@]}"

  stop_vmstat || true
  scripts/preserve_logs.sh "logs" "$LOGS_DIR"

  if [[ -f "${LOGS_DIR}/debug.log" ]]; then
    debug_log="${LOGS_DIR}/debug.log"
    cp -a "$debug_log" "$METRICS_DIR/debug.log" || true
  fi

  scripts/process_stats.sh "$debug_log" "$METRICS_DIR" "$PLOTS_DIR" "$VMSTAT_OUT" \
    "$MERGE_SCRIPT" "$TPS_WIRED_PLOT" "$PROFILE_OUTPUT" "$PROFILE_SVG"
}

run_server_mode() {
  local bench_log="${META_DIR}/bench.time.log"
  local server_log="${LOGS_DIR}/server-run.log"
  local server_debug_log="${LOGS_DIR}/debug.log"
  local server_requests_log="${LOGS_DIR}/server-requests.log"

  scripts/clean_logs.sh "logs" || true
  start_vmstat "$VM_INTERVAL" "$VM_FILTER" "$VMSTAT_OUT" "$VMSTAT_ERR" "${VM_FILTER_ARGS_DEFAULT[@]}"

  server_args=(
    --mode server
    --model "$MODEL_PATH"
    --prompt-config "$PROMPT_CONFIG"
    --profiles-file "$PROFILES_FILE"
    --dataset "$PROMPT_FILE"
    --turns "$TURN_LIMIT"
    --host "$HOST"
    --port "$PORT"
    --server-profile-out "$PROFILE_OUTPUT"
    --client-profile-out "$CLIENT_PROFILE_OUTPUT"
    --server-log "$server_log"
    --server-debug-log "$server_debug_log"
    --server-requests-log "$server_requests_log"
    --bench-log "$bench_log"
  )
  if [[ -n "${MAX_TOKENS}" ]]; then
    server_args+=( --max-tokens "$MAX_TOKENS" )
  fi
  scripts/run_profile.sh "${server_args[@]}"

  stop_vmstat || true
  scripts/preserve_logs.sh "logs" "$LOGS_DIR"

  if [[ -f "${LOGS_DIR}/debug.log" ]]; then
    cp -a "${LOGS_DIR}/debug.log" "$METRICS_DIR/debug.log" || true
  fi

  scripts/process_stats.sh "$server_debug_log" "$METRICS_DIR" "$PLOTS_DIR" "$VMSTAT_OUT" \
    "$MERGE_SCRIPT" "$TPS_WIRED_PLOT" "$PROFILE_OUTPUT" "$PROFILE_SVG"

  if [[ -f "$CLIENT_PROFILE_OUTPUT" ]]; then
    scripts/process_stats.sh "$server_debug_log" "$METRICS_DIR" "$PLOTS_DIR" "$VMSTAT_OUT" \
      "$MERGE_SCRIPT" "$TPS_WIRED_PLOT" "$CLIENT_PROFILE_OUTPUT" "${METRICS_DIR}/client-profile.svg"
  fi
}

run_one_mode() {
  local mode="$1"
  init_mode_paths "$mode"
  write_metadata "$mode"
  cleanup_vmstat_processes "logs/vm_stat-timing.tsv"
  cleanup_vmstat_processes "$VMSTAT_OUT"
  case "$mode" in
    cli) run_cli_mode ;;
    server) run_server_mode ;;
    *) die "Unknown mode: $mode" ;;
  esac
}

case "$TOP_MODE" in
  cli) run_one_mode cli ;;
  server) run_one_mode server ;;
  all) run_one_mode cli; run_one_mode server ;;
  *) die "Unknown mode: $TOP_MODE" ;;
esac
