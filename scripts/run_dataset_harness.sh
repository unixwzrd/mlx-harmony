#!/usr/bin/env bash
set -euo pipefail

[ -L "${BASH_SOURCE[0]}" ] && THIS_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}") || THIS_SCRIPT="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$THIS_SCRIPT")" && pwd -P)"

if [[ $# -lt 1 ]]; then
  echo "Usage: run_dataset_harness.sh <cli|server|all>" >&2
  exit 2
fi

MODE="$1"
source "${SCRIPT_DIR}/dataset_run_common.sh"

HOST=${MLX_HARMONY_HOST:-127.0.0.1}
PORT=${MLX_HARMONY_PORT:-8000}
PROFILES_FILE=${MLX_HARMONY_PROFILES_FILE:-configs/profiles.example.json}
PROMPT_FILE=${INTEGRATION_PROMPTS_FILE:-tests/data/english.json}
TURN_LIMIT=${INTEGRATION_TURNS:-3}
MODEL_PATH=${MLX_HARMONY_MODEL_PATH:-}
PROFILE_NAME=${MLX_HARMONY_PROFILE:-}
PROMPT_CONFIG=${PROMPT_CONFIG:-configs/prompt-config.deterministic.json}
REQUEST_TIMEOUT=${INTEGRATION_REQUEST_TIMEOUT:-300}
HEALTH_RETRIES=${INTEGRATION_HEALTH_RETRIES:-100}
MAX_TOKENS=${INTEGRATION_MAX_TOKENS:-512}

RUN_ID=${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-runs}
META_DIR=${META_DIR:-${RUN_ROOT}/${RUN_ID}/meta/${MODE}}
METRICS_DIR=${METRICS_DIR:-${RUN_ROOT}/${RUN_ID}/metrics/${MODE}}
LOGS_DIR=${LOGS_DIR:-${RUN_ROOT}/${RUN_ID}/logs/${MODE}}
PLOTS_DIR=${PLOTS_DIR:-${RUN_ROOT}/${RUN_ID}/plots/${MODE}}
DEBUG_LOG_PATH=${DEBUG_LOG_PATH:-logs/debug.log}
CHAT_LOG_PATH=${CHAT_LOG_PATH:-logs/profiling-chat.json}
SERVER_LOG=${SERVER_LOG:-${LOGS_DIR}/server-run.log}
SERVER_DEBUG_LOG=${SERVER_DEBUG_LOG:-${LOGS_DIR}/debug.log}
SERVER_PROFILE=${SERVER_PROFILE:-0}
SERVER_BENCH_LOG=${SERVER_BENCH_LOG:-${META_DIR}/bench.time.log}
SERVER_LOG_PROMPTS=${MLX_HARMONY_SERVER_LOG_PROMPTS:-1}
PROFILE_OUTPUT=${PROFILE_OUTPUT:-${METRICS_DIR}/profile.stats}
PROFILE_TEXT=${PROFILE_TEXT:-${METRICS_DIR}/profile.stats.txt}
PROFILE_DOT=${PROFILE_DOT:-${METRICS_DIR}/profile.dot}
PROFILE_SVG=${PROFILE_SVG:-${METRICS_DIR}/profile.svg}
PROFILE_METRICS_JSON=${PROFILE_METRICS_JSON:-${METRICS_DIR}/profile.metrics.json}
PROFILE_STATIC_TXT=${PROFILE_STATIC_TXT:-${METRICS_DIR}/profile.static.txt}
PROFILE_NODE_THRES=${PROFILE_NODE_THRES:-}
PROFILE_EDGE_THRES=${PROFILE_EDGE_THRES:-}
CLIENT_PROFILE_OUTPUT=${CLIENT_PROFILE_OUTPUT:-${RUN_ROOT}/${RUN_ID}/metrics/server/client-profile.stats}
CLIENT_PROFILE_SVG=${CLIENT_PROFILE_SVG:-${RUN_ROOT}/${RUN_ID}/metrics/server/client-profile.svg}
CLIENT_PROFILE_NODE_THRES=${CLIENT_PROFILE_NODE_THRES:-}
CLIENT_PROFILE_EDGE_THRES=${CLIENT_PROFILE_EDGE_THRES:-}
PROFILE_METRICS_JSON_ABS="$(abs_path "$PROFILE_METRICS_JSON")"
PROFILE_STATIC_TXT_ABS="$(abs_path "$PROFILE_STATIC_TXT")"
REPORT_FILE=${REPORT_FILE:-}
VM_FILTER=${VM_FILTER:-scripts/filter-vm_stat.py}
VM_INTERVAL=${VM_INTERVAL:-1}
VM_FILTER_ARGS_DEFAULT=( -t -b -s )
MERGE_SCRIPT=${MERGE_SCRIPT:-scripts/merge_timing_metrics.py}
TPS_WIRED_PLOT=${TPS_WIRED_PLOT:-scripts/TPSvsWiredMemory.py}
SERVER_REQUESTS_LOG=${SERVER_REQUESTS_LOG:-${LOGS_DIR}/server-requests.log}

mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

PROFILE_OUTPUT_ABS="$(abs_path "$PROFILE_OUTPUT")"
PROFILE_SVG_ABS="$(abs_path "$PROFILE_SVG")"

process_profile_artifacts() {
  if [[ ! -s "$PROFILE_OUTPUT" ]]; then
    return
  fi

  if [[ ! -s "$PROFILE_TEXT" ]]; then
    scripts/process_profile_artifacts.py \
      --profile-output "$PROFILE_OUTPUT" \
      --profile-text "$PROFILE_TEXT" \
      --profile-metrics-json "$PROFILE_METRICS_JSON_ABS" \
      --profile-static-txt "$PROFILE_STATIC_TXT_ABS"
  fi

  if command -v gprof2dot >/dev/null 2>&1; then
    if [[ ! -s "$PROFILE_DOT" ]]; then
      gprof2dot -f pstats "$PROFILE_OUTPUT" \
        ${PROFILE_NODE_THRES:+-n "$PROFILE_NODE_THRES"} \
        ${PROFILE_EDGE_THRES:+-e "$PROFILE_EDGE_THRES"} \
        > "$PROFILE_DOT" || true
    fi
    if [[ -s "$PROFILE_DOT" && ! -s "$PROFILE_SVG" ]]; then
      if command -v dot >/dev/null 2>&1; then
        dot -Tsvg "$PROFILE_DOT" -o "$PROFILE_SVG" || true
      fi
    fi
  fi
}

write_run_env "$META_DIR" \
  "run_id=${RUN_ID}" \
  "timestamp=$(date +%Y%m%d-%H%M%S)" \
  "mode=${MODE}" \
  "dataset=${PROMPT_FILE}" \
  "model=${MODEL_PATH}" \
  "limit=${TURN_LIMIT}" \
  "prompt_config=${PROMPT_CONFIG}" \
  "metrics_dir=${METRICS_DIR}" \
  "logs_dir=${LOGS_DIR}" \
  "plots_dir=${PLOTS_DIR}" \
  "pwd=$(pwd)" \
  "uname=$(uname -a)" \
  "python=$(python --version 2>&1 || true)" \
  "git_rev=$(git rev-parse HEAD 2>/dev/null || true)" \
  "git_dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

if [[ -f "$PROMPT_CONFIG" ]]; then
  cp -a "$PROMPT_CONFIG" "$META_DIR/prompt-config.json"
fi

cleanup() {
  stop_vmstat
}
trap cleanup EXIT

start_vmstat "$VM_INTERVAL" "$VM_FILTER" "$METRICS_DIR/vm_stat-timing.tsv" "$META_DIR/vm_stat.stderr" "${VM_FILTER_ARGS_DEFAULT[@]}"

run_cli_mode() {
  MODE="cli"
  META_DIR=${RUN_ROOT}/${RUN_ID}/meta/${MODE}
  METRICS_DIR=${RUN_ROOT}/${RUN_ID}/metrics/${MODE}
  LOGS_DIR=${RUN_ROOT}/${RUN_ID}/logs/${MODE}
  PLOTS_DIR=${RUN_ROOT}/${RUN_ID}/plots/${MODE}
  DEBUG_LOG_PATH=${DEBUG_LOG_PATH:-logs/debug.log}
  CHAT_LOG_PATH=${CHAT_LOG_PATH:-logs/profiling-chat.json}
  PROFILE_OUTPUT=${PROFILE_OUTPUT:-${METRICS_DIR}/profile.stats}
  PROFILE_TEXT=${PROFILE_TEXT:-${METRICS_DIR}/profile.stats.txt}
  PROFILE_DOT=${PROFILE_DOT:-${METRICS_DIR}/profile.dot}
  PROFILE_SVG=${PROFILE_SVG:-${METRICS_DIR}/profile.svg}
  PROFILE_METRICS_JSON=${PROFILE_METRICS_JSON:-${METRICS_DIR}/profile.metrics.json}
  PROFILE_STATIC_TXT=${PROFILE_STATIC_TXT:-${METRICS_DIR}/profile.static.txt}
  PROFILE_OUTPUT_ABS="$(abs_path "$PROFILE_OUTPUT")"
  PROFILE_SVG_ABS="$(abs_path "$PROFILE_SVG")"
  PROFILE_METRICS_JSON_ABS="$(abs_path "$PROFILE_METRICS_JSON")"
  PROFILE_STATIC_TXT_ABS="$(abs_path "$PROFILE_STATIC_TXT")"
  mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

  write_run_env "$META_DIR" \
    "run_id=${RUN_ID}" \
    "timestamp=$(date +%Y%m%d-%H%M%S)" \
    "mode=${MODE}" \
    "dataset=${PROMPT_FILE}" \
    "model=${MODEL_PATH}" \
    "limit=${TURN_LIMIT}" \
    "prompt_config=${PROMPT_CONFIG}" \
    "metrics_dir=${METRICS_DIR}"

  if [[ -f "scripts/clean_run_artifacts.sh" ]]; then
    bash scripts/clean_run_artifacts.sh "logs"
  fi
  if [[ "${BENCH_TTY:-0}" == "1" ]]; then
    DEBUG_LOG_PATH="$DEBUG_LOG_PATH" CHAT_LOG_PATH="$CHAT_LOG_PATH" VM_STAT_OUT="" \
      PROFILE_OUTPUT="$PROFILE_OUTPUT_ABS" GRAPH_OUTPUT="$PROFILE_SVG_ABS" \
      script -q "${META_DIR}/bench.time.log" bash -c "time -p bash tmp/deterministic-test.sh"
  elif [[ "${BENCH_TEE:-0}" == "1" ]]; then
    (
      DEBUG_LOG_PATH="$DEBUG_LOG_PATH" CHAT_LOG_PATH="$CHAT_LOG_PATH" VM_STAT_OUT="" \
        PROFILE_OUTPUT="$PROFILE_OUTPUT_ABS" GRAPH_OUTPUT="$PROFILE_SVG_ABS" \
        time -p bash tmp/deterministic-test.sh
    ) 2>&1 | tee "${META_DIR}/bench.time.log"
  else
    (
      DEBUG_LOG_PATH="$DEBUG_LOG_PATH" CHAT_LOG_PATH="$CHAT_LOG_PATH" VM_STAT_OUT="" \
        PROFILE_OUTPUT="$PROFILE_OUTPUT_ABS" GRAPH_OUTPUT="$PROFILE_SVG_ABS" \
        time -p bash tmp/deterministic-test.sh
    ) > "${META_DIR}/bench.time.log" 2>&1
  fi
  if [[ -d "logs" ]]; then
    find logs -maxdepth 1 -type f \( \
      -name 'completion.*' -o \
      -name 'parse.*' -o \
      -name 'prompt.*' -o \
      -name 'retry.*' -o \
      -name 'profiling-chat.json' -o \
      -name 'debug.log' \
    \) -exec mv -f {} "${LOGS_DIR}/" \; || true
    if [[ -f "${LOGS_DIR}/debug.log" ]]; then
      DEBUG_LOG_PATH="${LOGS_DIR}/debug.log"
    fi
  fi
  stop_vmstat
  write_timings_tsv "$DEBUG_LOG_PATH" "$METRICS_DIR/timings-debug.tsv"
  if [[ -f "$DEBUG_LOG_PATH" ]]; then
    cp -a "$DEBUG_LOG_PATH" "$METRICS_DIR/debug.log" || true
  fi
  merge_timings "$MERGE_SCRIPT" \
    "$METRICS_DIR/timings-debug.tsv" \
    "$METRICS_DIR/vm_stat-timing.tsv" \
    "$METRICS_DIR/merged-timings-vm_stat.tsv"
  plot_tps_vs_wired "$TPS_WIRED_PLOT" \
    "$METRICS_DIR/merged-timings-vm_stat.tsv" \
    "$PLOTS_DIR/tps_vs_wired.png"
  process_profile_artifacts
}

run_server_mode() {
  MODE="server"
  META_DIR=${RUN_ROOT}/${RUN_ID}/meta/${MODE}
  METRICS_DIR=${RUN_ROOT}/${RUN_ID}/metrics/${MODE}
  LOGS_DIR=${RUN_ROOT}/${RUN_ID}/logs/${MODE}
  PLOTS_DIR=${RUN_ROOT}/${RUN_ID}/plots/${MODE}
  SERVER_LOG=${SERVER_LOG:-${LOGS_DIR}/server-run.log}
  SERVER_DEBUG_LOG=${SERVER_DEBUG_LOG:-${LOGS_DIR}/debug.log}
  SERVER_BENCH_LOG=${SERVER_BENCH_LOG:-${META_DIR}/bench.time.log}
  PROFILE_OUTPUT=${PROFILE_OUTPUT:-${METRICS_DIR}/profile.stats}
  PROFILE_TEXT=${PROFILE_TEXT:-${METRICS_DIR}/profile.stats.txt}
  PROFILE_DOT=${PROFILE_DOT:-${METRICS_DIR}/profile.dot}
  PROFILE_SVG=${PROFILE_SVG:-${METRICS_DIR}/profile.svg}
  PROFILE_METRICS_JSON=${PROFILE_METRICS_JSON:-${METRICS_DIR}/profile.metrics.json}
  PROFILE_STATIC_TXT=${PROFILE_STATIC_TXT:-${METRICS_DIR}/profile.static.txt}
  PROFILE_METRICS_JSON_ABS="$(abs_path "$PROFILE_METRICS_JSON")"
  PROFILE_STATIC_TXT_ABS="$(abs_path "$PROFILE_STATIC_TXT")"
  mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

  write_run_env "$META_DIR" \
    "run_id=${RUN_ID}" \
    "timestamp=$(date +%Y%m%d-%H%M%S)" \
    "mode=${MODE}" \
    "dataset=${PROMPT_FILE}" \
    "model=${MODEL_PATH}" \
    "limit=${TURN_LIMIT}" \
    "prompt_config=${PROMPT_CONFIG}" \
    "metrics_dir=${METRICS_DIR}"

  SERVER_DEBUG_LOG_ABS="$(abs_path "$SERVER_DEBUG_LOG")"
  if [[ "$SERVER_PROFILE" == "1" ]]; then
    MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG_ABS" \
      MLX_HARMONY_SERVER_COLLECT_MEMORY=1 \
      MLX_HARMONY_SERVER_LOG_PROMPTS="$SERVER_LOG_PROMPTS" \
      python -m cProfile -o "$PROFILE_OUTPUT" -m mlx_harmony.server \
      --host "$HOST" \
      --port "$PORT" \
      --profiles-file "$PROFILES_FILE" \
      >"$SERVER_LOG" 2>&1 &
  else
    MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG_ABS" \
      MLX_HARMONY_SERVER_COLLECT_MEMORY=1 \
      MLX_HARMONY_SERVER_LOG_PROMPTS="$SERVER_LOG_PROMPTS" \
      python -m mlx_harmony.server \
      --host "$HOST" \
      --port "$PORT" \
      --profiles-file "$PROFILES_FILE" \
      >"$SERVER_LOG" 2>&1 &
  fi
  SERVER_PID=$!
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

  HEALTH_OK=0
  for _ in $(seq 1 "$HEALTH_RETRIES"); do
    if curl -fsS "http://${HOST}:${PORT}/v1/health" >/dev/null 2>&1; then
      HEALTH_OK=1
      break
    fi
    sleep 0.2
  done
  if [[ "$HEALTH_OK" -ne 1 ]]; then
    echo "Health check failed for http://${HOST}:${PORT}/v1/health" >&2
    exit 1
  fi

  (
    if [[ -z "$MODEL_PATH" && -z "$PROFILE_NAME" && -f "$PROFILES_FILE" ]]; then
      PROFILE_NAME="$(python -c "import json; data=json.load(open('${PROFILES_FILE}', encoding='utf-8')); print(next(iter(data.keys()), ''))")"
    fi
    if [[ -z "$MODEL_PATH" && -z "$PROFILE_NAME" ]]; then
      echo "Set MLX_HARMONY_MODEL_PATH or MLX_HARMONY_PROFILE to run the server dataset client." >&2
      exit 1
    fi
    scripts/build_prompt_stream.py "$PROMPT_FILE" "$TURN_LIMIT" | \
      scripts/profile_client.py \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --profile "$PROFILE_NAME" \
        --prompt-config "$PROMPT_CONFIG" \
        --max-tokens "$MAX_TOKENS" \
        --timeout "$REQUEST_TIMEOUT" \
        --health-retries "$HEALTH_RETRIES" \
        --health-sleep 0.2 \
        --report-file "${REPORT_FILE}" \
        --requests-log "${SERVER_REQUESTS_LOG}" \
        --profile-output "${CLIENT_PROFILE_OUTPUT}" \
        --graph "${CLIENT_PROFILE_SVG}" \
        ${CLIENT_PROFILE_NODE_THRES:+--node-thres "$CLIENT_PROFILE_NODE_THRES"} \
        ${CLIENT_PROFILE_EDGE_THRES:+--edge-thres "$CLIENT_PROFILE_EDGE_THRES"}
  ) > "${SERVER_BENCH_LOG}" 2>&1

  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -f "$SERVER_LOG" ]]; then
    server_log_abs="$(python -c "import os; print(os.path.abspath('${SERVER_LOG}'))")"
    server_log_dest="$(python -c "import os; print(os.path.abspath('${LOGS_DIR}/$(basename "${SERVER_LOG}")'))")"
    if [[ "$server_log_abs" != "$server_log_dest" ]]; then
      mv -f "$SERVER_LOG" "$LOGS_DIR/" || true
    fi
  fi
  if [[ -f "$SERVER_DEBUG_LOG" ]]; then
    cp -a "$SERVER_DEBUG_LOG" "$METRICS_DIR/debug.log" || true
  fi
  if [[ -d "logs" ]]; then
    find logs -maxdepth 1 -type f \( \
      -name 'completion.*' -o \
      -name 'parse.*' -o \
      -name 'prompt.*' -o \
      -name 'retry.*' -o \
      -name 'profiling-chat.json' -o \
      -name 'debug.log' \
    \) -exec mv -f {} "${LOGS_DIR}/" \; || true
  fi
  if [[ -d "logs/server" ]]; then
    find logs/server -maxdepth 1 -type f -exec mv -f {} "${LOGS_DIR}/" \; || true
  fi
  stop_vmstat
  stop_vmstat
  write_timings_tsv "$SERVER_DEBUG_LOG_ABS" "$METRICS_DIR/timings-debug.tsv"
  merge_timings "$MERGE_SCRIPT" \
    "$METRICS_DIR/timings-debug.tsv" \
    "$METRICS_DIR/vm_stat-timing.tsv" \
    "$METRICS_DIR/merged-timings-vm_stat.tsv"

  plot_tps_vs_wired "$TPS_WIRED_PLOT" \
    "$METRICS_DIR/merged-timings-vm_stat.tsv" \
    "$PLOTS_DIR/tps_vs_wired.png"

  process_profile_artifacts

  for f in \
    "timings-debug.tsv" \
    "vm_stat-timing.tsv" \
    "merged-timings-vm_stat.tsv" \
    "debug.log" \
    "profile.stats" \
    "profile.stats.txt" \
    "profile.static.txt" \
    "profile.dot" \
    "profile.svg" \
    "profile.metrics.json"; do
    if [[ ! -f "$METRICS_DIR/$f" ]]; then
      : > "$METRICS_DIR/$f"
    fi
  done
  if [[ ! -f "$PLOTS_DIR/tps_vs_wired.png" ]]; then
    : > "$PLOTS_DIR/tps_vs_wired.png"
  fi
}

case "$MODE" in
  cli)
    run_cli_mode
    ;;
  server)
    run_server_mode
    ;;
  all)
    run_cli_mode
    run_server_mode
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
