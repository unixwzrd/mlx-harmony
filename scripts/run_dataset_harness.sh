#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: run_dataset_harness.sh <cli|server>" >&2
  exit 2
fi

MODE="$1"
source scripts/dataset_run_common.sh

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
SERVER_DEBUG_LOG=${SERVER_DEBUG_LOG:-${METRICS_DIR}/server-debug.log}
SERVER_PROFILE=${SERVER_PROFILE:-0}
SERVER_BENCH_LOG=${SERVER_BENCH_LOG:-${META_DIR}/bench.time.log}
SERVER_LOG_PROMPTS=${MLX_HARMONY_SERVER_LOG_PROMPTS:-1}
PROFILE_OUTPUT=${PROFILE_OUTPUT:-${METRICS_DIR}/profile.stats}
PROFILE_TEXT=${PROFILE_TEXT:-${METRICS_DIR}/profile.stats.txt}
PROFILE_DOT=${PROFILE_DOT:-${METRICS_DIR}/profile.dot}
PROFILE_SVG=${PROFILE_SVG:-${METRICS_DIR}/profile.svg}
PROFILE_METRICS_JSON=${PROFILE_METRICS_JSON:-${METRICS_DIR}/profile.metrics.json}
PROFILE_STATIC_TXT=${PROFILE_STATIC_TXT:-${METRICS_DIR}/profile.static.txt}
PROFILE_METRICS_JSON_ABS="$(PROFILE_METRICS_JSON="${PROFILE_METRICS_JSON}" python -c "import os; print(os.path.abspath(os.environ['PROFILE_METRICS_JSON']))")"
PROFILE_STATIC_TXT_ABS="$(PROFILE_STATIC_TXT="${PROFILE_STATIC_TXT}" python -c "import os; print(os.path.abspath(os.environ['PROFILE_STATIC_TXT']))")"
REPORT_FILE=${REPORT_FILE:-}
VM_FILTER=${VM_FILTER:-scripts/filter-vm_stat.py}
VM_INTERVAL=${VM_INTERVAL:-1}
VM_FILTER_ARGS_DEFAULT=( -t -b -s )
MERGE_SCRIPT=${MERGE_SCRIPT:-scripts/merge_timing_metrics.py}
TPS_WIRED_PLOT=${TPS_WIRED_PLOT:-scripts/TPSvsWiredMemory.py}
SERVER_REQUESTS_LOG=${SERVER_REQUESTS_LOG:-${LOGS_DIR}/server-requests.log}

mkdir -p "$META_DIR" "$METRICS_DIR" "$LOGS_DIR" "$PLOTS_DIR"

PROFILE_OUTPUT_ABS="$(PROFILE_OUTPUT="${PROFILE_OUTPUT}" python -c "import os; print(os.path.abspath(os.environ['PROFILE_OUTPUT']))")"
PROFILE_SVG_ABS="$(PROFILE_SVG="${PROFILE_SVG}" python -c "import os; print(os.path.abspath(os.environ['PROFILE_SVG']))")"

process_profile_artifacts() {
  if [[ ! -s "$PROFILE_OUTPUT" ]]; then
    return
  fi

  if [[ ! -f "$PROFILE_TEXT" ]]; then
    PROFILE_OUTPUT="$PROFILE_OUTPUT" PROFILE_TEXT="$PROFILE_TEXT" PROFILE_METRICS_JSON="$PROFILE_METRICS_JSON_ABS" PROFILE_STATIC_TXT="$PROFILE_STATIC_TXT_ABS" python - <<'PY'
import os
import json
from pathlib import Path
import pstats

stats_path = Path(os.environ["PROFILE_OUTPUT"])
report_path = Path(os.environ["PROFILE_TEXT"])
metrics_path = Path(os.environ["PROFILE_METRICS_JSON"])
static_path = Path(os.environ["PROFILE_STATIC_TXT"])
try:
    stats = pstats.Stats(str(stats_path))
except Exception as exc:  # noqa: BLE001
    print(f"[WARNING] Failed to read pstats: {stats_path} ({exc})", flush=True)
else:
    stats.sort_stats("cumulative")
    with report_path.open("w", encoding="utf-8") as out:
        stats.stream = out
        stats.print_stats(50)
    metrics = {}
    static_txt = ""
    try:
        import scripts.profile_chat as profile_chat  # type: ignore
        metrics = profile_chat.derive_runtime_metrics(stats_path)
        static_txt, static_structured = profile_chat.build_static_reports(Path("src"))
        if isinstance(metrics, dict):
            metrics["static"] = static_structured
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to derive metrics/static from stats: {exc}", flush=True)
    if metrics_path and metrics:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if static_path and static_txt:
        static_path.write_text(static_txt, encoding="utf-8")
PY
  fi

  if command -v gprof2dot >/dev/null 2>&1; then
    if [[ ! -f "$PROFILE_DOT" ]]; then
      gprof2dot -f pstats "$PROFILE_OUTPUT" > "$PROFILE_DOT" || true
    fi
    if [[ -f "$PROFILE_DOT" && ! -f "$PROFILE_SVG" ]]; then
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

if [[ "$MODE" == "cli" ]]; then
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
elif [[ "$MODE" == "server" ]]; then
  if [[ "$SERVER_PROFILE" == "1" ]]; then
    MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG" \
      MLX_HARMONY_SERVER_COLLECT_MEMORY=1 \
      MLX_HARMONY_SERVER_LOG_PROMPTS="$SERVER_LOG_PROMPTS" \
      python -m cProfile -o "$PROFILE_OUTPUT" -m mlx_harmony.server \
      --host "$HOST" \
      --port "$PORT" \
      --profiles-file "$PROFILES_FILE" \
      >"$SERVER_LOG" 2>&1 &
  else
    MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG" \
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
      scripts/profile_server_client.py \
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
        --requests-log "${SERVER_REQUESTS_LOG}"
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
    cp -a "$SERVER_DEBUG_LOG" "$LOGS_DIR/debug.log" || true
  fi
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

if [[ "$MODE" == "cli" ]]; then
  stop_vmstat
  write_timings_tsv "$DEBUG_LOG_PATH" "$METRICS_DIR/timings-debug.tsv"
  if [[ -f "$DEBUG_LOG_PATH" ]]; then
    cp -a "$DEBUG_LOG_PATH" "$METRICS_DIR/debug.log" || true
  fi
else
  stop_vmstat
  write_timings_tsv "$SERVER_DEBUG_LOG" "$METRICS_DIR/timings-debug.tsv"
fi

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
