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

VM_FILTER_DEFAULT="scripts/filter-vm_stat.py"
VM_INTERVAL_DEFAULT="1"
VM_FILTER_ARGS_DEFAULT=(-t -b -s)   # TSV, bytes, silent

MERGE_SCRIPT_DEFAULT="scripts/merge_timing_metrics.py"
TPS_WIRED_PLOT_DEFAULT="scripts/TPSvsWiredMemory.py"

# -------- CLI (positional) --------
DATASET="${1:-$DATASET_DEFAULT}"
MODEL_PATH="${2:-$MODEL_DEFAULT}"
LIMIT="${3:-$LIMIT_DEFAULT}"

# -------- Env overrides --------
RUNS_ROOT="${RUNS_ROOT:-$RUNS_ROOT_DEFAULT}"
PROMPT_CONFIG="${PROMPT_CONFIG:-$PROMPT_CONFIG_DEFAULT}"
QUANT_LABEL="${QUANT_LABEL:-$QUANT_LABEL_DEFAULT}"
RUN_LABEL="${RUN_LABEL:-$RUN_LABEL_DEFAULT}"
VM_FILTER="${VM_FILTER:-$VM_FILTER_DEFAULT}"
VM_INTERVAL="${VM_INTERVAL:-$VM_INTERVAL_DEFAULT}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$MERGE_SCRIPT_DEFAULT}"
TPS_WIRED_PLOT="${TPS_WIRED_PLOT:-$TPS_WIRED_PLOT_DEFAULT}"

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

mkdir -p "${META_DIR}" "${LOGS_DIR}" "${METRICS_DIR}" "${ANALYSIS_DIR}" "${PLOTS_DIR}"

log() { printf '%s\n' "$*" >&2; }

# Default to shared logs/ so artifacts are collected in the expected place,
# then moved into the run directory after the run.
WORK_LOGS_DIR="${WORK_LOGS_DIR:-logs}"
DEBUG_LOG_PATH="${WORK_LOGS_DIR}/debug.log"
CHAT_LOG_PATH="${WORK_LOGS_DIR}/profiling-chat.json"

DEBUG_LOG_PATH="$(cd "${ROOT_DIR}" && DEBUG_LOG_PATH="${DEBUG_LOG_PATH}" python -c "import os; print(os.path.abspath(os.environ['DEBUG_LOG_PATH']))")"
CHAT_LOG_PATH="$(cd "${ROOT_DIR}" && CHAT_LOG_PATH="${CHAT_LOG_PATH}" python -c "import os; print(os.path.abspath(os.environ['CHAT_LOG_PATH']))")"

# -------- vm_stat management (kill pipeline reliably) --------
VM_PGID=""
start_vmstat() {
  local out_tsv="${METRICS_DIR}/vm_stat-timing.tsv"
  : >"${META_DIR}/vm_stat.stderr"

  if command -v setsid >/dev/null 2>&1; then
    # Start in a new session so pipeline has its own process group.
    setsid bash -c "
      set -euo pipefail
      vm_stat ${VM_INTERVAL} | '${VM_FILTER}' -d '${out_tsv}' ${VM_FILTER_ARGS_DEFAULT[*]}
    " >/dev/null 2>>"${META_DIR}/vm_stat.stderr" &
    VM_PGID="$!"
    VM_KILL_MODE="pgid"
    log "vm_stat started (pgid=${VM_PGID}) -> ${out_tsv}"
  else
    # Fallback for environments without setsid (e.g., macOS default).
    bash -c "
      set -euo pipefail
      vm_stat ${VM_INTERVAL} | '${VM_FILTER}' -d '${out_tsv}' ${VM_FILTER_ARGS_DEFAULT[*]}
    " >/dev/null 2>>"${META_DIR}/vm_stat.stderr" &
    VM_PGID="$!"
    VM_KILL_MODE="pid"
    log "vm_stat started (pid=${VM_PGID}) -> ${out_tsv}"
  fi
}

stop_vmstat() {
  if [[ -n "${VM_PGID}" ]]; then
    if [[ "${VM_KILL_MODE:-pid}" == "pgid" ]]; then
      # Kill entire process group.
      kill -- -"${VM_PGID}" >/dev/null 2>&1 || true
    else
      # Kill pipeline children then the parent shell.
      pkill -P "${VM_PGID}" >/dev/null 2>&1 || true
      kill "${VM_PGID}" >/dev/null 2>&1 || true
    fi
    wait "${VM_PGID}" >/dev/null 2>&1 || true
    log "vm_stat stopped (${VM_KILL_MODE:-pid}=${VM_PGID})"
    VM_PGID=""
  fi
}

cleanup() {
  stop_vmstat
}
trap cleanup EXIT INT TERM

# -------- Snapshot "what ran" --------
{
  echo "run_id=${RUN_ID}"
  echo "timestamp=${TS}"
  echo "dataset=${DATASET}"
  echo "model=${MODEL_PATH}"
  echo "limit=${LIMIT}"
  echo "prompt_config=${PROMPT_CONFIG}"
  echo "pwd=$(pwd)"
  echo "uname=$(uname -a)"
  echo "python=$(python --version 2>&1 || true)"
  echo "git_rev=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
} > "${META_DIR}/run.env"

# Copy prompt config for reproducibility
if [[ -f "${PROMPT_CONFIG}" ]]; then
  cp -a "${PROMPT_CONFIG}" "${META_DIR}/prompt-config.json"
fi

# -------- Prepare working logs (keep deterministic behavior) --------
if [[ "${WORK_LOGS_DIR}" == "logs" ]]; then
  if [[ -d "logs" ]]; then
    rm -f logs/profiling-chat.json logs/debug.log
    find logs -maxdepth 1 -type f \( \
      -name 'completion.*' -o \
      -name 'parse.*' -o \
      -name 'prompt.*' -o \
      -name 'retry.*' \
    \) -exec rm -f {} +
  fi
else
  mkdir -p "${WORK_LOGS_DIR}"
fi

# -------- Start vm_stat and run benchmark --------
start_vmstat

# Run the batch (your existing script). Keep a transcript.
log "running benchmark..."
if [[ "${BENCH_TTY:-0}" == "1" ]]; then
  DEBUG_LOG_PATH="${DEBUG_LOG_PATH}" CHAT_LOG_PATH="${CHAT_LOG_PATH}" VM_STAT_OUT="" \
    script -q "${META_DIR}/bench.time.log" bash -c "time -p bash tmp/deterministic-test.sh"
elif [[ "${BENCH_TEE:-0}" == "1" ]]; then
  (
    DEBUG_LOG_PATH="${DEBUG_LOG_PATH}" CHAT_LOG_PATH="${CHAT_LOG_PATH}" VM_STAT_OUT="" \
      time -p bash tmp/deterministic-test.sh
  ) 2>&1 | tee "${META_DIR}/bench.time.log"
else
  (
    DEBUG_LOG_PATH="${DEBUG_LOG_PATH}" CHAT_LOG_PATH="${CHAT_LOG_PATH}" VM_STAT_OUT="" \
      time -p bash tmp/deterministic-test.sh
  ) > "${META_DIR}/bench.time.log" 2>&1
  log "benchmark output -> ${META_DIR}/bench.time.log"
fi

# Stop vm_stat as soon as the model run is done
stop_vmstat

# -------- Collect analysis artifacts --------
# function-analysis and LOC stats go into analysis/
if command -v function-analysis >/dev/null 2>&1; then
  function-analysis src > "${ANALYSIS_DIR}/function-analysis.txt" || true
fi
find src -name "*.py" -print0 | xargs -0 wc -l | sort -rn > "${ANALYSIS_DIR}/LOC-stats.txt" || true

# timings from debug log into metrics/
if [[ -f "${DEBUG_LOG_PATH}" ]]; then
  grep ^TIMING_ "${DEBUG_LOG_PATH}" > "${METRICS_DIR}/timings-debug.tsv" || true
else
  log "timings skipped (missing debug log at ${DEBUG_LOG_PATH})"
fi

# Merge timings + vm_stat
if [[ -x "${MERGE_SCRIPT}" || -f "${MERGE_SCRIPT}" ]] && [[ -f "${METRICS_DIR}/vm_stat-timing.tsv" ]]; then
  python "${MERGE_SCRIPT}" \
    --timings "${METRICS_DIR}/timings-debug.tsv" \
    --vm-stat "${METRICS_DIR}/vm_stat-timing.tsv" \
    --output "${METRICS_DIR}/merged-timings-vm_stat.tsv" \
    || true
else
  log "merge skipped (missing vm_stat timing file)"
fi

# Plot TPS vs wired memory if merged output exists
if [[ -f "${METRICS_DIR}/merged-timings-vm_stat.tsv" ]] && [[ -x "${TPS_WIRED_PLOT}" || -f "${TPS_WIRED_PLOT}" ]]; then
  python "${TPS_WIRED_PLOT}" "${METRICS_DIR}/merged-timings-vm_stat.tsv" \
    --out "${PLOTS_DIR}/tps_vs_wired.png" \
    || true
fi

# -------- Move/copy run artifacts into run dir --------
# Move specific logs to avoid impacting subsequent runs.
if [[ "${WORK_LOGS_DIR}" == "logs" ]]; then
  if [[ -d "logs" ]]; then
    find logs -maxdepth 1 -type f \( \
      -name 'completion.*' -o \
      -name 'parse.*' -o \
      -name 'prompt.*' -o \
      -name 'retry.*' -o \
      -name 'profiling-chat.json' -o \
      -name 'debug.log' \
    \) -exec mv -f {} "${LOGS_DIR}/" \; || true
    # Ensure nothing is left behind for the next run.
    find logs -maxdepth 1 -type f \( \
      -name 'completion.*' -o \
      -name 'parse.*' -o \
      -name 'prompt.*' -o \
      -name 'retry.*' -o \
      -name 'profiling-chat.json' -o \
      -name 'debug.log' \
    \) -exec rm -f {} +
  fi
fi

# Archive logs at the end for reproducibility
if [[ -d "${WORK_LOGS_DIR}" ]]; then
  tar -czf "${META_DIR}/logs.tgz" -C "${WORK_LOGS_DIR}" . >/dev/null 2>&1 || true
fi

# Copy common profiling outputs if they exist
for f in profile.stats profile.stats.txt profile.static.txt profile.dot profile.svg profile.metrics.json; do
  [[ -f "stats/${f}" ]] && cp -a "stats/${f}" "${METRICS_DIR}/" || true
  [[ -f "${f}" ]] && cp -a "${f}" "${METRICS_DIR}/" || true
done

log "run complete -> ${RUN_DIR}"
