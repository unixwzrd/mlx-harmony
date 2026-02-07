#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  run_profile.sh --mode cli --model <path> --prompt-config <path> \
    --dataset <json> --turns <n> [--max-tokens <n>] \
    --profile-out <stats> --bench-log <path>

  run_profile.sh --mode server --model <path> --prompt-config <path> --profiles-file <path> \
    --dataset <json> --turns <n> [--max-tokens <n>] --host <host> --port <port> \
    --server-profile-out <stats> --client-profile-out <stats> \
    --server-log <path> --server-debug-log <path> --server-requests-log <path> --bench-log <path>
EOF
  exit 1
}

MODE=""
MODEL=""
PROMPT_CONFIG=""
PROFILES_FILE=""
DATASET=""
TURNS=""
MAX_TOKENS=""
HOST="127.0.0.1"
PORT="8000"
PROFILE_OUT=""
CLIENT_PROFILE_OUT=""
SERVER_PROFILE_OUT=""
SERVER_LOG=""
SERVER_DEBUG_LOG=""
SERVER_REQUESTS_LOG=""
BENCH_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt-config) PROMPT_CONFIG="$2"; shift 2 ;;
    --profiles-file) PROFILES_FILE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --profile-out) PROFILE_OUT="$2"; shift 2 ;;
    --client-profile-out) CLIENT_PROFILE_OUT="$2"; shift 2 ;;
    --server-profile-out) SERVER_PROFILE_OUT="$2"; shift 2 ;;
    --server-log) SERVER_LOG="$2"; shift 2 ;;
    --server-debug-log) SERVER_DEBUG_LOG="$2"; shift 2 ;;
    --server-requests-log) SERVER_REQUESTS_LOG="$2"; shift 2 ;;
    --bench-log) BENCH_LOG="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "$MODE" && -n "$MODEL" && -n "$PROMPT_CONFIG" && -n "$DATASET" && -n "$TURNS" ]] || usage

if [[ "$MODE" == "cli" ]]; then
  [[ -n "$PROFILE_OUT" && -n "$BENCH_LOG" ]] || usage
  cli_cmd=(
    scripts/profile_module.py cli --
    --model "$MODEL"
    --prompt-config "$PROMPT_CONFIG"
    --debug-file "debug.log"
    --profile-output "$PROFILE_OUT"
    --text-only
  )
  if [[ -n "$MAX_TOKENS" ]]; then
    cli_cmd+=( --max-tokens "$MAX_TOKENS" )
  fi
  scripts/build_prompt_stream.py "$DATASET" "$TURNS" | "${cli_cmd[@]}" > "$BENCH_LOG" 2>&1
  exit 0
fi

if [[ "$MODE" == "server" ]]; then
  [[ -n "$PROFILES_FILE" && -n "$CLIENT_PROFILE_OUT" && -n "$SERVER_PROFILE_OUT" ]] || usage
  [[ -n "$SERVER_LOG" && -n "$SERVER_DEBUG_LOG" && -n "$SERVER_REQUESTS_LOG" && -n "$BENCH_LOG" ]] || usage

  : >"$SERVER_LOG"
  {
    echo "[INFO] Starting server profiling: profile_output=${SERVER_PROFILE_OUT}"
    echo "[INFO] Using profiler: python -m cProfile"
  } >>"$SERVER_LOG"

  MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG" \
    MLX_HARMONY_SERVER_COLLECT_MEMORY=1 \
    MLX_HARMONY_SERVER_LOG_PROMPTS=1 \
    python -m cProfile -o "$SERVER_PROFILE_OUT" -m mlx_harmony.server \
      --host "$HOST" --port "$PORT" --profiles-file "$PROFILES_FILE" \
      >>"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  ok=0
  for _ in $(seq 1 100); do
    curl -fsS "http://${HOST}:${PORT}/v1/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 0.2
  done
  [[ "$ok" == "1" ]] || { echo "ERROR: Health check failed for http://${HOST}:${PORT}/v1/health" >&2; exit 1; }

  client_cmd=(
    scripts/profile_module.py client --
    --host "$HOST" --port "$PORT"
    --model "$MODEL"
    --prompt-config "$PROMPT_CONFIG"
    --timeout 300
    --health-retries 100
    --health-sleep 0.2
    --requests-log "$SERVER_REQUESTS_LOG"
    --profile-output "$CLIENT_PROFILE_OUT"
    --text-only
  )
  if [[ -n "$MAX_TOKENS" ]]; then
    client_cmd+=( --max-tokens "$MAX_TOKENS" )
  fi
  scripts/build_prompt_stream.py "$DATASET" "$TURNS" | "${client_cmd[@]}" > "$BENCH_LOG" 2>&1

  kill -INT "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true

  if [[ -s "$SERVER_PROFILE_OUT" ]]; then
    echo "[INFO] Server profile file present: $SERVER_PROFILE_OUT" >>"$SERVER_LOG"
  else
    echo "[WARNING] Server profile file missing after shutdown: $SERVER_PROFILE_OUT" >>"$SERVER_LOG"
  fi
  exit 0
fi

usage
