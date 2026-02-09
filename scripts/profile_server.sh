#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

MODEL_PATH=""
PROMPT_CONFIG=""
PROFILES_FILE=""
DATASET=""
TURNS=""
HOST="127.0.0.1"
PORT="8000"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-0.5}"
SERVER_PROFILE_OUTPUT=""
CLIENT_PROFILE_OUTPUT=""
SERVER_LOG=""
SERVER_DEBUG_LOG=""
SERVER_REQUESTS_LOG=""
REPORT_FILE=""
BENCH_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --prompt-config) PROMPT_CONFIG="$2"; shift 2 ;;
    --profiles-file) PROFILES_FILE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --turns) TURNS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --server-profile-output) SERVER_PROFILE_OUTPUT="$2"; shift 2 ;;
    --client-profile-output) CLIENT_PROFILE_OUTPUT="$2"; shift 2 ;;
    --server-log) SERVER_LOG="$2"; shift 2 ;;
    --server-debug-log) SERVER_DEBUG_LOG="$2"; shift 2 ;;
    --server-requests-log) SERVER_REQUESTS_LOG="$2"; shift 2 ;;
    --report-file) REPORT_FILE="$2"; shift 2 ;;
    --bench-log) BENCH_LOG="$2"; shift 2 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$MODEL_PATH" && -n "$PROMPT_CONFIG" && -n "$PROFILES_FILE" && -n "$DATASET" && -n "$TURNS" ]] \
  || die "Missing required arguments"
[[ -n "$SERVER_PROFILE_OUTPUT" && -n "$CLIENT_PROFILE_OUTPUT" && -n "$SERVER_LOG" && -n "$SERVER_DEBUG_LOG" ]] \
  || die "Missing server profiling output/log arguments"
[[ -n "$SERVER_REQUESTS_LOG" && -n "$BENCH_LOG" ]] || die "Missing request/bench log arguments"

: >"$SERVER_LOG"
echo "[INFO] Starting server profiling: profile_output=${SERVER_PROFILE_OUTPUT}" >>"$SERVER_LOG"

MLX_HARMONY_SERVER_DEBUG_LOG="$SERVER_DEBUG_LOG" \
  MLX_HARMONY_SERVER_COLLECT_MEMORY=1 \
  MLX_HARMONY_SERVER_LOG_PROMPTS=1 \
  python3 scripts/profile_module.py \
    --module mlx_harmony.server \
    --profile-output "$SERVER_PROFILE_OUTPUT" \
    -- \
    --host "$HOST" \
    --port "$PORT" \
    --profiles-file "$PROFILES_FILE" \
    --model "$MODEL_PATH" \
    --prompt-config "$PROMPT_CONFIG" \
    --preload \
    >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
ok=0
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  if curl -fsS "http://${HOST}:${PORT}/v1/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep "$HEALTH_POLL_SECONDS"
done
[[ "$ok" == "1" ]] || {
  echo "ERROR: Health check failed for http://${HOST}:${PORT}/v1/health after ${HEALTH_TIMEOUT_SECONDS}s" >&2
  exit 1
}

client_cmd=(
  python3
  scripts/profile_module.py
  --module mlx_harmony.client
  --profile-output "$CLIENT_PROFILE_OUTPUT"
  --
  --host "$HOST"
  --port "$PORT"
  --timeout 300
  --health-retries 100
  --health-sleep 0.2
  --requests-log "$SERVER_REQUESTS_LOG"
)
if [[ -n "$REPORT_FILE" ]]; then
  client_cmd+=( --report-file "$REPORT_FILE" )
fi
scripts/build_prompt_stream.py "$DATASET" "$TURNS" | "${client_cmd[@]}" > "$BENCH_LOG" 2>&1

kill -INT "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [[ -s "$SERVER_PROFILE_OUTPUT" ]]; then
  echo "[INFO] Server profile file present: $SERVER_PROFILE_OUTPUT" >>"$SERVER_LOG"
else
  echo "[WARNING] Server profile file missing after shutdown: $SERVER_PROFILE_OUTPUT" >>"$SERVER_LOG"
fi
