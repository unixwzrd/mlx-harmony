#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <logs_dir> [--strict]" >&2
  exit 1
fi

LOGS_DIR="$1"
STRICT="${2:-}"

if [[ ! -d "$LOGS_DIR" ]]; then
  echo "[WARNING] Logs directory missing: $LOGS_DIR" >&2
  exit 0
fi

cd "$LOGS_DIR"
shopt -s nullglob
files=(
  completion.* parse.* prompt.* retry.*
  profiling-chat.json debug.log server-run.log server-requests.log
)
rm -f "${files[@]}"

if [[ "$STRICT" == "--strict" ]]; then
  remaining=(
    completion.* parse.* prompt.* retry.*
    profiling-chat.json debug.log server-run.log server-requests.log
  )
  if (( ${#remaining[@]} > 0 )); then
    echo "[ERROR] Cleanup failed; remaining artifacts in $LOGS_DIR:" >&2
    printf '%s\n' "${remaining[@]}" >&2
    exit 1
  fi
fi
