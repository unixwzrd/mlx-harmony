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

find "$LOGS_DIR" -maxdepth 1 -type f \( \
  -name 'completion.*' -o -name 'parse.*' -o -name 'prompt.*' -o -name 'retry.*' -o \
  -name 'profiling-chat.json' -o -name 'debug.log' \
\) -exec rm -f {} \;

if [[ "$STRICT" == "--strict" ]]; then
  leftovers=$(find "$LOGS_DIR" -maxdepth 1 -type f \( \
    -name 'completion.*' -o -name 'parse.*' -o -name 'prompt.*' -o -name 'retry.*' -o \
    -name 'profiling-chat.json' -o -name 'debug.log' \
  \) -print)
  if [[ -n "$leftovers" ]]; then
    echo "[ERROR] Cleanup failed; remaining artifacts in $LOGS_DIR:" >&2
    echo "$leftovers" >&2
    exit 1
  fi
fi
