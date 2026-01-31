#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOGS_DIR=${1:-logs}

cd "$ROOT_DIR"

if [[ ! -d "$LOGS_DIR" ]]; then
  exit 0
fi

rm -f "${LOGS_DIR}/profiling-chat.json" "${LOGS_DIR}/debug.log"
find "$LOGS_DIR" -maxdepth 1 -type f \( \
  -name 'completion.*' -o \
  -name 'parse.*' -o \
  -name 'prompt.*' -o \
  -name 'retry.*' \
\) -exec rm -f {} +
