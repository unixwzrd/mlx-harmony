#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <source_logs_dir> <dest_logs_dir>" >&2
  exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[WARNING] Logs source directory missing: $SOURCE_DIR" >&2
  exit 0
fi

SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd -P)"
if [[ "$DEST_DIR" != /* ]]; then
  DEST_DIR="${PWD}/${DEST_DIR}"
fi
mkdir -p "$DEST_DIR"
DEST_DIR="$(cd "$DEST_DIR" && pwd -P)"

cd "$SOURCE_DIR"
shopt -s nullglob
files=(
  completion.* parse.* prompt.* retry.*
  profiling-chat.json debug.log server-run.log server-requests.log
)

moved=0
for file in "${files[@]}"; do
  if [[ -e "$file" ]]; then
    mv -f -- "$file" "$DEST_DIR/"
    moved=$((moved + 1))
  fi
done

if [[ "$moved" -eq 0 ]]; then
  if ! compgen -G "${DEST_DIR}/completion.*" >/dev/null 2>&1 \
    && ! compgen -G "${DEST_DIR}/parse.*" >/dev/null 2>&1 \
    && ! compgen -G "${DEST_DIR}/prompt.*" >/dev/null 2>&1 \
    && ! compgen -G "${DEST_DIR}/retry.*" >/dev/null 2>&1 \
    && [[ ! -f "${DEST_DIR}/profiling-chat.json" ]] \
    && [[ ! -f "${DEST_DIR}/debug.log" ]] \
    && [[ ! -f "${DEST_DIR}/server-run.log" ]] \
    && [[ ! -f "${DEST_DIR}/server-requests.log" ]]; then
    echo "[WARNING] No log artifacts moved from ${SOURCE_DIR} to ${DEST_DIR}" >&2
    ls -la "$SOURCE_DIR" 2>/dev/null || true
  fi
fi
