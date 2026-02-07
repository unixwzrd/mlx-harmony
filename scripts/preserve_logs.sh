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

mkdir -p "$DEST_DIR"

MOVED=0
while IFS= read -r -d '' file; do
  mv -f "$file" "$DEST_DIR/" || true
  MOVED=$((MOVED + 1))
done < <(find "$SOURCE_DIR" -maxdepth 1 -type f \( \
  -name 'completion.*' -o -name 'parse.*' -o -name 'prompt.*' -o -name 'retry.*' -o \
  -name 'profiling-chat.json' -o -name 'debug.log' \
\) -print0)

if [[ "$MOVED" -eq 0 ]]; then
  existing_in_dest="$(find "$DEST_DIR" -maxdepth 1 -type f \( \
    -name 'completion.*' -o -name 'parse.*' -o -name 'prompt.*' -o -name 'retry.*' -o \
    -name 'profiling-chat.json' -o -name 'debug.log' \
  \) -print -quit)"
  if [[ -z "$existing_in_dest" ]]; then
    echo "[WARNING] No log artifacts moved from ${SOURCE_DIR} to ${DEST_DIR}" >&2
    ls -la "$SOURCE_DIR" 2>/dev/null || true
  fi
fi
