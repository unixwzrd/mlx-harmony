#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx}"
EXTRA_ARGS=("${@:2}")
PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompt-config.deterministic.json}"
PROMPT_FILE="${PROMPT_FILE:-scripts/profile_chat_stdin.txt}"
CHAT_FILE="${CHAT_FILE:-profiling-chat.json}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-profile.stats}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-profile.svg}"
DEBUG_FILE="${DEBUG_FILE:-debug.log}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[ERROR] Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

cat "${PROMPT_FILE}" | scripts/profile_chat.py \
  --model "${MODEL_PATH}" \
  --profile-output "${PROFILE_OUTPUT}" \
  --graph "${GRAPH_OUTPUT}" \
  --prompt-config "${PROMPT_CONFIG}" \
  --debug-file "${DEBUG_FILE}" \
  --chat "${CHAT_FILE}" \
  "${EXTRA_ARGS[@]}"
