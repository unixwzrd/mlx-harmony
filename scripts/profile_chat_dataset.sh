#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: profile_chat_dataset.sh <english.json> <model_path> [limit] [extra profile_chat.py args...]" >&2
  exit 2
fi

DATASET_PATH="$1"
MODEL_PATH="$2"
LIMIT="${3:-}"
EXTRA_ARGS=("${@:4}")

PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompt-config.deterministic.json}"
CHAT_FILE="${CHAT_FILE:-profiling-chat.json}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-profile.stats}"
GRAPH_OUTPUT="${GRAPH_OUTPUT:-profile.svg}"
DEBUG_FILE="${DEBUG_FILE:-debug.log}"

if [[ -n "${LIMIT}" ]]; then
  scripts/build_prompt_stream.py "${DATASET_PATH}" "${LIMIT}" | \
    scripts/profile_chat.py \
      --model "${MODEL_PATH}" \
      --profile-output "${PROFILE_OUTPUT}" \
      --graph "${GRAPH_OUTPUT}" \
      --prompt-config "${PROMPT_CONFIG}" \
      --debug-file "${DEBUG_FILE}" \
      --chat "${CHAT_FILE}" \
      "${EXTRA_ARGS[@]}"
else
  scripts/build_prompt_stream.py "${DATASET_PATH}" | \
    scripts/profile_chat.py \
      --model "${MODEL_PATH}" \
      --profile-output "${PROFILE_OUTPUT}" \
      --graph "${GRAPH_OUTPUT}" \
      --prompt-config "${PROMPT_CONFIG}" \
      --debug-file "${DEBUG_FILE}" \
      --chat "${CHAT_FILE}" \
      "${EXTRA_ARGS[@]}"
fi
