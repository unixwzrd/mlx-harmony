#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: profile_server_dataset.sh <english.json> <model_path> [limit]" >&2
  exit 2
fi

DATASET_PATH="$1"
MODEL_PATH="$2"
LIMIT="${3:-}"

RUN_ID=${RUN_ID:-profile-server-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-runs}

INTEGRATION_PROMPTS_FILE="$DATASET_PATH" \
INTEGRATION_TURNS="$LIMIT" \
MLX_HARMONY_MODEL_PATH="$MODEL_PATH" \
RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" SERVER_PROFILE=1 \
  scripts/run_dataset_harness.sh server
