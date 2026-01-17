#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_DIR="${ROOT_DIR}/src/mlx_harmony"
VOICE_DIR="${SRC_DIR}/voice"
SPEECH_DIR="${SRC_DIR}/speech"
MOSHI_DIR="${SPEECH_DIR}/moshi"

VOICE_MOSHI="${VOICE_DIR}/voice_moshi.py"
MOSHI_LOADER="${MOSHI_DIR}/loader.py"
MOSHI_STT="${MOSHI_DIR}/stt.py"
MOSHI_TTS="${MOSHI_DIR}/tts.py"
MOSHI_INIT="${MOSHI_DIR}/__init__.py"
SPEECH_INIT="${SPEECH_DIR}/__init__.py"

echo "== Moshi speech realignment =="

mkdir -p "${MOSHI_DIR}"

if [[ -f "${VOICE_MOSHI}" ]]; then
  echo "Moving ${VOICE_MOSHI} -> ${MOSHI_LOADER}"
  git mv "${VOICE_MOSHI}" "${MOSHI_LOADER}"
else
  echo "WARN: ${VOICE_MOSHI} not found; skipping git mv."
fi

if [[ ! -f "${SPEECH_INIT}" ]]; then
  echo "Creating ${SPEECH_INIT}"
  cat <<'PY' > "${SPEECH_INIT}"
"""Speech subsystem packages."""
PY
fi

if [[ ! -f "${MOSHI_INIT}" ]]; then
  echo "Creating ${MOSHI_INIT}"
  cat <<'PY' > "${MOSHI_INIT}"
"""Moshi speech adapters."""

from mlx_harmony.speech.moshi.stt import MoshiSTT
from mlx_harmony.speech.moshi.tts import MoshiTTS

__all__ = ["MoshiSTT", "MoshiTTS"]
PY
fi

if [[ ! -f "${MOSHI_STT}" ]]; then
  echo "Creating ${MOSHI_STT}"
  cat <<'PY' > "${MOSHI_STT}"
"""Moshi STT adapter."""

from mlx_harmony.speech.moshi.loader import MoshiSTT

__all__ = ["MoshiSTT"]
PY
fi

if [[ ! -f "${MOSHI_TTS}" ]]; then
  echo "Creating ${MOSHI_TTS}"
  cat <<'PY' > "${MOSHI_TTS}"
"""Moshi TTS adapter."""

from mlx_harmony.speech.moshi.loader import MoshiTTS

__all__ = ["MoshiTTS"]
PY
fi

echo "Creating voice shim at ${VOICE_MOSHI}"
cat <<'PY' > "${VOICE_MOSHI}"
"""Compatibility shim for Moshi voice module (temporary)."""

from mlx_harmony.speech.moshi.loader import (
    MoshiSTT,
    MoshiTTS,
    chunk_text,
    log_moshi_config,
    require_moshi_mlx,
)

__all__ = [
    "MoshiSTT",
    "MoshiTTS",
    "chunk_text",
    "log_moshi_config",
    "require_moshi_mlx",
]
PY

echo "Done. Please update docs/SOURCE_FILE_MAP.md and imports as needed."
