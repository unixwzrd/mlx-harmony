#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

git rm src/mlx_harmony/config.py

echo "Removed src/mlx_harmony/config.py (use src/mlx_harmony/config/ package)."
