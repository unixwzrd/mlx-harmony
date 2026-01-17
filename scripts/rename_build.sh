#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

git mv src/mlx_harmony/voice/check_mic.py src/mlx_harmony/tools/hotmic.py
git mv scripts/check_mic.py scripts/hotmic.py

cat > src/mlx_harmony/check_mic.py <<'PY'
"""CLI entrypoint wrapper for mic check."""

from __future__ import annotations

from mlx_harmony.tools.hotmic import main

__all__ = ["main"]
PY

cat > scripts/hotmic.py <<'PY'
#!/usr/bin/env python
"""Quick microphone permission and input smoke test."""

from __future__ import annotations

from mlx_harmony.tools.hotmic import main

if __name__ == "__main__":
    main()
PY

python - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
data = path.read_text(encoding="utf-8")
data = data.replace(
    'hotmic = "mlx_harmony.check_mic:main"',
    'hotmic = "mlx_harmony.tools.hotmic:main"',
)
path.write_text(data, encoding="utf-8")
PY

python - <<'PY'
from pathlib import Path

path = Path("docs/SOURCE_FILE_MAP.md")
data = path.read_text(encoding="utf-8")
data = data.replace(
    "[check_mic.py](../src/mlx_harmony/check_mic.py): Mic checker wrapper (used by `hotmic`).",
    "[check_mic.py](../src/mlx_harmony/check_mic.py): Mic checker wrapper (used by `hotmic`).",
)
data = data.replace(
    "[voice/check_mic.py](../src/mlx_harmony/voice/check_mic.py): Mic permission test (used by `hotmic`).",
    "[tools/hotmic.py](../src/mlx_harmony/tools/hotmic.py): Mic permission test (used by `hotmic`).",
)
data = data.replace(
    "| [voice/check_mic.py](../src/mlx_harmony/voice/check_mic.py) | Mic permission checker. | — |",
    "| [tools/hotmic.py](../src/mlx_harmony/tools/hotmic.py) | Mic permission checker. | — |",
)
path.write_text(data, encoding="utf-8")
PY

echo "hotmic rename complete. Review git status before committing."
