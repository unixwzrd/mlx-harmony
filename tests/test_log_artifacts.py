"""Tests for benchmark log artifact parity scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

ARTIFACT_FILES = [
    "completion.raw.turn001.txt",
    "completion.cleaned.turn001.txt",
    "completion.tokens.turn001.json",
    "parse.channels.turn001.json",
    "prompt.full.turn001.txt",
    "prompt.tokens.turn001.json",
    "retry.decision.turn001.json",
    "profiling-chat.json",
    "debug.log",
    "server-run.log",
    "server-requests.log",
]


def _repo_root() -> Path:
    """Resolve repository root path from test file location."""
    return Path(__file__).resolve().parents[1]


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a shell script in repo root and capture output."""
    return subprocess.run(
        ["bash", *args],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_artifacts(directory: Path) -> None:
    """Create representative artifact files for move/cleanup tests."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        (directory / name).write_text("x", encoding="utf-8")


def test_preserve_logs_moves_all_artifacts() -> None:
    """Move completion/parse/prompt/retry/debug files into destination."""
    tmp_dir = _repo_root() / "tmp"
    source_dir = tmp_dir / "logs-preserve-source"
    dest_dir = tmp_dir / "logs-preserve-dest"
    _write_artifacts(source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    result = _run_script("scripts/preserve_logs.sh", str(source_dir), str(dest_dir))
    assert result.returncode == 0, result.stderr

    for name in ARTIFACT_FILES:
        assert not (source_dir / name).exists()
        assert (dest_dir / name).exists()


def test_preserve_logs_same_contract_for_cli_and_server_paths() -> None:
    """Ensure identical artifact contract for CLI and server destination dirs."""
    tmp_dir = _repo_root() / "tmp"
    cli_source = tmp_dir / "logs-cli-source"
    server_source = tmp_dir / "logs-server-source"
    cli_dest = tmp_dir / "runs" / "sample" / "logs" / "cli"
    server_dest = tmp_dir / "runs" / "sample" / "logs" / "server"
    _write_artifacts(cli_source)
    _write_artifacts(server_source)

    cli_result = _run_script("scripts/preserve_logs.sh", str(cli_source), str(cli_dest))
    server_result = _run_script("scripts/preserve_logs.sh", str(server_source), str(server_dest))
    assert cli_result.returncode == 0, cli_result.stderr
    assert server_result.returncode == 0, server_result.stderr

    cli_names = sorted(path.name for path in cli_dest.iterdir() if path.is_file())
    server_names = sorted(path.name for path in server_dest.iterdir() if path.is_file())
    assert cli_names == server_names


def test_clean_logs_removes_all_artifacts_strict() -> None:
    """Remove completion/parse/prompt/retry/debug files in strict mode."""
    tmp_dir = _repo_root() / "tmp"
    logs_dir = tmp_dir / "logs-clean-target"
    _write_artifacts(logs_dir)

    result = _run_script("scripts/clean_logs.sh", str(logs_dir), "--strict")
    assert result.returncode == 0, result.stderr

    for name in ARTIFACT_FILES:
        assert not (logs_dir / name).exists()

