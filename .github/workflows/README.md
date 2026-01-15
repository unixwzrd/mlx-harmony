# GitHub Actions Workflows

This directory contains CI/CD workflows for `mlx-harmony`.

## Workflows

### `ci.yml`

Main CI workflow that runs on push and pull requests.

**Jobs:**

1. **lint** - Runs on Ubuntu
   - Checks code formatting with `black`
   - Lints code with `ruff`
   - Fast feedback for style issues

2. **test-imports** - Runs on Ubuntu (Python 3.12, 3.13)
   - Tests basic imports
   - No MLX dependencies required
   - Fast validation that code structure is correct

3. **test-fast** - Runs on Ubuntu (Python 3.12, 3.13)
   - Runs all tests that don't require a model
   - Uses pytest markers: `-m "not slow and not requires_model"`
   - Tests config, tools, chat, server (with mocks)

4. **test-macos** - Runs on macOS-14 (Apple Silicon)
   - Full test suite including model tests
   - MLX requires macOS/Apple Silicon
   - Includes coverage reporting (on main branch)

## Test Matrix

The workflow uses pytest markers to categorize tests:

- **Fast tests** (`test-fast` job): No model required
  - `tests/test_imports.py`
  - `tests/test_config.py`
  - `tests/test_tools.py` (with mocks)
  - `tests/test_chat.py` (save/load only)
  - `tests/test_server.py` (with mocks)

- **Slow tests** (`test-macos` job): Requires model
  - `tests/test_generator.py` (marked `@pytest.mark.requires_model`)
  - `tests/test_server.py` (with real model, marked `@pytest.mark.requires_model`)

## Running Locally

You can run the same checks locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
black --check src/ tests/
ruff check src/ tests/
ruff format --check src/ tests/

# Run fast tests
pytest -m "not slow and not requires_model" -v

# Run all tests (requires macOS/Apple Silicon)
pytest -v

# Run with coverage
pytest --cov=mlx_harmony --cov-report=html -v
```

## Coverage

Coverage reports are generated on pushes to `main` branch:
- XML report for Codecov integration
- HTML report for local viewing
- Terminal summary

---

[← Back to README](../../README.md)
