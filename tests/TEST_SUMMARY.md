# Test Suite Summary

**Created**: 2026-01-07  
**Last Updated**: 2026-01-07

## Overview

The `mlx-harmony` test suite provides comprehensive coverage of core functionality, using a small test model from HuggingFace for inference tests.

## Test Files

### `test_imports.py`

- **Purpose**: Basic import smoke tests
- **Speed**: Fast (no model required)
- **Coverage**: Verifies all modules can be imported

### `test_config.py`

- **Purpose**: Config module testing
- **Speed**: Fast (no model required)
- **Coverage**:
  - Placeholder expansion (date, time, user-defined)
  - Config loading from JSON
  - Profile loading
  - Dialogue text parsing

### `test_generator.py`

- **Purpose**: TokenGenerator testing
- **Speed**: Slow (requires model download on first run)
- **Model**: `mlx-community/Qwen1.5-0.5B-Chat-4bit` (~300MB)
- **Coverage**:
  - Model loading
  - Format detection (GPT-OSS vs other models)
  - Token generation
  - Sampling parameters
  - Stop sequences
  - Prompt config integration

### `test_tools.py`

- **Purpose**: Tool parsing and execution
- **Speed**: Fast (no model required)
- **Coverage**:
  - Tool call parsing from Harmony messages
  - Tool execution (stubs)
  - Tool configuration

### `test_chat.py`

- **Purpose**: Chat module integration tests
- **Speed**: Fast (no model required)
- **Coverage**:
  - Conversation save/load
  - Metadata preservation
  - Timestamps
  - Hyperparameters per turn
  - Conversation appending

### `test_server.py`

- **Purpose**: HTTP API server tests
- **Speed**: Fast (with mocks) / Slow (with real model)
- **Coverage**:
  - `/v1/chat/completions` endpoint
  - Streaming responses
  - Profile resolution
  - Error handling
  - Integration with real model

## Test Model

**Model**: `mlx-community/Qwen1.5-0.5B-Chat-4bit`

- **Size**: ~300MB (small enough for CI/CD)
- **First Run**: Downloaded and cached by HuggingFace
- **Subsequent Runs**: Uses cache (fast)
- **Purpose**: Exercise full inference pipeline without requiring large models

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only fast tests (skip model downloads)
pytest -m "not slow and not requires_model"

# Run with coverage
pytest --cov=mlx_harmony --cov-report=html
```

### Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.requires_model` - Tests that need a model download
- `@pytest.mark.integration` - Integration tests

Run specific categories:

```bash
# Skip slow tests
pytest -m "not slow"

# Only run model tests
pytest -m "requires_model"

# Skip integration tests
pytest -m "not integration"
```

## Test Coverage

### ✅ Covered

- Config loading and validation
- Placeholder expansion (all types)
- Profile management
- Dialogue parsing
- Model loading (with test model)
- Token generation
- Sampling parameters
- Stop sequences
- Tool parsing
- Conversation save/load
- Metadata preservation

### ⏳ Planned

- End-to-end chat flow with model
- Server API endpoints
- Tool execution integration
- Error handling edge cases
- Performance benchmarks

## Fixtures

Common fixtures in `conftest.py`:

- `test_model_path` - HuggingFace model path for testing
- `temp_dir` - Temporary directory for test output
- `sample_prompt_config` - Sample prompt config dict
- `sample_conversation` - Sample conversation list
- `test_data_dir` - Test data directory
- `test_configs_dir` - Test configs directory

## CI/CD Ready

Tests are designed for CI/CD:

- Fast tests run without model downloads
- Model tests use small, cached model
- Tests are isolated and don't require external services
- All tests should pass on clean Python 3.12+ environments

## Next Steps

1. Add server API tests (`test_server.py`)
2. Add end-to-end integration tests with full chat flow
3. Set up CI/CD (GitHub Actions)
4. Add coverage reporting
5. Add performance benchmarks

---

[← Back to Tests README](README.md)
