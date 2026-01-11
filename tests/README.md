# Test Suite

**Created**: 2026-01-11
**Updated**: 2026-01-11

This directory contains the test suite for `mlx-harmony`.

## Test Structure

- [test_imports.py](test_imports.py) - Basic import tests (fast, no model required)
- [test_config.py](test_config.py) - Config loading, placeholder expansion, profile management
- [test_generator.py](test_generator.py) - TokenGenerator tests (requires model download)
- [test_tools.py](test_tools.py) - Tool parsing and execution tests
- [test_chat.py](test_chat.py) - Conversation save/load and integration tests
- [conftest.py](conftest.py) - Shared fixtures and test configuration

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

This installs `pytest` and other development dependencies.

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestPlaceholderExpansion::test_date_placeholder
```

### Run Model Tests

Model tests are skipped by default. To enable them, set `MLX_HARMONY_RUN_MODEL_TESTS=1`.

```bash
MLX_HARMONY_RUN_MODEL_TESTS=1 pytest -m "requires_model"
```

### Run Tests by Category

```bash
# Run only fast tests (skip model downloads)
pytest -m "not slow and not requires_model"

# Run only unit tests (skip integration tests)
pytest -m "not integration"

# Run only tests that require a model
pytest -m "requires_model"
```

## Test Model

Tests that require inference use a small model from HuggingFace:

- **Model**: `mlx-community/Qwen1.5-0.5B-Chat-4bit`
- **Size**: ~300MB
- **First Run**: Will be downloaded and cached by HuggingFace
- **Subsequent Runs**: Uses cached model (fast)

This model is small enough for CI/CD and local testing while still exercising the full inference pipeline.

## Test Coverage

Current test coverage includes:

- [x] **Config Module**
  - Placeholder expansion (date, time, user-defined)
  - Config loading from JSON
  - Profile loading
  - Dialogue parsing

- [x] **Generator Module**
  - Model loading
  - Format detection (GPT-OSS vs other models)
  - Token generation
  - Sampling parameters
  - Stop sequences

- [x] **Tools Module**
  - Tool call parsing
  - Tool execution (stubs)
  - Tool configuration

- [x] **Chat Module**
  - Conversation save/load
  - Metadata preservation
  - Timestamps
  - Hyperparameters per turn

- [ ] **Integration Tests** (planned)
  - End-to-end chat flow
  - Tool call integration
  - Server API endpoints

## Writing New Tests

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_something_slow():
    """This test takes a long time."""
    pass

@pytest.mark.requires_model
def test_with_model():
    """This test requires downloading a model."""
    pass

@pytest.mark.integration
def test_integration():
    """This is an integration test."""
    pass
```

### Fixtures

Common fixtures are available in [conftest.py](conftest.py):

- `test_model_path` - HuggingFace model path for testing
- `temp_dir` - Temporary directory for test output
- `sample_prompt_config` - Sample prompt config dict
- `sample_conversation` - Sample conversation list

### Example Test

```python
def test_my_feature(sample_prompt_config: dict, temp_dir: Path):
    """Test my new feature."""
    # Use fixtures
    config_file = temp_dir / "config.json"
    config_file.write_text(json.dumps(sample_prompt_config))
    
    # Test the feature
    config = load_prompt_config(str(config_file))
    assert config is not None
```

## CI/CD Integration

Tests are designed to work in CI/CD environments:

- Fast tests run without model downloads
- Model tests use a small, cached model
- Tests are isolated and don't require external services
- All tests should pass on clean Python 3.12+ environments

## Troubleshooting

### Tests Fail with Import Errors

Make sure you've installed the package in development mode:

```bash
pip install -e .
```

### Model Download Fails

If HuggingFace model download fails:

1. Check internet connection
2. Verify HuggingFace Hub access
3. Try manually downloading: `python -c "from mlx_lm import load; load('mlx-community/Qwen1.5-0.5B-Chat-4bit')"`

### Tests Are Slow

Use markers to skip slow tests:

```bash
pytest -m "not slow"
```

---

[← Back to README](../README.md)
