# Troubleshooting Guide

**Created**: 2026-01-07  
**Last Updated**: 2026-01-17

Common issues and solutions for `mlx-harmony`.

---

## Installation Issues

### MLX Installation Fails

**Problem**: `pip install mlx-harmony` fails with MLX-related errors.

**Solutions**:

1. Ensure you're on macOS with Apple Silicon (M1/M2/M3)
2. Install MLX separately first:

   ```bash
   pip install mlx
   ```

3. Ensure Python 3.12+ is installed:

   ```bash
   python --version  # Should be 3.12 or higher
   ```

4. Try installing in a fresh virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install git+https://github.com/unixwzrd/mlx-harmony.git
   ```

### UnicodeFix Dependency Error

**Problem**: `unicodefix` dependency fails to install from git.

**Solutions**:

1. Ensure you have git installed and accessible
2. Try installing directly:

   ```bash
   pip install git+https://github.com/unixwzrd/UnicodeFix.git
   ```

3. Check network connectivity and GitHub access

---

## Model Loading Issues

### Model Not Found

**Problem**: `FileNotFoundError` or model path not recognized.

**Solutions**:

1. For HuggingFace models, ensure the model exists:

   ```bash
   # Test with mlx-lm directly
   python -c "from mlx_lm import load; load('mlx-community/Qwen1.5-0.5B-Chat-4bit')"
   ```

2. For local models, use absolute paths:

   ```bash
   mlx-harmony-chat --model /absolute/path/to/model
   ```

3. Check model directory structure (should have `config.json`, `*.safetensors`, etc.)

### Model Download Hangs or Fails

**Problem**: HuggingFace model download is slow or fails.

**Solutions**:

1. Check internet connection
2. Set HuggingFace cache directory:

   ```bash
   export HF_HOME=/path/to/cache
   ```

3. Pre-download model manually:

   ```bash
   python -c "from mlx_lm import load; load('model-name')"
   ```

4. Use a smaller model for testing (e.g., `mlx-community/Qwen1.5-0.5B-Chat-4bit`)

### Out of Memory Errors

**Problem**: Model is too large for available RAM.

**Solutions**:

1. Use a smaller/quantized model (e.g., 4-bit quantized)
2. Close other applications to free memory
3. Don't enable `mlock` if memory is limited (it reserves memory)
4. Use `--lazy` flag for lazy loading (slower but uses less memory upfront)

---

## Generation Issues

### No Response or Empty Output

**Problem**: Model generates nothing or very short responses.

**Solutions**:

1. Check `max_tokens` setting (may be too low):

   ```bash
   mlx-harmony-chat --max-tokens 512
   ```

2. Increase `temperature` (0.0 is deterministic, may repeat):

   ```bash
   mlx-harmony-chat --temperature 0.7
   ```

3. Check `repetition_penalty` (too high may stop generation):

   ```bash
   mlx-harmony-chat --repetition-penalty 1.1
   ```

4. For GPT-OSS models, check if only analysis is generated (no final response):
   - Increase `max_tokens`
   - Adjust `repetition_penalty`
   - Check prompt config for issues

### Repetitive Output

**Problem**: Model keeps repeating the same text.

**Solutions**:

1. Increase `repetition_penalty`:

   ```bash
   mlx-harmony-chat --repetition-penalty 1.2
   ```

2. Adjust `repetition_context_size`:

   ```bash
   mlx-harmony-chat --repetition-penalty 1.2 --repetition-context-size 64
   ```

3. Increase `temperature` for more variety:

   ```bash
   mlx-harmony-chat --temperature 0.8
   ```

### Slow Generation

**Problem**: Token generation is very slow.

**Solutions**:

1. Check if model is swapped to disk (macOS Activity Monitor)
2. Enable `mlock` in prompt config to keep model in wired memory:

   ```json
   {
     "mlock": true
   }
   ```

3. Use a smaller/quantized model
4. Check CPU/GPU utilization (MLX should use GPU on Apple Silicon)

### Slow Model Load

**Problem**: Model load takes longer than expected, even with sufficient RAM.

**Solutions**:

1. Try the experimental filesystem cache bypass:

   ```bash
   mlx-harmony-chat --model <path-or-repo> --no-fs-cache
   ```

2. If `mlock` is enabled, keep the process alive to avoid repeated load/unload cycles.

---

## Configuration Issues

### Prompt Config Not Loading

**Problem**: Prompt config file not found or invalid.

**Solutions**:

1. Use absolute path:

   ```bash
   mlx-harmony-chat --prompt-config /absolute/path/to/config.json
   ```

2. Check JSON syntax is valid:

   ```bash
   python -m json.tool config.json
   ```

3. Ensure file exists and is readable:

   ```bash
   ls -l configs/prompt-config.example.json
   ```

### Placeholders Not Expanding

**Problem**: Placeholders like `<|DATE|>` or `{key}` appear literally.

**Solutions**:

1. Ensure placeholders are in supported fields (system_model_identity, assistant_greeting, etc.)
2. Check placeholder format:
   - Built-in: `<|DATE|>`, `<|TIMEZ|>`, etc. (angle brackets)
   - Custom: `{key}` (curly braces) or `<|KEY|>` (angle brackets, case-insensitive)
3. Verify `placeholders` dict in config:

   ```json
   {
     "placeholders": {
       "key": "value"
     }
   }
   ```

### Profile Not Found

**Problem**: Profile specified with `--profile` not found.

**Solutions**:

1. Check profiles file exists:

   ```bash
   ls -l configs/profiles.example.json
   ```

2. Verify profile name matches (case-sensitive):

   ```bash
   cat configs/profiles.example.json | grep -A 3 "profile-name"
   ```

3. Use correct profiles file path:

   ```bash
   mlx-harmony-chat --profiles-file /path/to/profiles.json --profile profile-name
   ```

---

## Chat/Server Issues

### Markdown Rendering Issues

**Problem**: Assistant responses don't appear formatted, or formatting looks broken.

**Solutions**:

1. Markdown rendering is enabled by default. To disable:

   ```bash
   mlx-harmony-chat --no-markdown
   ```

2. If markdown rendering fails, ensure `rich` is installed:

   ```bash
   pip install rich>=13.0.0
   ```

3. Rich automatically handles both markdown and plain text, so even if the model generates plain text, it will display correctly

4. If output looks garbled when piping to files, use `--no-markdown`:

   ```bash
   mlx-harmony-chat --model openai/gpt-oss-20b --no-markdown > output.txt
   ```

### Out-of-Band Commands

**Problem**: Unclear what commands are available during chat, or getting errors on `\` commands.

**Solutions**:

1. Type `\help` during chat to see all available out-of-band commands:

   ```bash
   >> \help
   
   [INFO] Out-of-band commands:
     q, Control-D           - Quit the chat
     \help, /help          - Show this help message
     \list, /list          - List current hyperparameters
     \show, /show          - List current hyperparameters (alias for \list)
     \set <param>=<value>  - Set a hyperparameter
                             Example: \set temperature=0.7
                             Valid parameters: temperature, top_p, min_p, top_k,
                             max_tokens, min_tokens_to_keep, repetition_penalty,
                             repetition_context_size, xtc_probability, xtc_threshold
   ```

2. If you enter an invalid `\` command, you'll automatically see the list of valid commands

3. All commands work with either `\` or `/` prefix (e.g., both `\help` and `/help` work)

### Chat Session Not Saving

**Problem**: Conversation not saved to file.

**Solutions**:

1. Check write permissions on chat directory:

   ```bash
   ls -ld logs/
   chmod 755 logs/
   ```

2. Verify `chats_dir` in prompt config is valid:

   ```json
   {
     "chats_dir": "logs"
   }
   ```

3. Use absolute path if relative path doesn't work:

   ```json
   {
     "chats_dir": "/absolute/path/to/chats"
   }
   ```

4. Check disk space:

   ```bash
   df -h
   ```

### Server Won't Start

**Problem**: `mlx-harmony-server` fails to start.

**Solutions**:

1. Check if port 8000 is already in use:

   ```bash
   lsof -i :8000
   ```

2. Specify different port:

   ```bash
   # Note: This requires modifying server.py or using uvicorn directly
   uvicorn mlx_harmony.server:app --port 8080
   ```

3. Check model path is correct:

   ```bash
   mlx-harmony-server --model mlx-community/Qwen1.5-0.5B-Chat-4bit
   ```

### Server Timeouts

**Problem**: Server requests timeout or hang.

**Solutions**:

1. Increase client timeout:

   ```python
   httpx.Client(timeout=60.0)
   ```

2. Check server logs for errors
3. Ensure model is loaded (first request may be slow)
4. Reduce `max_tokens` in requests to speed up generation

---

## Tool Issues

### Tools Not Working

**Problem**: Tool calls not being parsed or executed.

**Solutions**:

1. Ensure you're using a GPT-OSS model (tools only work with GPT-OSS):

   ```bash
   mlx-harmony-chat --model openai/gpt-oss-20b --browser
   ```

2. Enable tools with flags:

   ```bash
   mlx-harmony-chat --browser --python --apply-patch
   ```

3. Note: Tool executors are currently stubs and need implementation
4. Check tool call parsing in debug mode:

   ```bash
   mlx-harmony-chat --debug --model openai/gpt-oss-20b --browser
   ```

---

## Debugging Tips

### Enable Debug Mode

**Problem**: Need to see what's happening internally.

**Solutions**:

1. Enable debug output:

   ```bash
   mlx-harmony-chat --debug
   ```

2. Output to file:

   ```bash
   mlx-harmony-chat --debug-file logs/debug.log
   ```

3. Check debug log for:
   - Raw prompts sent to model
   - Raw responses from model
   - Tool call detection
   - Placeholder expansion

### Verbose Test Output

**Problem**: Tests failing and need more info.

**Solutions**:

```bash
# Run tests with verbose output
pytest -vv tests/test_config.py

# Run specific test with output
pytest -vv tests/test_config.py::TestPlaceholderExpansion::test_date_placeholder -s

# Show print statements
pytest -s tests/
```

### Check Model Format Detection

**Problem**: Not sure if Harmony format is being used.

**Solutions**:

```python
from mlx_harmony import TokenGenerator

gen = TokenGenerator("openai/gpt-oss-20b", lazy=True)
print(f"Is GPT-OSS: {gen.is_gpt_oss}")
print(f"Uses Harmony: {gen.use_harmony}")
```

---

## Performance Issues

### High Memory Usage

**Problem**: Process using too much memory.

**Solutions**:

1. Check actual memory vs. wired memory (Activity Monitor on macOS)
2. Disable `mlock` if memory is limited:

   ```json
   {
     "mlock": false
   }
   ```

3. Use smaller models
4. Clear MLX cache between runs:

   ```python
   import mlx.core as mx
   mx.clear_cache()
   ```

### Slow Startup

**Problem**: Model loading takes a long time.

**Solutions**:

1. Use lazy loading:

   ```bash
   mlx-harmony-chat --lazy
   ```

3. Keep model in memory with `mlock` (faster subsequent loads)

---

## Getting Help

### Check Logs

- Debug logs: `logs/prompt-debug.log` (if `--debug-file` used)
- Chat logs: `logs/*.json` (conversation history)
- Server logs: Check console output

### Common Error Messages

- `ValueError: Unknown role` → Check message format, ensure roles are lowercase
- `FileNotFoundError` → Verify paths exist (absolute paths recommended)
- `ImportError` → Check dependencies installed: `pip install -e ".[dev]"`
- `MemoryError` → Model too large, use smaller/quantized model

### Report Issues

When reporting issues, include:

1. Python version: `python --version`
2. macOS version: `sw_vers`
3. Model path/name
4. Full error traceback
5. Relevant config files
6. Debug log output (if applicable)

---

[← Back to README](../README.md)
