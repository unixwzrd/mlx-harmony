# Examples

This directory contains practical examples showing how to use `mlx-harmony`.

## Examples

### Basic Usage

- [`basic_chat.py`](basic_chat.py) - Simple chat interface using the Python API
- [`one_shot_generation.py`](one_shot_generation.py) - One-shot text generation
- [`with_prompt_config.py`](with_prompt_config.py) - Using prompt configs for customization

### Advanced Usage

- [`profile_usage.py`](profile_usage.py) - Using profiles to bundle models and configs
- [`conversation_resume.py`](conversation_resume.py) - Saving and resuming conversations
- [`custom_placeholders.py`](custom_placeholders.py) - Using custom placeholders in prompts
- [`sampling_parameters.py`](sampling_parameters.py) - Advanced sampling configuration

### API Server

- [`server_client.py`](server_client.py) - Client example for the HTTP API server

### Tool Integration

- [`tools_example.py`](tools_example.py) - Tool call parsing and execution (stubs)

## Running Examples

Make sure you have `mlx-harmony` installed:

```bash
pip install -e ".[dev]"
```

Then run any example:

```bash
python examples/basic_chat.py
```

## Notes

- Most examples use the test model `mlx-community/Qwen1.5-0.5B-Chat-4bit` (~300MB)
- For GPT-OSS models, you'll need access to `openai/gpt-oss-*` models
- Examples are designed to be educational and easy to understand

---

[← Back to README](../README.md)
