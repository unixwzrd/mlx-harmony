# Server Guide

**Created**: 2026-01-28
**Updated**: 2026-01-28

## Purpose

Explain how to run the MLX Harmony API server and the minimal configuration needed to get started.

## Quick Start

```bash
python -m mlx_harmony.server \
  --host 0.0.0.0 \
  --port 8000 \
  --profiles-file configs/profiles.example.json
```

## Environment Variables

- `MLX_HARMONY_HOST`: Override the server host (default: `0.0.0.0`).
- `MLX_HARMONY_PORT`: Override the server port (default: `8000`).
- `MLX_HARMONY_LOG_LEVEL`: Uvicorn log level (default: `info`).
- `MLX_HARMONY_RELOAD`: Set to `true` to enable reloads in development.
- `MLX_HARMONY_WORKERS`: Number of worker processes (default: `1`).
- `MLX_HARMONY_PROFILES_FILE`: Profiles file for model selection (default: `configs/profiles.example.json`).

## Endpoints

- `POST /v1/chat/completions`: OpenAI-compatible chat completions.
- `GET /v1/models`: List available models from the profiles file.

## Profiles

Profiles provide a shortcut to map profile names to model paths and prompt configs. See [PROMPT_CONFIG_REFERENCE.md](./PROMPT_CONFIG_REFERENCE.md) for prompt config fields and [SOURCE_FILE_MAP.md](./SOURCE_FILE_MAP.md) for related files.
