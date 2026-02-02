from __future__ import annotations

import os
from pathlib import Path

from mlx_harmony.config import DEFAULT_CONFIG_DIR, load_profiles, resolve_config_path


def build_help_text() -> str:
    """Return CLI help text for out-of-band chat commands."""
    return (
        "\n[INFO] Out-of-band commands:\n"
        "  q, Control-D           - Quit the chat\n"
        "  \\help, /help          - Show this help message\n"
        "  \\list, /list          - List current hyperparameters\n"
        "  \\show, /show          - List current hyperparameters (alias for \\list)\n"
        "  \\models, /models      - List available models\n"
        "  \\set <param>=<value>  - Set a hyperparameter\n"
        "                          Example: \\set temperature=0.7\n"
        "                          Valid parameters: temperature, top_p, min_p, top_k,\n"
        "                          max_tokens, min_tokens_to_keep, repetition_penalty,\n"
        "                          repetition_context_size, xtc_probability, xtc_threshold, seed\n"
    )


def render_hyperparameters(hyperparameters: dict[str, float | int | bool | str]) -> str:
    """Render hyperparameters to a user-facing string."""
    if not hyperparameters:
        return "\n[INFO] Current hyperparameters:\n  (using defaults)\n"
    lines = ["\n[INFO] Current hyperparameters:"]
    for param, value in sorted(hyperparameters.items()):
        lines.append(f"  {param} = {value}")
    lines.append("")
    return "\n".join(lines)


def normalize_set_command(input_str: str) -> str:
    """Normalize a \\set command by removing leading slashes and keyword."""
    stripped = input_str.strip().lstrip("\\/")
    return stripped.removeprefix("set ").strip()


def parse_hyperparameter_update(
    param_name: str,
    param_value: str,
) -> tuple[bool, str, dict[str, float | int]]:
    """Parse a hyperparameter update into (ok, message, updates)."""
    try:
        parsed_value = float(param_value)
    except ValueError:
        return (
            False,
            f"[ERROR] Invalid value '{param_value}' for parameter '{param_name}'. Must be a number.",
            {},
        )

    float_params = [
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "xtc_probability",
        "xtc_threshold",
    ]
    int_params = [
        "max_tokens",
        "top_k",
        "min_tokens_to_keep",
        "repetition_context_size",
        "seed",
    ]

    if param_name in float_params:
        return True, f"[INFO] Set {param_name} = {parsed_value}", {param_name: parsed_value}
    if param_name in int_params:
        return True, f"[INFO] Set {param_name} = {int(parsed_value)}", {param_name: int(parsed_value)}

    valid_params = ", ".join(float_params + int_params)
    return (
        False,
        f"[ERROR] Unknown parameter '{param_name}'. Valid parameters: {valid_params}",
        {},
    )


def render_models_list(models_dir: str | None, profiles_file: str | None) -> str:
    """Render model list from models dir or profiles file."""
    models: list[str] = []
    if models_dir:
        path = Path(models_dir)
        if path.exists():
            for entry in sorted(path.iterdir()):
                if entry.is_dir():
                    models.append(str(entry))
    if not models and profiles_file:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        profiles_path = resolve_config_path(profiles_file, config_dir)
        if profiles_path and profiles_path.exists():
            profiles = load_profiles(str(profiles_path))
            for name, profile in profiles.items():
                model_id = profile.get("model", name)
                models.append(str(model_id))
    if not models:
        return "\n[INFO] No models found.\n"
    lines = ["\n[INFO] Available models:"]
    for model in models:
        lines.append(f"  {model}")
    lines.append("")
    return "\n".join(lines)


def parse_command(
    user_input: str,
    hyperparameters: dict[str, float | int | bool | str],
    models_dir: str | None = None,
    profiles_file: str | None = None,
) -> tuple[bool, bool, str, dict[str, float | int | bool | str]]:
    """Parse chat commands and return (handled, should_apply, message, updates)."""
    user_input_stripped = user_input.strip()
    user_input_lower = user_input_stripped.lower()

    if user_input_lower in ("\\help", "/help"):
        return True, False, build_help_text(), {}

    if user_input_lower in ("\\list", "/list", "\\show", "/show"):
        return True, False, render_hyperparameters(hyperparameters), {}

    if user_input_lower in ("\\models", "/models"):
        return True, False, render_models_list(models_dir, profiles_file), {}

    if user_input_stripped.startswith("\\set ") or user_input_stripped.startswith("/set "):
        set_cmd = normalize_set_command(user_input)
        if "=" in set_cmd:
            param_name, param_value = set_cmd.split("=", 1)
            param_name = param_name.strip().lower()
            param_value = param_value.strip()
            ok, message, updates = parse_hyperparameter_update(param_name, param_value)
            return True, ok, message, updates

        valid_params = (
            "temperature, top_p, min_p, top_k, max_tokens, "
            "min_tokens_to_keep, repetition_penalty, "
            "repetition_context_size, xtc_probability, xtc_threshold, seed"
        )
        return (
            True,
            False,
            "\n[ERROR] Invalid \\set command format.\n"
            "[INFO] Usage: \\set <param>=<value>\n"
            "[INFO] Example: \\set temperature=0.7\n"
            f"[INFO] Valid parameters: {valid_params}\n",
            {},
        )

    if user_input_stripped.startswith("\\") or user_input_stripped.startswith("/"):
        return True, False, f"\n[ERROR] Unknown out-of-band command.{build_help_text()}", {}

    return False, False, "", {}
