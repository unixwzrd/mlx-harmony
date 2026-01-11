from __future__ import annotations

from collections.abc import Callable
from typing import Any


def get_assistant_name(prompt_config: Any | None) -> str:
    """Get assistant name from prompt config, defaulting to 'Assistant'."""
    if prompt_config and getattr(prompt_config, "placeholders", None):
        return prompt_config.placeholders.get("assistant", "Assistant")
    return "Assistant"


def get_truncate_limits(prompt_config: Any | None) -> tuple[int, int]:
    """
    Get truncate limits from prompt config.

    Returns:
        (thinking_limit, response_limit) tuple with defaults (1000, 1000)
    """
    thinking_limit = (
        prompt_config.truncate_thinking
        if prompt_config and prompt_config.truncate_thinking is not None
        else 1000
    )
    response_limit = (
        prompt_config.truncate_response
        if prompt_config and prompt_config.truncate_response is not None
        else 1000
    )
    return (thinking_limit, response_limit)


def truncate_text(text: str, limit: int) -> str:
    """Truncate text to limit, appending '... [truncated]' if needed."""
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


def build_help_text() -> str:
    return (
        "\n[INFO] Out-of-band commands:\n"
        "  q, Control-D           - Quit the chat\n"
        "  \\help, /help          - Show this help message\n"
        "  \\list, /list          - List current hyperparameters\n"
        "  \\show, /show          - List current hyperparameters (alias for \\list)\n"
        "  \\set <param>=<value>  - Set a hyperparameter\n"
        "                          Example: \\set temperature=0.7\n"
        "                          Valid parameters: temperature, top_p, min_p, top_k,\n"
        "                          max_tokens, min_tokens_to_keep, repetition_penalty,\n"
        "                          repetition_context_size, xtc_probability, xtc_threshold\n"
    )


def render_hyperparameters(hyperparameters: dict[str, float | int]) -> str:
    if not hyperparameters:
        return "\n[INFO] Current hyperparameters:\n  (using defaults)\n"
    lines = ["\n[INFO] Current hyperparameters:"]
    for param, value in sorted(hyperparameters.items()):
        lines.append(f"  {param} = {value}")
    lines.append("")
    return "\n".join(lines)


def normalize_set_command(input_str: str) -> str:
    stripped = input_str.strip().lstrip("\\/")
    return stripped.removeprefix("set ").strip()


def parse_hyperparameter_update(
    param_name: str,
    param_value: str,
) -> tuple[bool, str, dict[str, float | int]]:
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


def parse_command(
    user_input: str,
    hyperparameters: dict[str, float | int],
) -> tuple[bool, bool, str, dict[str, float | int]]:
    user_input_stripped = user_input.strip()
    user_input_lower = user_input_stripped.lower()

    if user_input_lower in ("\\help", "/help"):
        return True, False, build_help_text(), {}

    if user_input_lower in ("\\list", "/list", "\\show", "/show"):
        return True, False, render_hyperparameters(hyperparameters), {}

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
            "repetition_context_size, xtc_probability, xtc_threshold"
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


def resolve_profile_and_prompt_config(
    args: Any,
    load_profiles: Callable[[str], dict[str, dict[str, Any]]],
    load_prompt_config: Callable[[str], Any | None],
) -> tuple[str, str | None, Any | None]:
    profile_model = None
    profile_prompt_cfg = None
    if args.profile:
        profiles = load_profiles(args.profiles_file)
        if args.profile not in profiles:
            raise SystemExit(
                f"Profile '{args.profile}' not found in {args.profiles_file}"
            )
        profile = profiles[args.profile]
        profile_model = profile.get("model")
        profile_prompt_cfg = profile.get("prompt_config")

    model_path = args.model or profile_model
    if not model_path:
        raise SystemExit("Model must be provided via --model or --profile")

    prompt_config_path = args.prompt_config or profile_prompt_cfg
    prompt_config = (
        load_prompt_config(prompt_config_path) if prompt_config_path else None
    )

    return model_path, prompt_config_path, prompt_config


def build_hyperparameters(
    args: Any,
    loaded_hyperparameters: dict[str, float | int],
    prompt_config: Any | None,
    is_harmony: bool,
) -> dict[str, float | int]:
    default_max_tokens = 1024 if is_harmony else 512
    hyperparameters = {
        "max_tokens": (
            args.max_tokens
            if args.max_tokens is not None
            else (
                loaded_hyperparameters.get("max_tokens")
                or (prompt_config.max_tokens if prompt_config else None)
                or default_max_tokens
            )
        ),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else (
                loaded_hyperparameters.get("temperature")
                or (prompt_config.temperature if prompt_config else None)
            )
        ),
        "top_p": (
            args.top_p
            if args.top_p is not None
            else (
                loaded_hyperparameters.get("top_p")
                or (prompt_config.top_p if prompt_config else None)
            )
        ),
        "min_p": (
            args.min_p
            if args.min_p is not None
            else (
                loaded_hyperparameters.get("min_p")
                or (prompt_config.min_p if prompt_config else None)
            )
        ),
        "top_k": (
            args.top_k
            if args.top_k is not None
            else (
                loaded_hyperparameters.get("top_k")
                or (prompt_config.top_k if prompt_config else None)
            )
        ),
        "repetition_penalty": (
            args.repetition_penalty
            if args.repetition_penalty is not None
            else (
                loaded_hyperparameters.get("repetition_penalty")
                or (prompt_config.repetition_penalty if prompt_config else None)
            )
        ),
        "repetition_context_size": (
            args.repetition_context_size
            if args.repetition_context_size is not None
            else (
                loaded_hyperparameters.get("repetition_context_size")
                or (prompt_config.repetition_context_size if prompt_config else None)
            )
        ),
    }
    return {k: v for k, v in hyperparameters.items() if v is not None}

