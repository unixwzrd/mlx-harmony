from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_harmony.logging import get_logger
from mlx_harmony.runtime.hyperparameters import resolve_param

logger = get_logger(__name__)


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
    """Return CLI help text for out-of-band chat commands."""
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
        "                          repetition_context_size, xtc_probability, xtc_threshold, seed\n"
    )


def render_hyperparameters(hyperparameters: dict[str, float | int | bool]) -> str:
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


def parse_command(
    user_input: str,
    hyperparameters: dict[str, float | int | bool],
) -> tuple[bool, bool, str, dict[str, float | int | bool]]:
    """Parse chat commands and return (handled, should_apply, message, updates)."""
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


def resolve_profile_and_prompt_config(
    args: Any,
    load_profiles: Callable[[str], dict[str, dict[str, Any]]],
    load_prompt_config: Callable[[str], Any | None],
) -> tuple[str, str | None, Any | None, dict[str, Any] | None]:
    """Resolve model and prompt config paths from CLI args and profiles."""
    profile_model = None
    profile_prompt_cfg = None
    profile_data: dict[str, Any] | None = None
    if args.profile:
        profiles = load_profiles(args.profiles_file)
        if args.profile not in profiles:
            raise SystemExit(
                f"Profile '{args.profile}' not found in {args.profiles_file}"
            )
        profile_data = profiles[args.profile]
        profile_model = profile_data.get("model")
        profile_prompt_cfg = profile_data.get("prompt_config")

    model_path = args.model or profile_model
    if not model_path:
        raise SystemExit("Model must be provided via --model or --profile")

    prompt_config_path = args.prompt_config or profile_prompt_cfg
    prompt_config = load_prompt_config(prompt_config_path) if prompt_config_path else None
    if prompt_config_path and prompt_config is None:
        raise SystemExit(f"Prompt config not found: {prompt_config_path}")

    return model_path, prompt_config_path, prompt_config, profile_data


def resolve_max_context_tokens(
    *,
    args: Any,
    loaded_max_context_tokens: int | None,
    loaded_model_path: str | None,
    prompt_config: Any | None,
    profile_data: dict[str, Any] | None,
    model_path: str,
    default_value: int = 4096,
) -> int:
    """Resolve max_context_tokens with CLI > metadata > config > profile > model > default."""
    if args.max_context_tokens is not None:
        return args.max_context_tokens
    if loaded_max_context_tokens is not None and loaded_model_path == model_path:
        return loaded_max_context_tokens
    if prompt_config and prompt_config.max_context_tokens:
        return prompt_config.max_context_tokens
    if profile_data and profile_data.get("max_context_tokens"):
        return int(profile_data["max_context_tokens"])

    detected = detect_model_max_context_tokens(model_path)
    if detected is not None:
        return detected

    logger.info("Using default max_context_tokens=%s", default_value)
    return default_value


def detect_model_max_context_tokens(model_path: str) -> int | None:
    """Return model_max_length from tokenizer_config.json when available."""
    path = Path(model_path).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        return None
    tokenizer_config = path / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return None
    try:
        data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid tokenizer_config.json at %s: %s (skipping auto-detect)",
            tokenizer_config,
            exc,
        )
        return None
    model_max_length = data.get("model_max_length")
    if isinstance(model_max_length, int) and model_max_length > 0:
        return model_max_length
    return None


def build_hyperparameters(
    args: Any,
    loaded_hyperparameters: dict[str, float | int | bool],
    prompt_config: Any | None,
    is_harmony: bool,
) -> dict[str, float | int | bool]:
    """Merge CLI, loaded, and config hyperparameters into a single dict."""
    default_max_tokens = 1024 if is_harmony else 512
    cfg = prompt_config
    hyperparameters = {
        "max_tokens": (
            args.max_tokens
            if args.max_tokens is not None
            else (
                loaded_hyperparameters.get("max_tokens")
                or resolve_param(
                    None,
                    cfg.max_tokens if cfg else None,
                    default_max_tokens,
                )
            )
        ),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else (
                loaded_hyperparameters.get("temperature")
                or resolve_param(None, cfg.temperature if cfg else None, None)
            )
        ),
        "top_p": (
            args.top_p
            if args.top_p is not None
            else (
                loaded_hyperparameters.get("top_p")
                or resolve_param(None, cfg.top_p if cfg else None, None)
            )
        ),
        "min_p": (
            args.min_p
            if args.min_p is not None
            else (
                loaded_hyperparameters.get("min_p")
                or resolve_param(None, cfg.min_p if cfg else None, None)
            )
        ),
        "top_k": (
            args.top_k
            if args.top_k is not None
            else (
                loaded_hyperparameters.get("top_k")
                or resolve_param(None, cfg.top_k if cfg else None, None)
            )
        ),
        "repetition_penalty": (
            args.repetition_penalty
            if args.repetition_penalty is not None
            else (
                loaded_hyperparameters.get("repetition_penalty")
                or resolve_param(None, cfg.repetition_penalty if cfg else None, None)
            )
        ),
        "repetition_context_size": (
            args.repetition_context_size
            if args.repetition_context_size is not None
            else (
                loaded_hyperparameters.get("repetition_context_size")
                or resolve_param(None, cfg.repetition_context_size if cfg else None, None)
            )
        ),
        "seed": (
            args.seed
            if args.seed is not None
            else (
                loaded_hyperparameters.get("seed")
                if loaded_hyperparameters.get("seed") is not None
                else resolve_param(None, cfg.seed if cfg else None, -1)
            )
        ),
        "reseed_each_turn": (
            args.reseed_each_turn
            if args.reseed_each_turn is not None
            else (
                loaded_hyperparameters.get("reseed_each_turn")
                if loaded_hyperparameters.get("reseed_each_turn") is not None
                else resolve_param(None, cfg.reseed_each_turn if cfg else None, False)
            )
        ),
    }
    return {k: v for k, v in hyperparameters.items() if v is not None}
