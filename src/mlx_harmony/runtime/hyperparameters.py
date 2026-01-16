from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def resolve_param(value: T | None, config_value: T | None, default: T | None) -> T | None:
    """Resolve a parameter from CLI, config, and defaults."""
    if value is None and config_value is None:
        return default
    return value if value is not None else config_value
