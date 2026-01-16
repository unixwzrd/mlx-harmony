"""Compatibility wrapper for the legacy chat_migration module."""

from __future__ import annotations

from mlx_harmony.conversation.conversation_migration import (
    build_chat_container,
    main,
    migrate_chat_data,
    validate_chat_container,
)

__all__ = [
    "build_chat_container",
    "main",
    "migrate_chat_data",
    "validate_chat_container",
]
