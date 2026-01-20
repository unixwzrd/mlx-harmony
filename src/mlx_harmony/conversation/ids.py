from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def make_timestamp() -> dict[str, str | float]:
    """Return a UTC timestamp dict with unix and iso fields."""
    dt = datetime.utcnow()
    return {
        "unix": dt.timestamp(),
        "iso": dt.isoformat() + "Z",
    }


def make_message_id() -> str:
    """Return a stable message ID for chat logs."""
    return uuid4().hex


def make_chat_id() -> str:
    """Return a stable chat ID for chat logs."""
    return uuid4().hex
