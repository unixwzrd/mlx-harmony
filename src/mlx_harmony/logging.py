from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format=fmt or _DEFAULT_FORMAT)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    configure_logging(level=level)
    return logging.getLogger(name)
