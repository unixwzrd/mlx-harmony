from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format=fmt or _DEFAULT_FORMAT)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    configure_logging(level=level)
    return logging.getLogger(name)


def configure_debug_file_logging(
    debug_path: Path,
    *,
    level: int = logging.WARNING,
    suppress_console_warnings: bool = True,
) -> None:
    logger = logging.getLogger()
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(debug_path):
            return
    file_handler = logging.FileHandler(debug_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(file_handler)
    if suppress_console_warnings:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.addFilter(_MaxLevelFilter(logging.INFO))
