"""
Structured logging setup.

All loggers in this project use the "phishing_clip" hierarchy so the
root log level can be controlled from a single place via setup_logging().
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER_NAME = "phishing_clip"


def setup_logging(
    log_dir: str | Path = "outputs/logs",
    level: str = "INFO",
    console: bool = True,
) -> None:
    """
    Initialize the logging system. Call once at the start of each script.

    Sets up:
      - Console handler (StreamHandler → stderr) at the given level.
      - File handler → {log_dir}/training.log at DEBUG level.

    Args:
        log_dir:  Directory to write training.log.
        level:    Minimum level for console output ("DEBUG", "INFO", etc.).
        console:  If False, skip the console handler (useful for batch jobs).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter
    root.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(fmt)
        root.addHandler(ch)

    fh = logging.FileHandler(Path(log_dir) / "training.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the "phishing_clip" hierarchy.

    Usage:
        logger = get_logger(__name__)

    Args:
        name: Usually __name__ of the calling module.
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
