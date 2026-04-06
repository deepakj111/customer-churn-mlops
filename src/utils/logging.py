import logging
import os
import sys
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_to_file: bool = False,
    log_file: str = "logs/app.log",
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    All modules in this project call get_logger(__name__) at the top.
    This gives every log line a consistent format with a clear source.

    Args:
        name: Logger name. Always pass __name__ from the calling module.
        log_to_file: If True, also write logs to a file on disk.
        log_file: Path to the log file. Only used when log_to_file is True.

    Returns:
        A Python Logger instance with console (and optionally file) handlers.
    """
    logger = logging.getLogger(name)

    # Guard: if this logger already has handlers, return it as-is.
    # Without this guard, calling get_logger() twice in the same module
    # would keep adding handlers, causing every log line to print twice.
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler — always active
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optional, off by default
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log records from bubbling up to the root logger.
    # Without this, if the root logger also has handlers, each message
    # would print once from our handler and once from root — duplicates.
    logger.propagate = False

    return logger
