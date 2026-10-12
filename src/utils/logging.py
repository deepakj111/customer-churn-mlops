"""
Centralized logging configuration for the Customer Churn MLOps platform.

Provides both human-readable and structured JSON log formats:
    - TEXT format (default): For local development and notebook debugging.
    - JSON format: For production deployments where logs are aggregated by
      ELK, Datadog, CloudWatch, or similar platforms. JSON logs enable
      structured querying (e.g., filter by request_id, model_version).

Configuration via environment variables:
    LOG_LEVEL:   DEBUG | INFO | WARNING | ERROR (default: INFO)
    LOG_FORMAT:  text | json (default: text)

Public API:
    get_logger(name)  → configured Python Logger
"""

import json
import logging
import os
import sys
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_TYPE = os.getenv("LOG_FORMAT", "text").lower()
TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production log aggregation.

    Each log line is a single JSON object with consistent keys:
        timestamp, level, logger, message, module, funcName, lineno

    This enables log platforms to parse, index, and query logs without
    fragile regex patterns on free-text log lines.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


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

    # Select formatter based on LOG_FORMAT environment variable
    formatter: logging.Formatter
    if LOG_FORMAT_TYPE == "json":
        formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(fmt=TEXT_FORMAT, datefmt=DATE_FORMAT)

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
