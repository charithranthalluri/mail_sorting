"""Structured logging configuration for the mail sorter application.

Provides a consistent logging setup with:
- Console output with colored formatting (when available)
- Optional rotating file handler
- Trace ID support for per-email processing correlation
- Safe defaults that avoid logging sensitive email content
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# Context variable for per-email trace IDs
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Return the current trace ID, or empty string if not set."""
    return _trace_id.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """Set a new trace ID for the current processing context.

    Args:
        trace_id: Explicit trace ID. If None, a new UUID-based ID is generated.

    Returns:
        The trace ID that was set.
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def clear_trace_id() -> None:
    """Clear the current trace ID."""
    _trace_id.set("")


class TraceFormatter(logging.Formatter):
    """Formatter that injects the trace ID into log records."""

    def format(self, record: logging.LogRecord) -> str:
        trace = get_trace_id()
        record.trace_id = f"[{trace}] " if trace else ""
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the root application logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output. Parent directories are created.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        The configured root logger for the application.
    """
    root_logger = logging.getLogger("mail_sorter")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root_logger.handlers:
        return root_logger

    fmt = "%(asctime)s %(levelname)-8s %(trace_id)s%(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = TraceFormatter(fmt, datefmt=date_fmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the mail_sorter namespace.

    Args:
        name: Logger name, typically the module name (e.g., 'gmail.reader').

    Returns:
        A logger instance.
    """
    return logging.getLogger(f"mail_sorter.{name}")
