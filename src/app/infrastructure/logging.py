import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production-grade logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
        }

        # Include custom extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict): # type: ignore
            log_data.update(record.extra_fields)

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class StructuredLogger:
    """
    Logger wrapper to easily inject extra fields into structured logs.
    """
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        if self.logger.isEnabledFor(level):
            record_extra = {"extra_fields": extra} if extra else {}
            self.logger.log(level, msg, extra=record_extra, **kwargs)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self._log(logging.INFO, msg, extra)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self._log(logging.ERROR, msg, extra)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self._log(logging.WARNING, msg, extra)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        self._log(logging.DEBUG, msg, extra)


def setup_logging(level: str = "INFO"):
    """Sets up the global logging configuration with a JSON formatter."""
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
