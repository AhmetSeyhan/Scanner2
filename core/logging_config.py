"""
Scanner Prime - Structured Logging Configuration
JSON-formatted logging for enterprise observability.

Provides:
- JSON-structured log output for log aggregation (ELK, Datadog, etc.)
- Correlation IDs per request
- Consistent field names across all modules
- Configurable log levels via environment

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # Merge any extra fields attached to the record
        for key in ("session_id", "user", "stage", "duration_ms", "error_code", "video_hash"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: Optional[str] = None,
    json_output: Optional[bool] = None,
) -> None:
    """
    Configure root logging for the Scanner application.

    Args:
        level: Log level override (default: from SCANNER_LOG_LEVEL env or INFO)
        json_output: Force JSON output (default: from SCANNER_LOG_JSON env or True)
    """
    log_level = level or os.getenv("SCANNER_LOG_LEVEL", "INFO")
    use_json = json_output if json_output is not None else os.getenv("SCANNER_LOG_JSON", "true").lower() == "true"

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for lib in ("urllib3", "botocore", "boto3", "s3transfer", "httpx", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Usage:
        from core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Analysis started", extra={"session_id": sid})
    """
    return logging.getLogger(f"scanner.{name}")
