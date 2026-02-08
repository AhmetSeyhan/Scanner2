"""
Scanner Prime - Audit Logger
Compliance-grade audit trail for GDPR/KVKK/SOC 2 requirements.

Logs every analysis request with:
- User/ID, IP address, timestamp
- Video hash, filename
- Verdict and decision details
- Request/response metadata

Writes to a dedicated JSON audit log file and optionally to Redis stream.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from core.logging_config import get_logger

logger = get_logger("audit")


class AuditLogger:
    """
    Append-only audit trail writer.

    Each entry is a single JSON line in the audit log file,
    suitable for ingestion by SIEM systems.
    """

    DEFAULT_LOG_DIR = Path.home() / ".scanner" / "audit"

    def __init__(
        self,
        log_dir: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._redis_client = None
        self._redis_stream = "scanner:audit"

        if redis_url:
            try:
                import redis
                self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
                logger.info("Audit logger connected to Redis stream")
            except Exception:
                self._redis_client = None
                logger.warning("Redis unavailable for audit stream, using file-only mode")

    def log_analysis_request(
        self,
        session_id: str,
        user: str,
        ip_address: str,
        filename: str,
        video_hash: str,
        action: str = "analysis_request",
    ) -> None:
        """Log an incoming analysis request."""
        self._write_entry({
            "event": action,
            "session_id": session_id,
            "user": user,
            "ip_address": ip_address,
            "filename": filename,
            "video_hash": video_hash,
        })

    def log_analysis_result(
        self,
        session_id: str,
        user: str,
        verdict: str,
        integrity_score: float,
        confidence: float,
        leading_core: str,
        duration_ms: float,
        filename: str = "",
        video_hash: str = "",
    ) -> None:
        """Log an analysis result."""
        self._write_entry({
            "event": "analysis_result",
            "session_id": session_id,
            "user": user,
            "filename": filename,
            "video_hash": video_hash,
            "verdict": verdict,
            "integrity_score": integrity_score,
            "confidence": confidence,
            "leading_core": leading_core,
            "duration_ms": duration_ms,
        })

    def log_auth_event(
        self,
        user: str,
        ip_address: str,
        event: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log authentication events (login, failed login, token refresh)."""
        self._write_entry({
            "event": f"auth.{event}",
            "user": user,
            "ip_address": ip_address,
            "success": success,
            "details": details or {},
        })

    def log_admin_action(
        self,
        user: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log administrative actions."""
        self._write_entry({
            "event": f"admin.{action}",
            "user": user,
            "details": details or {},
        })

    def log_error(
        self,
        session_id: str,
        user: str,
        error_code: str,
        message: str,
    ) -> None:
        """Log an error event."""
        self._write_entry({
            "event": "error",
            "session_id": session_id,
            "user": user,
            "error_code": error_code,
            "message": message,
        })

    def _write_entry(self, data: Dict[str, Any]) -> None:
        """Write a single audit entry to file and optionally Redis."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "scanner",
            "version": "3.3.0",
            **data,
        }

        # File output (one JSON line per entry)
        log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.error(f"Failed to write audit log: {exc}")

        # Redis stream (if available)
        if self._redis_client is not None:
            try:
                self._redis_client.xadd(
                    self._redis_stream,
                    {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in entry.items()},
                    maxlen=10000,
                )
            except Exception:
                pass  # File log is the authoritative source
