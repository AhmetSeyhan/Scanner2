"""
Scanner Prime - History Service
Business logic for scan history retrieval and statistics.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from typing import Any, Dict, List, Optional

from core.exceptions import ProcessingError
from core.logging_config import get_logger

logger = get_logger("history_service")

# Lazy-loaded singleton
_history_manager = None
_init_attempted = False


def _get_manager():
    """Lazy-load HistoryManager (gracefully handles missing deps)."""
    global _history_manager, _init_attempted
    if not _init_attempted:
        _init_attempted = True
        try:
            from utils.history_manager import HistoryManager
            _history_manager = HistoryManager()
            logger.info("HistoryManager initialised")
        except Exception as exc:
            logger.warning(f"HistoryManager unavailable: {exc}")
    return _history_manager


class HistoryService:
    """Thin service over HistoryManager with validation."""

    @staticmethod
    def is_available() -> bool:
        return _get_manager() is not None

    @staticmethod
    def get_recent(limit: int = 20) -> List[Dict[str, Any]]:
        mgr = _get_manager()
        if mgr is None:
            raise ProcessingError("history", "History service unavailable")
        limit = max(1, min(limit, 100))
        entries = mgr.get_recent(limit)
        return [e.to_dict() for e in entries]

    @staticmethod
    def get_by_session(session_id: str):
        mgr = _get_manager()
        if mgr is None:
            raise ProcessingError("history", "History service unavailable")
        entry = mgr.get_by_session(session_id)
        if entry is None:
            return None
        return entry

    @staticmethod
    def get_statistics() -> Dict[str, Any]:
        mgr = _get_manager()
        if mgr is None:
            raise ProcessingError("history", "History service unavailable")
        return mgr.get_statistics()

    @staticmethod
    def add_entry_from_result(
        session_id: str,
        filename: str,
        result: Dict[str, Any],
        sha256_hash: str,
        user: str,
    ) -> Optional[int]:
        """Record an analysis result in history."""
        mgr = _get_manager()
        if mgr is None:
            return None
        try:
            return mgr.add_entry_dict(
                session_id=session_id,
                filename=filename,
                verdict_str=result.get("verdict", "UNKNOWN"),
                integrity_score=result.get("integrity_score", 0.0),
                biosignal_score=result.get("core_scores", {}).get("biosignal", 0.0),
                artifact_score=result.get("core_scores", {}).get("artifact", 0.0),
                alignment_score=result.get("core_scores", {}).get("alignment", 0.0),
                resolution=result.get("video_profile", {}).get("resolution", ""),
                duration=result.get("video_profile", {}).get("duration_seconds", 0.0),
                sha256_hash=sha256_hash,
                user=user,
            )
        except Exception as exc:
            logger.error(f"Failed to record history: {exc}")
            return None
