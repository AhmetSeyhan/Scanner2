"""
Scanner Prime - History Manager
SQLite-based scan history for analytics.

Stores and retrieves scan history with full verdict details.
Supports recent activity display and statistics aggregation.

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from core.forensic_types import ScanHistoryEntry, FusionVerdict


class HistoryManager:
    """
    SQLite-based scan history manager.

    Stores last N scans with full verdict details.
    Default location: ~/.scanner/history.db
    """

    DEFAULT_DB_PATH = Path.home() / ".scanner" / "history.db"
    MAX_ENTRIES = 100  # Keep last 100 scans

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize history manager.

        Args:
            db_path: Custom database path (optional)
        """
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    integrity_score REAL NOT NULL,
                    biosignal_score REAL NOT NULL,
                    artifact_score REAL NOT NULL,
                    alignment_score REAL NOT NULL,
                    resolution TEXT,
                    duration_seconds REAL,
                    timestamp TEXT NOT NULL,
                    sha256_hash TEXT,
                    user TEXT
                )
            """)

            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON scan_history(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON scan_history(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verdict
                ON scan_history(verdict)
            """)
            conn.commit()

    def add_entry(
        self,
        session_id: str,
        filename: str,
        verdict: FusionVerdict,
        resolution: str,
        duration: float,
        sha256_hash: str,
        user: str = "anonymous"
    ) -> int:
        """
        Add new scan entry to history.

        Args:
            session_id: Unique session identifier
            filename: Original filename
            verdict: FusionVerdict from analysis
            resolution: Video resolution string
            duration: Video duration in seconds
            sha256_hash: SHA-256 hash of file
            user: Username who performed scan

        Returns:
            ID of inserted entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO scan_history (
                    session_id, filename, verdict, integrity_score,
                    biosignal_score, artifact_score, alignment_score,
                    resolution, duration_seconds, timestamp, sha256_hash, user
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                filename,
                verdict.verdict,
                verdict.integrity_score,
                verdict.biosignal_score,
                verdict.artifact_score,
                verdict.alignment_score,
                resolution,
                duration,
                datetime.utcnow().isoformat(),
                sha256_hash,
                user
            ))
            entry_id = cursor.lastrowid
            conn.commit()

        # Cleanup old entries
        self._cleanup_old_entries()

        return entry_id

    def add_entry_dict(
        self,
        session_id: str,
        filename: str,
        verdict_str: str,
        integrity_score: float,
        biosignal_score: float,
        artifact_score: float,
        alignment_score: float,
        resolution: str,
        duration: float,
        sha256_hash: str,
        user: str = "anonymous"
    ) -> int:
        """
        Add new scan entry using individual values (no FusionVerdict needed).

        Returns:
            ID of inserted entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO scan_history (
                    session_id, filename, verdict, integrity_score,
                    biosignal_score, artifact_score, alignment_score,
                    resolution, duration_seconds, timestamp, sha256_hash, user
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                filename,
                verdict_str,
                integrity_score,
                biosignal_score,
                artifact_score,
                alignment_score,
                resolution,
                duration,
                datetime.utcnow().isoformat(),
                sha256_hash,
                user
            ))
            entry_id = cursor.lastrowid
            conn.commit()

        self._cleanup_old_entries()
        return entry_id

    def get_recent(self, limit: int = 20) -> List[ScanHistoryEntry]:
        """
        Get recent scan entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of ScanHistoryEntry objects
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM scan_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            entries = []
            for row in cursor.fetchall():
                entries.append(ScanHistoryEntry(
                    id=row["id"],
                    session_id=row["session_id"],
                    filename=row["filename"],
                    verdict=row["verdict"],
                    integrity_score=row["integrity_score"],
                    biosignal_score=row["biosignal_score"],
                    artifact_score=row["artifact_score"],
                    alignment_score=row["alignment_score"],
                    resolution=row["resolution"] or "",
                    duration_seconds=row["duration_seconds"] or 0.0,
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    sha256_hash=row["sha256_hash"] or "",
                    user=row["user"] or "anonymous"
                ))

            return entries

    def get_by_session(self, session_id: str) -> Optional[ScanHistoryEntry]:
        """Get entry by session ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM scan_history WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return ScanHistoryEntry(
                    id=row["id"],
                    session_id=row["session_id"],
                    filename=row["filename"],
                    verdict=row["verdict"],
                    integrity_score=row["integrity_score"],
                    biosignal_score=row["biosignal_score"],
                    artifact_score=row["artifact_score"],
                    alignment_score=row["alignment_score"],
                    resolution=row["resolution"] or "",
                    duration_seconds=row["duration_seconds"] or 0.0,
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    sha256_hash=row["sha256_hash"] or "",
                    user=row["user"] or "anonymous"
                )
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_scans,
                    SUM(CASE WHEN verdict = 'AUTHENTIC' THEN 1 ELSE 0 END) as authentic_count,
                    SUM(CASE WHEN verdict = 'MANIPULATED' THEN 1 ELSE 0 END) as manipulated_count,
                    SUM(CASE WHEN verdict = 'UNCERTAIN' THEN 1 ELSE 0 END) as uncertain_count,
                    SUM(CASE WHEN verdict = 'INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive_count,
                    AVG(integrity_score) as avg_integrity_score
                FROM scan_history
            """)
            row = cursor.fetchone()
            return {
                "total_scans": row[0] or 0,
                "authentic_count": row[1] or 0,
                "manipulated_count": row[2] or 0,
                "uncertain_count": row[3] or 0,
                "inconclusive_count": row[4] or 0,
                "avg_integrity_score": round(row[5] or 0, 2)
            }

    def get_statistics_by_date(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily statistics for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as scans,
                    SUM(CASE WHEN verdict = 'MANIPULATED' THEN 1 ELSE 0 END) as threats
                FROM scan_history
                WHERE timestamp >= datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (f"-{days} days",))

            return [
                {"date": row[0], "scans": row[1], "threats": row[2]}
                for row in cursor.fetchall()
            ]

    def delete_entry(self, entry_id: int) -> bool:
        """Delete entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM scan_history WHERE id = ?",
                (entry_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> int:
        """Clear all history entries. Returns count of deleted entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM scan_history")
            count = cursor.rowcount
            conn.commit()
            return count

    def _cleanup_old_entries(self):
        """Remove entries beyond MAX_ENTRIES."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM scan_history
                WHERE id NOT IN (
                    SELECT id FROM scan_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, (self.MAX_ENTRIES,))
            conn.commit()
