"""
Unit tests for HistoryManager (SQLite scan history).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forensic_types import FusionVerdict
from utils.history_manager import HistoryManager


class TestHistoryManager:
    """Test suite for HistoryManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a temporary history manager."""
        db_path = str(tmp_path / "test_history.db")
        return HistoryManager(db_path=db_path)

    @pytest.fixture
    def sample_verdict(self):
        return FusionVerdict(
            verdict="AUTHENTIC", integrity_score=85.0, confidence=0.82,
            biosignal_score=0.15, artifact_score=0.10, alignment_score=0.12,
            weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
            leading_core="alignment"
        )

    def test_initialization(self, manager):
        """Test database initialization."""
        assert manager.db_path.exists()

    def test_add_entry(self, manager, sample_verdict):
        """Test adding a scan entry."""
        entry_id = manager.add_entry(
            session_id="test-001",
            filename="test_video.mp4",
            verdict=sample_verdict,
            resolution="1080p",
            duration=30.0,
            sha256_hash="abc123def456",
            user="test_user"
        )
        assert entry_id > 0

    def test_add_entry_dict(self, manager):
        """Test adding entry with individual values."""
        entry_id = manager.add_entry_dict(
            session_id="test-002",
            filename="video2.mp4",
            verdict_str="MANIPULATED",
            integrity_score=25.0,
            biosignal_score=0.8,
            artifact_score=0.85,
            alignment_score=0.7,
            resolution="720p",
            duration=15.0,
            sha256_hash="xyz789",
            user="analyst"
        )
        assert entry_id > 0

    def test_get_recent(self, manager, sample_verdict):
        """Test retrieving recent entries."""
        for i in range(5):
            manager.add_entry(
                session_id=f"session-{i}",
                filename=f"video_{i}.mp4",
                verdict=sample_verdict,
                resolution="1080p",
                duration=30.0,
                sha256_hash=f"hash_{i}"
            )

        entries = manager.get_recent(limit=3)
        assert len(entries) == 3

    def test_get_by_session(self, manager, sample_verdict):
        """Test retrieving by session ID."""
        manager.add_entry(
            session_id="unique-session",
            filename="test.mp4",
            verdict=sample_verdict,
            resolution="1080p",
            duration=30.0,
            sha256_hash="hash123"
        )

        entry = manager.get_by_session("unique-session")
        assert entry is not None
        assert entry.session_id == "unique-session"
        assert entry.verdict == "AUTHENTIC"

    def test_get_by_session_not_found(self, manager):
        """Test retrieving non-existent session."""
        entry = manager.get_by_session("nonexistent")
        assert entry is None

    def test_get_statistics(self, manager, sample_verdict):
        """Test statistics aggregation."""
        manager.add_entry(
            session_id="s1", filename="v1.mp4", verdict=sample_verdict,
            resolution="1080p", duration=30.0, sha256_hash="h1"
        )

        stats = manager.get_statistics()
        assert stats["total_scans"] == 1
        assert stats["authentic_count"] == 1
        assert stats["manipulated_count"] == 0

    def test_delete_entry(self, manager, sample_verdict):
        """Test deleting an entry."""
        entry_id = manager.add_entry(
            session_id="to-delete", filename="v.mp4", verdict=sample_verdict,
            resolution="720p", duration=10.0, sha256_hash="h"
        )

        result = manager.delete_entry(entry_id)
        assert result is True

        entry = manager.get_by_session("to-delete")
        assert entry is None

    def test_clear_history(self, manager, sample_verdict):
        """Test clearing all history."""
        for i in range(3):
            manager.add_entry(
                session_id=f"s-{i}", filename=f"v{i}.mp4", verdict=sample_verdict,
                resolution="1080p", duration=30.0, sha256_hash=f"h{i}"
            )

        count = manager.clear_history()
        assert count == 3

        entries = manager.get_recent()
        assert len(entries) == 0

    def test_max_entries_cleanup(self, manager, sample_verdict):
        """Test that old entries are cleaned up beyond MAX_ENTRIES."""
        manager.MAX_ENTRIES = 5

        for i in range(10):
            manager.add_entry(
                session_id=f"overflow-{i}", filename=f"v{i}.mp4",
                verdict=sample_verdict, resolution="1080p",
                duration=30.0, sha256_hash=f"h{i}"
            )

        entries = manager.get_recent(limit=100)
        assert len(entries) <= 5

    def test_entry_to_dict(self, manager, sample_verdict):
        """Test entry serialization."""
        manager.add_entry(
            session_id="dict-test", filename="v.mp4", verdict=sample_verdict,
            resolution="1080p", duration=30.0, sha256_hash="hash"
        )

        entry = manager.get_by_session("dict-test")
        d = entry.to_dict()
        assert "session_id" in d
        assert "verdict" in d
        assert "timestamp" in d
