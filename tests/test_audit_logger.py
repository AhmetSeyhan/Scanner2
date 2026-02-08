"""
Scanner Test Suite - Audit Logger Tests
Unit tests for utils/audit_logger.py
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audit_logger import AuditLogger


class TestAuditLogger:
    """Tests for the compliance audit trail."""

    @pytest.fixture
    def audit_dir(self, tmp_path):
        return str(tmp_path / "audit")

    def test_creates_log_directory(self, audit_dir):
        """AuditLogger creates its log directory."""
        al = AuditLogger(log_dir=audit_dir)
        assert os.path.isdir(audit_dir)

    def test_log_analysis_request(self, audit_dir):
        """log_analysis_request writes a JSONL entry."""
        al = AuditLogger(log_dir=audit_dir)
        al.log_analysis_request(
            session_id="S001",
            user="admin",
            ip_address="127.0.0.1",
            filename="test.mp4",
            video_hash="abc123",
        )

        # Find the log file
        log_files = [f for f in os.listdir(audit_dir) if f.startswith("audit_")]
        assert len(log_files) == 1

        with open(os.path.join(audit_dir, log_files[0])) as f:
            entry = json.loads(f.readline())

        assert entry["event"] == "analysis_request"
        assert entry["session_id"] == "S001"
        assert entry["user"] == "admin"
        assert entry["service"] == "scanner"

    def test_log_analysis_result(self, audit_dir):
        """log_analysis_result writes verdict details."""
        al = AuditLogger(log_dir=audit_dir)
        al.log_analysis_result(
            session_id="S002",
            user="analyst",
            verdict="AUTHENTIC",
            integrity_score=85.5,
            confidence=0.92,
            leading_core="BIOSIGNAL",
            duration_ms=1234.5,
        )

        log_files = [f for f in os.listdir(audit_dir) if f.startswith("audit_")]
        with open(os.path.join(audit_dir, log_files[0])) as f:
            entry = json.loads(f.readline())

        assert entry["event"] == "analysis_result"
        assert entry["verdict"] == "AUTHENTIC"
        assert entry["integrity_score"] == 85.5

    def test_log_auth_event(self, audit_dir):
        """log_auth_event writes auth events."""
        al = AuditLogger(log_dir=audit_dir)
        al.log_auth_event(
            user="admin",
            ip_address="10.0.0.1",
            event="login",
            success=True,
        )

        log_files = [f for f in os.listdir(audit_dir) if f.startswith("audit_")]
        with open(os.path.join(audit_dir, log_files[0])) as f:
            entry = json.loads(f.readline())

        assert entry["event"] == "auth.login"
        assert entry["success"] is True

    def test_log_error(self, audit_dir):
        """log_error writes error events."""
        al = AuditLogger(log_dir=audit_dir)
        al.log_error(
            session_id="S003",
            user="tester",
            error_code="PROCESSING_ERROR",
            message="Something failed",
        )

        log_files = [f for f in os.listdir(audit_dir) if f.startswith("audit_")]
        with open(os.path.join(audit_dir, log_files[0])) as f:
            entry = json.loads(f.readline())

        assert entry["event"] == "error"
        assert entry["error_code"] == "PROCESSING_ERROR"

    def test_multiple_entries_appended(self, audit_dir):
        """Multiple log calls append to same file."""
        al = AuditLogger(log_dir=audit_dir)
        al.log_error("S1", "u1", "E1", "msg1")
        al.log_error("S2", "u2", "E2", "msg2")

        log_files = [f for f in os.listdir(audit_dir) if f.startswith("audit_")]
        with open(os.path.join(audit_dir, log_files[0])) as f:
            lines = f.readlines()

        assert len(lines) == 2
