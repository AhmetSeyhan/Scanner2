"""
Scanner Test Suite - Forensic Hash Chain Tests
Unit tests for utils/forensic_hash.py
"""

import os
import sys
import json
import hashlib
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.forensic_hash import ForensicHashChain, compute_file_hash


class TestForensicHashChain:
    """Tests for the chain-of-custody hash system."""

    def test_chain_initialisation(self):
        """Chain starts empty with session_id."""
        chain = ForensicHashChain("TEST-001")
        assert chain.session_id == "TEST-001"
        assert len(chain.chain) == 0

    def test_hash_file(self, tmp_path):
        """hash_file produces correct SHA-256."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        chain = ForensicHashChain("TEST-002")
        digest = chain.hash_file(str(test_file))

        expected = hashlib.sha256(b"hello world").hexdigest()
        assert digest == expected
        assert len(chain.chain) == 1
        assert chain.chain[0]["label"] == "original_file"

    def test_hash_bytes(self):
        """hash_bytes produces correct SHA-256."""
        chain = ForensicHashChain("TEST-003")
        data = b"test data"
        digest = chain.hash_bytes(data, "test_step")

        expected = hashlib.sha256(data).hexdigest()
        assert digest == expected
        assert chain.chain[-1]["label"] == "test_step"

    def test_hash_result(self):
        """hash_result is deterministic for same input."""
        chain = ForensicHashChain("TEST-004")
        result = {"verdict": "AUTHENTIC", "score": 0.15}

        hash1 = chain.hash_result(result)
        hash2 = chain.hash_result(result)
        assert hash1 == hash2

    def test_hash_result_different_for_different_input(self):
        """hash_result differs for different results."""
        chain = ForensicHashChain("TEST-005")
        hash1 = chain.hash_result({"verdict": "AUTHENTIC"})
        hash2 = chain.hash_result({"verdict": "MANIPULATED"})
        assert hash1 != hash2

    def test_hash_frames(self):
        """hash_frames hashes first, middle, and last frames."""
        chain = ForensicHashChain("TEST-006")
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(10)]
        digest = chain.hash_frames(frames)

        assert len(digest) == 64  # SHA-256 hex
        assert chain.chain[-1]["label"] == "extracted_frames"
        assert chain.chain[-1]["metadata"]["frame_count"] == 10

    def test_hash_frames_empty(self):
        """hash_frames handles empty list."""
        chain = ForensicHashChain("TEST-007")
        digest = chain.hash_frames([])
        assert len(digest) == 64

    def test_verify_success(self, tmp_path):
        """verify returns True for matching hash."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"verify me")

        chain = ForensicHashChain("TEST-008")
        digest = chain.hash_file(str(test_file))
        assert chain.verify("original_file", digest) is True

    def test_verify_failure(self, tmp_path):
        """verify returns False for mismatched hash."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"verify me")

        chain = ForensicHashChain("TEST-009")
        chain.hash_file(str(test_file))
        assert chain.verify("original_file", "0000" * 16) is False

    def test_verify_missing_label(self):
        """verify returns False for non-existent label."""
        chain = ForensicHashChain("TEST-010")
        assert chain.verify("nonexistent", "abc") is False

    def test_to_dict(self, tmp_path):
        """to_dict serialises the full chain."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"data")

        chain = ForensicHashChain("TEST-011")
        chain.hash_file(str(test_file))
        chain.hash_bytes(b"more", "step_2")

        d = chain.to_dict()
        assert d["session_id"] == "TEST-011"
        assert d["chain_length"] == 2
        assert len(d["entries"]) == 2

    def test_summary(self, tmp_path):
        """summary returns label->hash mapping."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"data")

        chain = ForensicHashChain("TEST-012")
        chain.hash_file(str(test_file))

        summary = chain.summary()
        assert "original_file" in summary
        assert len(summary["original_file"]) == 64


class TestComputeFileHash:
    """Tests for the standalone compute_file_hash function."""

    def test_computes_correct_hash(self, tmp_path):
        """compute_file_hash matches manual SHA-256."""
        test_file = tmp_path / "test.bin"
        content = b"standalone hash test"
        test_file.write_bytes(content)

        result = compute_file_hash(str(test_file))
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected
