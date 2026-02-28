"""Tests for blockchain-ready verdict ledger."""

import json
import os
import tempfile

import pytest

from utils.verdict_ledger import VerdictBlock, VerdictLedger


@pytest.fixture
def ledger():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    ledger = VerdictLedger(db_path=path)
    yield ledger
    os.unlink(path)


@pytest.fixture
def sample_result():
    return {
        "session_id": "TEST001",
        "verdict": "MANIPULATED",
        "integrity_score": 25.5,
        "core_scores": {"biosignal": 0.7, "artifact": 0.8, "alignment": 0.3},
        "weights": {"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
        "forensic_hashes": {"file_hash": "abc123def456"},
    }


class TestVerdictBlock:
    def test_compute_hash_deterministic(self):
        block = VerdictBlock(
            block_index=0, timestamp="2026-01-01T00:00:00",
            session_id="TEST", file_hash="abc", verdict="AUTHENTIC",
            integrity_score=90.0, core_scores={"bio": 0.1},
            weights_used={"bio": 0.33}, previous_hash="0" * 64,
        )
        h1 = block.compute_hash()
        h2 = block.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_different_data_different_hash(self):
        block1 = VerdictBlock(
            block_index=0, timestamp="2026-01-01T00:00:00",
            session_id="TEST1", file_hash="abc", verdict="AUTHENTIC",
            integrity_score=90.0, core_scores={}, weights_used={},
            previous_hash="0" * 64,
        )
        block2 = VerdictBlock(
            block_index=0, timestamp="2026-01-01T00:00:00",
            session_id="TEST2", file_hash="abc", verdict="MANIPULATED",
            integrity_score=20.0, core_scores={}, weights_used={},
            previous_hash="0" * 64,
        )
        assert block1.compute_hash() != block2.compute_hash()


class TestVerdictLedger:
    def test_append_and_verify(self, ledger, sample_result):
        block = ledger.append_verdict(sample_result)
        assert block.block_index == 0
        assert block.block_hash != ""
        assert block.previous_hash == "0" * 64

        verification = ledger.verify_chain()
        assert verification["valid"] is True
        assert verification["blocks"] == 1

    def test_chain_linking(self, ledger, sample_result):
        b1 = ledger.append_verdict(sample_result)
        sample_result["session_id"] = "TEST002"
        b2 = ledger.append_verdict(sample_result)

        assert b2.previous_hash == b1.block_hash
        assert b2.block_index == 1

        verification = ledger.verify_chain()
        assert verification["valid"] is True
        assert verification["blocks"] == 2

    def test_multiple_blocks(self, ledger, sample_result):
        for i in range(5):
            sample_result["session_id"] = f"TEST{i:03d}"
            ledger.append_verdict(sample_result)

        assert ledger.chain_length() == 5
        verification = ledger.verify_chain()
        assert verification["valid"] is True

    def test_get_block(self, ledger, sample_result):
        ledger.append_verdict(sample_result)
        block = ledger.get_block(0)
        assert block is not None
        assert block.session_id == "TEST001"

    def test_get_nonexistent_block(self, ledger):
        block = ledger.get_block(999)
        assert block is None

    def test_export_chain(self, ledger, sample_result):
        ledger.append_verdict(sample_result)
        export = ledger.export_chain()
        data = json.loads(export)
        assert "chain" in data
        assert "verification" in data
        assert data["verification"]["valid"] is True
        assert len(data["chain"]) == 1

    def test_empty_chain_valid(self, ledger):
        verification = ledger.verify_chain()
        assert verification["valid"] is True
        assert verification["blocks"] == 0

    def test_chain_length(self, ledger, sample_result):
        assert ledger.chain_length() == 0
        ledger.append_verdict(sample_result)
        assert ledger.chain_length() == 1
