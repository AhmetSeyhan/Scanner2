"""
Unit tests for ForensicReporter (PDF generation).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forensic_types import FusionVerdict, TransparencyReport


class TestForensicReporter:
    """Test suite for ForensicReporter."""

    @pytest.fixture
    def reporter(self):
        try:
            from utils.forensic_reporter import ForensicReporter
            return ForensicReporter()
        except ImportError:
            pytest.skip("reportlab not installed")

    @pytest.fixture
    def sample_verdict(self):
        return FusionVerdict(
            verdict="AUTHENTIC", integrity_score=85.0, confidence=0.82,
            biosignal_score=0.15, artifact_score=0.10, alignment_score=0.12,
            weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
            leading_core="alignment",
            transparency_report=TransparencyReport(
                summary="All forensic checks passed.",
                biosignal_explanation="Normal biological signals",
                artifact_explanation="No generative fingerprints",
                alignment_explanation="Audio-visual aligned",
                environmental_factors=["HD resolution: full analysis"],
                supporting_evidence=[]
            )
        )

    @pytest.fixture
    def manipulated_verdict(self):
        return FusionVerdict(
            verdict="MANIPULATED", integrity_score=22.0, confidence=0.78,
            biosignal_score=0.80, artifact_score=0.85, alignment_score=0.72,
            weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
            leading_core="artifact",
            transparency_report=TransparencyReport(
                summary="Manipulation detected.",
                biosignal_explanation="Absent biological signals",
                artifact_explanation="GAN fingerprints detected",
                alignment_explanation="Lip sync mismatch",
                primary_concern="Generative model fingerprints detected",
                supporting_evidence=["GAN_GRID_ARTIFACT", "LIP_SYNC_MISMATCH"]
            )
        )

    def test_generate_authentic_report(self, reporter, sample_verdict):
        """Test PDF generation for authentic verdict."""
        pdf_bytes = reporter.generate_report(
            verdict=sample_verdict,
            filename="test_video.mp4",
            sha256_hash="abc123def456789",
            resolution="1080p",
            duration=30.0,
            session_id="TEST-001"
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100
        assert pdf_bytes[:4] == b'%PDF'

    def test_generate_manipulated_report(self, reporter, manipulated_verdict):
        """Test PDF generation for manipulated verdict."""
        pdf_bytes = reporter.generate_report(
            verdict=manipulated_verdict,
            filename="fake_video.mp4",
            sha256_hash="xyz789",
            resolution="720p",
            duration=15.0,
            session_id="TEST-002"
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b'%PDF'

    def test_generate_report_from_dict(self, reporter):
        """Test PDF generation from dictionary."""
        result = {
            "verdict": "UNCERTAIN",
            "integrity_score": 55.0,
            "confidence": 0.45
        }
        pdf_bytes = reporter.generate_report_from_dict(
            result=result,
            filename="uncertain.mp4",
            sha256_hash="hash123",
            resolution="480p",
            duration=10.0,
            session_id="TEST-003"
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b'%PDF'

    def test_get_status(self, reporter):
        """Test score to status conversion."""
        assert reporter._get_status(0.7) == "FAIL"
        assert reporter._get_status(0.4) == "WARN"
        assert reporter._get_status(0.2) == "PASS"

    def test_get_recommendations_authentic(self, reporter, sample_verdict):
        """Test recommendations for authentic verdict."""
        recs = reporter._get_recommendations(sample_verdict)
        assert len(recs) > 0
        assert any("genuine" in r.lower() or "manipulation" not in r.lower() for r in recs)

    def test_get_recommendations_manipulated(self, reporter, manipulated_verdict):
        """Test recommendations for manipulated verdict."""
        recs = reporter._get_recommendations(manipulated_verdict)
        assert len(recs) > 0
        assert any("not" in r.lower() or "investigate" in r.lower() for r in recs)
