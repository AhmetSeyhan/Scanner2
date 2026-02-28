"""
Unit tests for FUSION ENGINE.
Tests unified decision engine and verdict generation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forensic_types import (
    AlignmentCoreResult,
    ArtifactCoreResult,
    BioSignalCoreResult,
)
from core.fusion_engine import FusionEngine, FusionMode


class TestFusionEngine:
    """Test suite for FusionEngine."""

    @pytest.fixture
    def engine(self):
        return FusionEngine()

    @pytest.fixture
    def authentic_results(self):
        """Create sample core results indicating authentic media."""
        biosignal = BioSignalCoreResult(
            core_name="BIOSIGNAL CORE",
            score=0.2, confidence=0.8, status="PASS",
            details={}, anomalies=[], data_quality="GOOD",
            biological_sync_score=0.8, pulse_coverage=0.7, hr_consistency=0.9
        )
        artifact = ArtifactCoreResult(
            core_name="ARTIFACT CORE",
            score=0.15, confidence=0.7, status="PASS",
            details={}, anomalies=[], data_quality="GOOD",
            gan_score=0.1, diffusion_score=0.1, vae_score=0.1
        )
        alignment = AlignmentCoreResult(
            core_name="ALIGNMENT CORE",
            score=0.1, confidence=0.9, status="PASS",
            details={}, anomalies=[], data_quality="GOOD",
            av_alignment_score=0.9, phoneme_viseme_score=0.85
        )
        return biosignal, artifact, alignment

    @pytest.fixture
    def manipulated_results(self):
        """Create sample core results indicating manipulated media."""
        biosignal = BioSignalCoreResult(
            core_name="BIOSIGNAL CORE",
            score=0.8, confidence=0.7, status="FAIL",
            details={}, anomalies=["LOW_BIOLOGICAL_SYNC"], data_quality="GOOD",
            biological_sync_score=0.2, pulse_coverage=0.3, hr_consistency=0.4
        )
        artifact = ArtifactCoreResult(
            core_name="ARTIFACT CORE",
            score=0.85, confidence=0.8, status="FAIL",
            details={}, anomalies=["GAN_GRID_ARTIFACT_DETECTED"], data_quality="GOOD",
            gan_score=0.8, diffusion_score=0.1, vae_score=0.1
        )
        alignment = AlignmentCoreResult(
            core_name="ALIGNMENT CORE",
            score=0.7, confidence=0.6, status="FAIL",
            details={}, anomalies=["LIP_SYNC_MISMATCH"], data_quality="GOOD",
            av_alignment_score=0.3, phoneme_viseme_score=0.4
        )
        return biosignal, artifact, alignment

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.name == "FUSION ENGINE"
        assert engine.fusion_mode == FusionMode.WEIGHTED_AVERAGE

    def test_authentic_verdict(self, engine, authentic_results):
        """Test verdict for authentic media."""
        biosignal, artifact, alignment = authentic_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        assert verdict.verdict in ["AUTHENTIC", "UNCERTAIN"]
        assert verdict.integrity_score >= 50
        assert 0 <= verdict.confidence <= 1

    def test_manipulated_verdict(self, engine, manipulated_results):
        """Test verdict for manipulated media."""
        biosignal, artifact, alignment = manipulated_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        assert verdict.verdict in ["MANIPULATED", "UNCERTAIN"]
        assert verdict.integrity_score <= 50

    def test_weight_redistribution(self, engine):
        """Test weight redistribution for low confidence core."""
        biosignal = BioSignalCoreResult(
            core_name="BIOSIGNAL CORE",
            score=0.8, confidence=0.2,  # Low confidence
            status="FAIL", details={}, anomalies=[], data_quality="LIMITED"
        )
        artifact = ArtifactCoreResult(
            core_name="ARTIFACT CORE",
            score=0.3, confidence=0.8,
            status="WARN", details={}, anomalies=[], data_quality="GOOD"
        )
        alignment = AlignmentCoreResult(
            core_name="ALIGNMENT CORE",
            score=0.2, confidence=0.9,
            status="PASS", details={}, anomalies=[], data_quality="GOOD"
        )

        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        # Biosignal weight should be reduced due to low confidence
        assert verdict.weights["biosignal"] < 0.33

    def test_consensus_fail_detection(self, engine, manipulated_results):
        """Test consensus detection when cores agree on manipulation."""
        biosignal, artifact, alignment = manipulated_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        assert verdict.consensus_type == "CONSENSUS_FAIL"
        assert verdict.verdict == "MANIPULATED"

    def test_consensus_pass_detection(self, engine, authentic_results):
        """Test consensus detection when cores agree on authenticity."""
        biosignal, artifact, alignment = authentic_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        assert verdict.consensus_type in ["CONSENSUS_PASS", "DEFER_TO_ALIGNMENT"]

    def test_transparency_report(self, engine, authentic_results):
        """Test transparency report generation."""
        biosignal, artifact, alignment = authentic_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        assert verdict.transparency_report is not None
        assert verdict.transparency_report.summary != ""
        assert verdict.transparency_report.biosignal_explanation != ""

    def test_verdict_to_dict(self, engine, authentic_results):
        """Test verdict serialization."""
        biosignal, artifact, alignment = authentic_results
        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)
        verdict_dict = verdict.to_dict()

        assert "verdict" in verdict_dict
        assert "integrity_score" in verdict_dict
        assert "confidence" in verdict_dict
        assert "weights" in verdict_dict

    def test_conflicting_signals(self, engine):
        """Test handling of conflicting signals between cores."""
        biosignal = BioSignalCoreResult(
            core_name="BIOSIGNAL CORE",
            score=0.8, confidence=0.8, status="FAIL",
            details={}, anomalies=[], data_quality="GOOD"
        )
        artifact = ArtifactCoreResult(
            core_name="ARTIFACT CORE",
            score=0.1, confidence=0.8, status="PASS",
            details={}, anomalies=[], data_quality="GOOD"
        )
        alignment = AlignmentCoreResult(
            core_name="ALIGNMENT CORE",
            score=0.15, confidence=0.8, status="PASS",
            details={}, anomalies=[], data_quality="GOOD"
        )

        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)

        # Should detect conflicting signals
        assert verdict.conflicting_signals or verdict.verdict in ["UNCERTAIN", "INCONCLUSIVE"]

    def test_confidence_product_mode(self):
        """Test confidence product fusion mode."""
        engine = FusionEngine(fusion_mode=FusionMode.CONFIDENCE_PRODUCT)
        assert engine.fusion_mode == FusionMode.CONFIDENCE_PRODUCT

        biosignal = BioSignalCoreResult(
            core_name="BIOSIGNAL CORE",
            score=0.3, confidence=0.9, status="WARN",
            details={}, anomalies=[], data_quality="GOOD"
        )
        artifact = ArtifactCoreResult(
            core_name="ARTIFACT CORE",
            score=0.2, confidence=0.8, status="PASS",
            details={}, anomalies=[], data_quality="GOOD"
        )
        alignment = AlignmentCoreResult(
            core_name="ALIGNMENT CORE",
            score=0.25, confidence=0.85, status="PASS",
            details={}, anomalies=[], data_quality="GOOD"
        )

        verdict = engine.get_final_integrity_score(biosignal, artifact, alignment)
        assert verdict.fusion_method == "confidence_product"
