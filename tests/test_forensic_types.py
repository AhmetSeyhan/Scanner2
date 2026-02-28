"""
Unit tests for forensic_types.py shared dataclasses.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from core.forensic_types import (
    AlignmentCoreResult,
    ArtifactCoreResult,
    BiologicalSignal,
    BioSignalCoreResult,
    CoreResult,
    FusionVerdict,
    HeatmapAnalysis,
    HeatmapCell,
    ResolutionTier,
    ROIRegion,
    SanityCheckResult,
    ScanHistoryEntry,
    TransparencyReport,
    VideoProfile,
)


class TestVideoProfile:
    def test_resolution_label_ultra_low(self):
        vp = VideoProfile(
            width=320, height=240, fps=30, frame_count=900,
            duration_seconds=30, resolution_tier=ResolutionTier.ULTRA_LOW,
            pixel_count=76800, aspect_ratio=4/3,
            rppg_viable=False, mesh_viable=False, recommended_analysis="BASIC"
        )
        assert "Ultra-Low" in vp.resolution_label

    def test_resolution_label_hd(self):
        vp = VideoProfile(
            width=1920, height=1080, fps=30, frame_count=900,
            duration_seconds=30, resolution_tier=ResolutionTier.HIGH,
            pixel_count=1920*1080, aspect_ratio=16/9,
            rppg_viable=True, mesh_viable=True, recommended_analysis="FULL"
        )
        assert "HD" in vp.resolution_label

    def test_is_low_res(self):
        vp = VideoProfile(
            width=640, height=480, fps=30, frame_count=900,
            duration_seconds=30, resolution_tier=ResolutionTier.LOW,
            pixel_count=640*480, aspect_ratio=4/3,
            rppg_viable=True, mesh_viable=False, recommended_analysis="BASIC"
        )
        assert vp.is_low_res is True

    def test_is_not_low_res(self):
        vp = VideoProfile(
            width=1280, height=720, fps=30, frame_count=900,
            duration_seconds=30, resolution_tier=ResolutionTier.MEDIUM,
            pixel_count=1280*720, aspect_ratio=16/9,
            rppg_viable=True, mesh_viable=True, recommended_analysis="FULL"
        )
        assert vp.is_low_res is False


class TestROIRegion:
    def test_properties(self):
        roi = ROIRegion(x1=10, y1=20, x2=110, y2=120, weight=1.2, name="cheek")
        assert roi.width == 100
        assert roi.height == 100
        assert roi.area == 10000

    def test_extract_from_frame(self):
        roi = ROIRegion(x1=0, y1=0, x2=50, y2=50)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        extracted = roi.extract_from_frame(frame)
        assert extracted.shape == (50, 50, 3)


class TestBiologicalSignal:
    def test_length(self):
        signal = BiologicalSignal(
            roi_index=0, roi_name="forehead",
            signal=np.zeros(100), quality=0.8,
            estimated_hr=72.0, signal_strength=0.5, is_valid=True
        )
        assert signal.length == 100


class TestCoreResult:
    def test_to_dict(self):
        result = CoreResult(
            core_name="TEST", score=0.5, confidence=0.8,
            status="WARN", anomalies=["TEST_ANOMALY"]
        )
        d = result.to_dict()
        assert d["core_name"] == "TEST"
        assert d["score"] == 0.5
        assert "TEST_ANOMALY" in d["anomalies"]

    def test_biosignal_result_name(self):
        result = BioSignalCoreResult(
            core_name="", score=0.3, confidence=0.7, status="PASS"
        )
        assert result.core_name == "BIOSIGNAL CORE"

    def test_artifact_result_name(self):
        result = ArtifactCoreResult(
            core_name="", score=0.4, confidence=0.6, status="WARN"
        )
        assert result.core_name == "ARTIFACT CORE"

    def test_alignment_result_name(self):
        result = AlignmentCoreResult(
            core_name="", score=0.2, confidence=0.9, status="PASS"
        )
        assert result.core_name == "ALIGNMENT CORE"


class TestFusionVerdict:
    def test_to_dict(self):
        verdict = FusionVerdict(
            verdict="AUTHENTIC", integrity_score=85.0, confidence=0.82,
            biosignal_score=0.15, artifact_score=0.10, alignment_score=0.12,
            weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
            leading_core="alignment"
        )
        d = verdict.to_dict()
        assert d["verdict"] == "AUTHENTIC"
        assert d["integrity_score"] == 85.0
        assert "weights" in d

    def test_to_dict_with_transparency(self):
        report = TransparencyReport(
            summary="Test summary",
            biosignal_explanation="Bio ok",
            artifact_explanation="Art ok",
            alignment_explanation="Ali ok"
        )
        verdict = FusionVerdict(
            verdict="AUTHENTIC", integrity_score=85.0, confidence=0.82,
            biosignal_score=0.15, artifact_score=0.10, alignment_score=0.12,
            transparency_report=report
        )
        d = verdict.to_dict()
        assert "transparency" in d
        assert d["transparency"]["summary"] == "Test summary"


class TestHeatmapAnalysis:
    def test_to_numpy(self):
        cells = [
            HeatmapCell(x=0, y=0, anomaly_score=0.5, anomaly_type="GAN", confidence=0.8),
            HeatmapCell(x=1, y=0, anomaly_score=0.3, anomaly_type="NONE", confidence=0.9),
        ]
        heatmap = HeatmapAnalysis(grid_size=(2, 2), cells=cells)
        arr = heatmap.to_numpy()
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 0.5
        assert arr[0, 1] == 0.3

    def test_to_dict(self):
        heatmap = HeatmapAnalysis(
            grid_size=(8, 8), cells=[], overall_anomaly_score=0.2,
            dominant_anomaly_type="NONE"
        )
        d = heatmap.to_dict()
        assert d["grid_size"] == (8, 8)
        assert d["overall_anomaly_score"] == 0.2


class TestSanityCheckResult:
    def test_to_dict(self):
        result = SanityCheckResult(
            is_valid=True,
            checks_passed=["A", "B"],
            checks_failed=[],
            frame_consistency_score=0.9
        )
        d = result.to_dict()
        assert d["is_valid"] is True
        assert len(d["checks_passed"]) == 2


class TestScanHistoryEntry:
    def test_to_dict(self):
        entry = ScanHistoryEntry(
            id=1, session_id="abc-123", filename="test.mp4",
            verdict="AUTHENTIC", integrity_score=85.0,
            biosignal_score=0.15, artifact_score=0.10, alignment_score=0.12,
            resolution="1080p", duration_seconds=30.0,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            sha256_hash="abc123"
        )
        d = entry.to_dict()
        assert d["session_id"] == "abc-123"
        assert d["verdict"] == "AUTHENTIC"
        assert "2026" in d["timestamp"]


class TestTransparencyReport:
    def test_to_dict(self):
        report = TransparencyReport(
            summary="All clear",
            biosignal_explanation="Normal pulse",
            artifact_explanation="No artifacts",
            alignment_explanation="Synced",
            environmental_factors=["Low res"],
            primary_concern=None,
            supporting_evidence=["evidence1"]
        )
        d = report.to_dict()
        assert d["summary"] == "All clear"
        assert d["primary_concern"] is None
        assert len(d["environmental_factors"]) == 1
