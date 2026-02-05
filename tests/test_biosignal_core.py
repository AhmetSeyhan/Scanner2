"""
Unit tests for BIOSIGNAL CORE.
Tests 32-ROI rPPG biological signal analysis.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.biosignal_core import BioSignalCore
from core.forensic_types import VideoProfile, ResolutionTier


class TestBioSignalCore:
    """Test suite for BioSignalCore."""

    @pytest.fixture
    def core(self):
        return BioSignalCore()

    @pytest.fixture
    def sample_frames(self):
        """Generate synthetic frames with face-like patterns."""
        frames = []
        for i in range(60):
            frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            # Add slight variation to simulate pulse
            variation = int(5 * np.sin(i * 0.2))
            frame[:, :, 1] = np.clip(frame[:, :, 1].astype(int) + variation, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames

    def test_core_initialization(self, core):
        """Test core initializes correctly."""
        assert core.name == "BIOSIGNAL CORE"
        assert core.ROI_GRID_ROWS == 8
        assert core.ROI_GRID_COLS == 4

    def test_roi_grid_generation(self, core, sample_frames):
        """Test 32 ROI regions are generated."""
        regions = core._generate_roi_grid(sample_frames[0].shape)
        assert len(regions) == 32

    def test_analyze_returns_result(self, core, sample_frames):
        """Test analyze returns valid result."""
        result = core.analyze(sample_frames, fps=30.0)

        assert result.core_name == "BIOSIGNAL CORE"
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1
        assert result.status in ["PASS", "WARN", "FAIL"]

    def test_analyze_insufficient_frames(self, core):
        """Test handling of insufficient frames."""
        short_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = core.analyze(short_frames, fps=30.0)

        assert "LOW_FRAME_COUNT" in result.anomalies
        assert result.data_quality == "INSUFFICIENT"

    def test_biological_sync_calculation(self, core, sample_frames):
        """Test biological sync score calculation."""
        result = core.analyze(sample_frames, fps=30.0)

        assert 0 <= result.biological_sync_score <= 1
        assert 0 <= result.pulse_coverage <= 1

    def test_low_res_handling(self, core):
        """Test handling of low resolution video."""
        low_res_frames = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(60)
        ]

        video_profile = VideoProfile(
            width=320, height=240, fps=30, frame_count=60,
            duration_seconds=2.0, resolution_tier=ResolutionTier.ULTRA_LOW,
            pixel_count=76800, aspect_ratio=1.33, rppg_viable=False,
            mesh_viable=False, recommended_analysis="ARTIFACT_FOCUSED"
        )

        result = core.analyze(low_res_frames, fps=30.0, video_profile=video_profile)

        # Low-res should have reduced confidence
        assert result.confidence <= 0.7
        assert result.data_quality in ["LIMITED", "INSUFFICIENT"]

    def test_result_to_dict(self, core, sample_frames):
        """Test result serialization."""
        result = core.analyze(sample_frames, fps=30.0)
        result_dict = result.to_dict()

        assert "core_name" in result_dict
        assert "score" in result_dict
        assert "confidence" in result_dict
        assert "status" in result_dict
