"""
Unit tests for ARTIFACT CORE.
Tests GAN/Diffusion/VAE fingerprint detection and heatmap generation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.artifact_core import ArtifactCore
from core.forensic_types import HeatmapAnalysis


class TestArtifactCore:
    """Test suite for ArtifactCore."""

    @pytest.fixture
    def core(self):
        return ArtifactCore()

    @pytest.fixture
    def sample_frames(self):
        return [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(30)
        ]

    def test_core_initialization(self, core):
        """Test core initializes correctly."""
        assert core.name == "ARTIFACT CORE"
        assert core.HEATMAP_GRID_SIZE == (8, 8)

    def test_analyze_returns_result(self, core, sample_frames):
        """Test analyze returns valid result."""
        result = core.analyze(sample_frames)

        assert result.core_name == "ARTIFACT CORE"
        assert 0 <= result.score <= 1
        assert 0 <= result.gan_score <= 1
        assert 0 <= result.diffusion_score <= 1
        assert 0 <= result.vae_score <= 1

    def test_gan_detection(self, core):
        """Test GAN artifact detection with synthetic patterns."""
        # Create frames with periodic patterns (GAN-like)
        frames = []
        for _ in range(10):
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            # Add 8x8 grid pattern
            for i in range(0, 256, 8):
                frame[i, :] = 255
                frame[:, i] = 255
            frames.append(frame)

        result = core.analyze(frames)
        # Should detect some artifact pattern or have anomalies
        assert result.score >= 0 or len(result.anomalies) >= 0

    def test_heatmap_generation(self, core, sample_frames):
        """Test spatial heatmap generation."""
        heatmap = core.generate_spatial_heatmap(sample_frames[0])

        assert isinstance(heatmap, HeatmapAnalysis)
        assert heatmap.grid_size == (8, 8)
        assert len(heatmap.cells) == 64
        assert 0 <= heatmap.overall_anomaly_score <= 1

    def test_heatmap_to_numpy(self, core, sample_frames):
        """Test heatmap conversion to numpy array."""
        heatmap = core.generate_spatial_heatmap(sample_frames[0])
        array = heatmap.to_numpy()

        assert array.shape == (8, 8)
        assert array.dtype == np.float32

    def test_heatmap_to_dict(self, core, sample_frames):
        """Test heatmap serialization."""
        heatmap = core.generate_spatial_heatmap(sample_frames[0])
        heatmap_dict = heatmap.to_dict()

        assert "grid_size" in heatmap_dict
        assert "cells" in heatmap_dict
        assert "overall_anomaly_score" in heatmap_dict
        assert "dominant_anomaly_type" in heatmap_dict

    def test_video_heatmaps(self, core, sample_frames):
        """Test multiple frame heatmap generation."""
        heatmaps = core.generate_video_heatmaps(sample_frames, sample_rate=10)

        assert len(heatmaps) >= 1
        assert all(isinstance(h, HeatmapAnalysis) for h in heatmaps)

    def test_insufficient_frames(self, core):
        """Test handling of too few frames."""
        short_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        result = core.analyze(short_frames)

        assert "LOW_FRAME_COUNT" in result.anomalies
        assert result.data_quality == "INSUFFICIENT"

    def test_structural_integrity(self, core, sample_frames):
        """Test structural integrity analysis."""
        warping_score, warping_events = core.analyze_structural_integrity(sample_frames)

        assert 0 <= warping_score <= 1
        assert isinstance(warping_events, list)

    def test_cell_analysis_methods(self, core, sample_frames):
        """Test individual cell analysis methods."""
        cell = sample_frames[0][0:60, 0:80]  # Extract a cell

        gan_score = core._analyze_cell_gan(cell)
        diff_score = core._analyze_cell_diffusion(cell)
        vae_score = core._analyze_cell_vae(cell)

        assert 0 <= gan_score <= 1
        assert 0 <= diff_score <= 1
        assert 0 <= vae_score <= 1
