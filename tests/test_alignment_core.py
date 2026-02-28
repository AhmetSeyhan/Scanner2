"""
Unit tests for ALIGNMENT CORE.
Tests Phoneme-Viseme mapping and A/V synchronization.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alignment_core import AlignmentCore


class TestAlignmentCore:
    """Test suite for AlignmentCore."""

    @pytest.fixture
    def core(self):
        return AlignmentCore()

    @pytest.fixture
    def sample_frames(self):
        return [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(30)
        ]

    def test_core_initialization(self, core):
        """Test core initializes correctly."""
        assert core.name == "ALIGNMENT CORE"

    def test_analyze_returns_result(self, core, sample_frames):
        """Test analyze returns valid result."""
        result = core.analyze(sample_frames, fps=30.0)

        assert result.core_name == "ALIGNMENT CORE"
        assert 0 <= result.score <= 1
        assert 0 <= result.av_alignment_score <= 1
        assert 0 <= result.phoneme_viseme_score <= 1

    def test_lip_closure_detection(self, core, sample_frames):
        """Test lip closure event detection."""
        result = core.analyze(sample_frames, fps=30.0)

        # Should return a list (possibly empty)
        assert isinstance(result.lip_closure_events, list)

    def test_metadata_integrity(self, core, sample_frames):
        """Test metadata integrity analysis."""
        result = core.analyze(sample_frames, fps=30.0)

        assert 0 <= result.metadata_integrity <= 1

    def test_speech_rhythm_score(self, core, sample_frames):
        """Test speech rhythm analysis."""
        result = core.analyze(sample_frames, fps=30.0)

        assert 0 <= result.speech_rhythm_score <= 1

    def test_insufficient_frames(self, core):
        """Test handling of insufficient frames."""
        short_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = core.analyze(short_frames, fps=30.0)

        assert result.data_quality in ["LIMITED", "INSUFFICIENT"]

    def test_result_to_dict(self, core, sample_frames):
        """Test result serialization."""
        result = core.analyze(sample_frames, fps=30.0)
        result_dict = result.to_dict()

        assert "core_name" in result_dict
        assert "score" in result_dict
        assert "confidence" in result_dict
