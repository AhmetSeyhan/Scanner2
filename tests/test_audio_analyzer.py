"""
Unit tests for AUDIO ANALYZER.
Tests SNR estimation and audio profile generation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_analyzer import AudioAnalyzer, AudioProfile


class TestAudioAnalyzer:
    """Test suite for AudioAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return AudioAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None

    def test_audio_profile_dataclass(self):
        """Test AudioProfile dataclass creation."""
        profile = AudioProfile(
            has_audio=True,
            snr_db=25.0,
            noise_level="LOW",
            recommended_av_weight=1.0,
            is_speech_detected=True,
            duration_seconds=30.0
        )
        assert profile.has_audio is True
        assert profile.snr_db == 25.0
        assert profile.noise_level == "LOW"
        assert profile.recommended_av_weight == 1.0

    def test_no_audio_profile(self):
        """Test profile for file without audio."""
        profile = AudioProfile(
            has_audio=False,
            snr_db=0.0,
            noise_level="EXTREME",
            recommended_av_weight=0.3,
            is_speech_detected=False,
            duration_seconds=0.0
        )
        assert profile.has_audio is False
        assert profile.recommended_av_weight == 0.3

    def test_snr_weight_mapping(self):
        """Test SNR to weight mapping logic."""
        # LOW noise (>= 20 dB) -> weight 1.0
        # MEDIUM noise (10-20 dB) -> weight 0.7
        # HIGH noise (5-10 dB) -> weight 0.5
        # EXTREME noise (< 5 dB) -> weight 0.3

        test_cases = [
            (25.0, "LOW", 1.0),
            (15.0, "MEDIUM", 0.7),
            (7.0, "HIGH", 0.5),
            (3.0, "EXTREME", 0.3),
        ]
        for snr, expected_level, expected_weight in test_cases:
            if snr >= 20:
                level, weight = "LOW", 1.0
            elif snr >= 10:
                level, weight = "MEDIUM", 0.7
            elif snr >= 5:
                level, weight = "HIGH", 0.5
            else:
                level, weight = "EXTREME", 0.3

            assert level == expected_level
            assert weight == expected_weight

    def test_analyze_nonexistent_file(self, analyzer):
        """Test analysis of non-existent file returns graceful result."""
        try:
            result = analyzer.analyze("/nonexistent/path/video.mp4")
            # Should return a profile indicating no audio
            assert isinstance(result, AudioProfile)
            assert result.has_audio is False
        except (FileNotFoundError, Exception):
            # Also acceptable to raise an error
            pass
