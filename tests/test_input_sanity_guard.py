"""
Unit tests for INPUT SANITY GUARD.
Tests adversarial input detection and frame validation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.input_sanity_guard import InputSanityGuard


class TestInputSanityGuard:
    """Test suite for InputSanityGuard."""

    @pytest.fixture
    def guard(self):
        return InputSanityGuard()

    @pytest.fixture
    def valid_frames(self):
        """Generate valid video frames."""
        frames = []
        for i in range(30):
            # Simulate natural frame-to-frame changes
            base = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
            noise = np.random.randint(-5, 5, (480, 640, 3), dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise + i, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames

    @pytest.fixture
    def identical_frames(self):
        """Generate identical frames (suspicious)."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return [frame.copy() for _ in range(30)]

    def test_guard_initialization(self, guard):
        """Test guard initializes correctly."""
        assert guard.name == "INPUT SANITY GUARD"

    def test_validate_valid_frames(self, guard, valid_frames):
        """Test validation of valid frames."""
        result = guard.validate(valid_frames)

        assert result.is_valid
        assert "FRAME_SEQUENCE_VALID" in result.checks_passed
        assert "RESOLUTION_CONSISTENT" in result.checks_passed
        assert "CONTENT_INTEGRITY_OK" in result.checks_passed

    def test_validate_insufficient_frames(self, guard):
        """Test validation fails with insufficient frames."""
        short_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = guard.validate(short_frames)

        assert not result.is_valid
        assert "INSUFFICIENT_FRAMES" in result.checks_failed

    def test_validate_identical_frames(self, guard, identical_frames):
        """Test detection of identical frames."""
        result = guard.validate(identical_frames)

        # Should warn about nearly identical frames
        assert any("identical" in w.lower() for w in result.warnings)

    def test_resolution_consistency(self, guard):
        """Test resolution consistency check."""
        # Mixed resolution frames
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),  # Different resolution
        ]
        result = guard.validate(frames)

        assert not result.resolution_consistency
        assert "RESOLUTION_MISMATCH" in result.checks_failed

    def test_content_integrity_nan(self, guard):
        """Test detection of NaN values."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.float32),
            np.zeros((480, 640, 3), dtype=np.float32),
        ]
        frames[0][0, 0, 0] = np.nan

        result = guard.validate(frames)

        assert not result.is_valid
        assert "CONTENT_INTEGRITY_FAILED" in result.checks_failed

    def test_quick_check(self, guard, valid_frames):
        """Test quick validation check."""
        assert guard.quick_check(valid_frames)

    def test_quick_check_fails_insufficient(self, guard):
        """Test quick check fails with insufficient frames."""
        assert not guard.quick_check([np.zeros((480, 640, 3), dtype=np.uint8)])

    def test_result_to_dict(self, guard, valid_frames):
        """Test result serialization."""
        result = guard.validate(valid_frames)
        result_dict = result.to_dict()

        assert "is_valid" in result_dict
        assert "checks_passed" in result_dict
        assert "checks_failed" in result_dict
        assert "frame_consistency_score" in result_dict
