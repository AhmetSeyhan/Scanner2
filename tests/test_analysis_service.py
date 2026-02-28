"""
Scanner Test Suite - Analysis Service Tests
Unit tests for services/analysis_service.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalysisService:
    """Tests for the AnalysisService orchestration layer."""

    def test_service_instantiation(self):
        """AnalysisService can be created without side effects."""
        from services.analysis_service import AnalysisService
        service = AnalysisService()
        assert service.profiler is not None

    @patch("services.analysis_service.ModelManager")
    def test_analyze_image_no_face(self, mock_mm_cls):
        """analyze_image raises NoFaceDetectedError when no face found."""
        from core.exceptions import NoFaceDetectedError
        from services.analysis_service import AnalysisService

        mock_mm = MagicMock()
        mock_mm.face_extractor.extract_primary_face.return_value = None
        mock_mm_cls.get_instance.return_value = mock_mm

        service = AnalysisService()

        # Create a valid black image
        import cv2
        _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))

        with pytest.raises(NoFaceDetectedError):
            service.analyze_image(buf.tobytes(), "test.jpg", "tester")

    @patch("services.analysis_service.ModelManager")
    def test_analyze_image_success(self, mock_mm_cls):
        """analyze_image returns result dict on success."""
        from services.analysis_service import AnalysisService

        mock_mm = MagicMock()
        mock_mm.face_extractor.extract_primary_face.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_mm.face_extractor.preprocess_for_model.return_value = np.zeros((1, 3, 224, 224))
        mock_mm.inference_engine.predict_single.return_value = (0.85, "FAKE")
        mock_mm_cls.get_instance.return_value = mock_mm

        service = AnalysisService()
        import cv2
        _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))

        result = service.analyze_image(buf.tobytes(), "test.jpg", "tester")
        assert result["verdict"] == "FAKE"
        assert result["analyzed_by"] == "tester"
        assert "fake_probability" in result


class TestVideoProfiler:
    """Tests for the VideoProfiler service."""

    def test_validate_extension_video_valid(self):
        """Valid video extensions pass."""
        from services.video_profiler import VideoProfiler
        profiler = VideoProfiler()
        assert profiler.validate_extension("video.mp4", "video") == ".mp4"
        assert profiler.validate_extension("video.avi", "video") == ".avi"

    def test_validate_extension_video_invalid(self):
        """Invalid extensions raise UnsupportedFileTypeError."""
        from core.exceptions import UnsupportedFileTypeError
        from services.video_profiler import VideoProfiler
        profiler = VideoProfiler()
        with pytest.raises(UnsupportedFileTypeError):
            profiler.validate_extension("malware.exe", "video")

    def test_validate_extension_image_valid(self):
        """Valid image extensions pass."""
        from services.video_profiler import VideoProfiler
        profiler = VideoProfiler()
        assert profiler.validate_extension("photo.jpg", "image") == ".jpg"
        assert profiler.validate_extension("photo.png", "image") == ".png"

    def test_get_resolution_tier(self):
        """Resolution tiers mapped correctly."""
        from core.forensic_types import ResolutionTier
        from services.video_profiler import get_resolution_tier

        assert get_resolution_tier(240) == ResolutionTier.ULTRA_LOW
        assert get_resolution_tier(480) == ResolutionTier.LOW
        assert get_resolution_tier(720) == ResolutionTier.MEDIUM
        assert get_resolution_tier(1080) == ResolutionTier.HIGH
        assert get_resolution_tier(2160) == ResolutionTier.ULTRA_HIGH
