"""
Tests for the adapter wrappers that bridge existing cores to the BaseDetector interface.
"""

import pytest
import numpy as np

from detectors.base import DetectorInput, DetectorResult, DetectorStatus, DetectorType
from detectors.registry import DetectorRegistry


# ---------------------------------------------------------------------------
# BioSignalDetector
# ---------------------------------------------------------------------------

class TestBioSignalDetector:

    def test_skip_when_no_frames(self):
        from detectors.visual.biosignal_detector import BioSignalDetector
        det = BioSignalDetector()
        result = det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED

    def test_properties(self):
        from detectors.visual.biosignal_detector import BioSignalDetector
        det = BioSignalDetector()
        assert det.name == "BioSignal Detector"
        assert det.detector_type == DetectorType.BIOLOGICAL

    def test_detect_with_frames(self):
        from detectors.visual.biosignal_detector import BioSignalDetector
        det = BioSignalDetector()
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(60)]
        inp = DetectorInput(frames=frames, fps=30.0)
        result = det.detect(inp)
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# ArtifactDetector
# ---------------------------------------------------------------------------

class TestArtifactDetector:

    def test_skip_when_no_input(self):
        from detectors.visual.artifact_detector import ArtifactDetector
        det = ArtifactDetector()
        result = det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED

    def test_properties(self):
        from detectors.visual.artifact_detector import ArtifactDetector
        det = ArtifactDetector()
        assert det.name == "Artifact Detector"
        assert det.detector_type == DetectorType.VISUAL

    def test_detect_with_frames(self):
        from detectors.visual.artifact_detector import ArtifactDetector
        det = ArtifactDetector()
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        inp = DetectorInput(frames=frames, fps=30.0)
        result = det.detect(inp)
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# AlignmentDetector
# ---------------------------------------------------------------------------

class TestAlignmentDetector:

    def test_skip_when_no_frames(self):
        from detectors.visual.alignment_detector import AlignmentDetector
        det = AlignmentDetector()
        result = det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED

    def test_properties(self):
        from detectors.visual.alignment_detector import AlignmentDetector
        det = AlignmentDetector()
        assert det.name == "Alignment Detector"
        assert det.detector_type == DetectorType.MULTIMODAL


# ---------------------------------------------------------------------------
# TextDetector
# ---------------------------------------------------------------------------

class TestTextDetector:

    def test_skip_when_no_text(self):
        from detectors.text.text_detector import TextDetector
        det = TextDetector()
        result = det.detect(DetectorInput())
        assert result.status == DetectorStatus.SKIPPED

    def test_properties(self):
        from detectors.text.text_detector import TextDetector
        det = TextDetector()
        assert det.name == "Text Detector"
        assert det.detector_type == DetectorType.TEXT

    def test_detect_with_text(self):
        from detectors.text.text_detector import TextDetector
        det = TextDetector()
        sample = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sufficiently long piece of sample text that should allow "
            "the statistical analysis routines to execute properly without failing "
            "due to insufficient input length constraints in the TextCore module."
        )
        result = det.detect(DetectorInput(text=sample))
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Registry default registration
# ---------------------------------------------------------------------------

class TestRegistryDefaults:

    def setup_method(self):
        DetectorRegistry.reset()

    def test_register_defaults_populates_registry(self):
        reg = DetectorRegistry.get_instance()
        reg.register_defaults()
        # At minimum the text detector should always load (no heavy deps)
        names = reg.list_names()
        assert len(names) >= 1
        assert "Text Detector" in names
