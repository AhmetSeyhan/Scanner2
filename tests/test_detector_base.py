"""
Tests for the BaseDetector framework, DetectorResult, DetectorInput,
enums, and the DetectorRegistry.
"""

import pytest
from typing import Set

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)
from detectors.registry import DetectorRegistry


# ---------------------------------------------------------------------------
# Concrete stub for testing
# ---------------------------------------------------------------------------

class StubDetector(BaseDetector):
    """Minimal concrete detector used in tests."""

    def __init__(self, *, score: float = 0.2, should_raise: bool = False) -> None:
        self._score = score
        self._should_raise = should_raise

    @property
    def name(self) -> str:
        return "Stub Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if self._should_raise:
            raise RuntimeError("intentional test error")
        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=self._score,
            confidence=0.9,
            status=DetectorStatus.PASS,
        )


# ---------------------------------------------------------------------------
# DetectorResult tests
# ---------------------------------------------------------------------------

class TestDetectorResult:

    def test_to_dict_keys(self):
        r = DetectorResult(
            detector_name="test",
            detector_type=DetectorType.VISUAL,
            score=0.42,
            confidence=0.88,
        )
        d = r.to_dict()
        assert d["detector_name"] == "test"
        assert d["detector_type"] == "visual"
        assert d["score"] == 0.42
        assert d["confidence"] == 0.88
        assert d["status"] == "PASS"

    def test_to_dict_rounds_values(self):
        r = DetectorResult(
            detector_name="t",
            detector_type=DetectorType.TEXT,
            score=0.123456789,
            confidence=0.987654321,
            duration_ms=123.456789,
        )
        d = r.to_dict()
        assert d["score"] == 0.1235
        assert d["confidence"] == 0.9877
        assert d["duration_ms"] == 123.46


# ---------------------------------------------------------------------------
# DetectorInput tests
# ---------------------------------------------------------------------------

class TestDetectorInput:

    def test_defaults_are_none(self):
        inp = DetectorInput()
        assert inp.frames is None
        assert inp.image is None
        assert inp.text is None
        assert inp.fps == 0.0

    def test_text_input(self):
        inp = DetectorInput(text="hello world")
        assert inp.text == "hello world"


# ---------------------------------------------------------------------------
# BaseDetector tests
# ---------------------------------------------------------------------------

class TestBaseDetector:

    def test_detect_returns_result(self):
        det = StubDetector(score=0.1)
        result = det.detect(DetectorInput())
        assert isinstance(result, DetectorResult)
        assert result.score == 0.1
        assert result.duration_ms > 0

    def test_detect_handles_exception(self):
        det = StubDetector(should_raise=True)
        result = det.detect(DetectorInput())
        assert result.status == DetectorStatus.ERROR
        assert result.confidence == 0.0
        assert "error" in result.details

    def test_repr(self):
        det = StubDetector()
        r = repr(det)
        assert "StubDetector" in r
        assert "visual" in r

    def test_health_check(self):
        det = StubDetector()
        h = det.health_check()
        assert h["name"] == "Stub Detector"
        assert h["status"] == "ok"


# ---------------------------------------------------------------------------
# DetectorRegistry tests
# ---------------------------------------------------------------------------

class TestDetectorRegistry:

    def setup_method(self):
        DetectorRegistry.reset()

    def test_register_and_get(self):
        reg = DetectorRegistry.get_instance()
        det = StubDetector()
        reg.register(det)
        assert reg.get("Stub Detector") is det
        assert len(reg) == 1

    def test_unregister(self):
        reg = DetectorRegistry.get_instance()
        reg.register(StubDetector())
        reg.unregister("Stub Detector")
        assert reg.get("Stub Detector") is None
        assert len(reg) == 0

    def test_get_by_type(self):
        reg = DetectorRegistry.get_instance()
        reg.register(StubDetector())
        assert len(reg.get_by_type(DetectorType.VISUAL)) == 1
        assert len(reg.get_by_type(DetectorType.TEXT)) == 0

    def test_get_by_capability(self):
        reg = DetectorRegistry.get_instance()
        reg.register(StubDetector())
        assert len(reg.get_by_capability(DetectorCapability.VIDEO_FRAMES)) == 1
        assert len(reg.get_by_capability(DetectorCapability.TEXT_CONTENT)) == 0

    def test_health_check_all(self):
        reg = DetectorRegistry.get_instance()
        reg.register(StubDetector())
        health = reg.health_check_all()
        assert "Stub Detector" in health

    def test_singleton(self):
        a = DetectorRegistry.get_instance()
        b = DetectorRegistry.get_instance()
        assert a is b
