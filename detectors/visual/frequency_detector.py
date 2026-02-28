"""
Scanner - Frequency Analyzer Detector Adapter (v5.1.0)
Wraps the FrequencyAnalyzer core module as a BaseDetector.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from typing import Set

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


class FrequencyDetector(BaseDetector):
    """Adapter for frequency-domain forensic analysis."""

    def __init__(self) -> None:
        self._core = None

    @property
    def name(self) -> str:
        return "Frequency Analyzer"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    @property
    def version(self) -> str:
        return "1.0.0"

    def _ensure_core(self) -> None:
        if self._core is None:
            from core.frequency_analyzer import FrequencyAnalyzer
            self._core = FrequencyAnalyzer()

    def initialize(self) -> None:
        self._ensure_core()

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        self._ensure_core()

        # Determine input
        if inp.frames and len(inp.frames) > 0:
            result = self._core.analyze(inp.frames)
        elif inp.image is not None:
            result = self._core.analyze([inp.image])
        else:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No image or frames provided"},
            )

        status = DetectorStatus.PASS
        if result.score > 0.65:
            status = DetectorStatus.FAIL
        elif result.score > 0.30:
            status = DetectorStatus.WARN

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=result.score,
            confidence=result.confidence,
            status=status,
            details=result.details,
            anomalies=result.anomalies,
            data_quality="GOOD" if result.confidence > 0.5 else "LIMITED",
        )
