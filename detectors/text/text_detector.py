"""
Scanner - Text Detector Adapter (v5.0.0)
Wraps the existing TextCore as a BaseDetector.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Set

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


class TextDetector(BaseDetector):
    """Adapter that delegates to ``core.text_core.TextCore``."""

    def __init__(self) -> None:
        self._core = None

    @property
    def name(self) -> str:
        return "Text Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.TEXT

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.TEXT_CONTENT}

    def _ensure_core(self) -> Any:
        if self._core is None:
            from core.text_core import TextCore
            self._core = TextCore()
        return self._core

    def initialize(self) -> None:
        self._ensure_core()

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.text:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No text provided"},
            )

        core = self._ensure_core()
        result = core.analyze(inp.text)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=result.score,
            confidence=result.confidence,
            status=DetectorStatus(result.status),
            details=result.details,
            anomalies=result.anomalies,
            data_quality=result.data_quality,
        )
