"""
Scanner - BioSignal Detector Adapter (v5.0.0)
Wraps the existing BioSignalCore as a BaseDetector.

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


class BioSignalDetector(BaseDetector):
    """Adapter that delegates to ``core.biosignal_core.BioSignalCore``."""

    def __init__(self) -> None:
        self._core = None

    # ---- properties ----

    @property
    def name(self) -> str:
        return "BioSignal Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.BIOLOGICAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.BIOLOGICAL_SIGNAL}

    # ---- lifecycle ----

    def _ensure_core(self) -> Any:
        if self._core is None:
            from core.biosignal_core import BioSignalCore
            self._core = BioSignalCore()
        return self._core

    def initialize(self) -> None:
        self._ensure_core()

    # ---- detection ----

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if not inp.frames or inp.fps <= 0:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No frames or fps provided"},
            )

        core = self._ensure_core()
        result = core.analyze(inp.frames, inp.fps, video_profile=inp.video_profile)

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
