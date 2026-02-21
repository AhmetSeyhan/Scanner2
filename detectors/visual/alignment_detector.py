"""
Scanner - Alignment Detector Adapter (v5.0.0)
Wraps the existing AlignmentCore as a BaseDetector.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


class AlignmentDetector(BaseDetector):
    """Adapter that delegates to ``core.alignment_core.AlignmentCore``."""

    def __init__(self) -> None:
        self._core = None

    @property
    def name(self) -> str:
        return "Alignment Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.MULTIMODAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.AV_SYNC}

    def _ensure_core(self) -> Any:
        if self._core is None:
            from core.alignment_core import AlignmentCore
            self._core = AlignmentCore()
        return self._core

    def initialize(self) -> None:
        self._ensure_core()

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
        result = core.analyze(
            inp.frames,
            inp.fps,
            video_path=inp.video_path,
            video_profile=inp.video_profile,
        )

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
