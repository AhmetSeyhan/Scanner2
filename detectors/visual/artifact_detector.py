"""
Scanner - Artifact Detector Adapter (v5.0.0)
Wraps the existing ArtifactCore as a BaseDetector.

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


class ArtifactDetector(BaseDetector):
    """Adapter that delegates to ``core.artifact_core.ArtifactCore``."""

    def __init__(self) -> None:
        self._core = None

    @property
    def name(self) -> str:
        return "Artifact Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.VIDEO_FRAMES, DetectorCapability.SINGLE_IMAGE}

    def _ensure_core(self) -> Any:
        if self._core is None:
            from core.artifact_core import ArtifactCore
            self._core = ArtifactCore()
        return self._core

    def initialize(self) -> None:
        self._ensure_core()

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        frames = inp.frames
        if frames is None and inp.image is not None:
            frames = [inp.image]

        if not frames:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No frames or image provided"},
            )

        core = self._ensure_core()
        result = core.analyze(frames, video_profile=inp.video_profile)

        details = dict(result.details)
        if result.heatmap is not None and hasattr(result.heatmap, "to_dict"):
            details["heatmap"] = result.heatmap.to_dict()

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=result.score,
            confidence=result.confidence,
            status=DetectorStatus(result.status),
            details=details,
            anomalies=result.anomalies,
            data_quality=result.data_quality,
        )
