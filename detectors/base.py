"""
Scanner - BaseDetector & Shared Types (v6.0.0)

Every detection engine (visual, audio, text, multimodal) inherits from
BaseDetector and communicates through DetectorResult / DetectorInput.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DetectorType(str, Enum):
    """Broad category a detector belongs to."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    BIOLOGICAL = "biological"
    PENTASHIELD = "pentashield"


class DetectorCapability(str, Enum):
    """Fine-grained capabilities a detector can declare."""
    VIDEO_FRAMES = "video_frames"
    SINGLE_IMAGE = "single_image"
    AUDIO_TRACK = "audio_track"
    TEXT_CONTENT = "text_content"
    AV_SYNC = "av_sync"
    BIOLOGICAL_SIGNAL = "biological_signal"
    ADVERSARIAL_IMMUNE = "adversarial_immune"
    ZERO_DAY = "zero_day"
    FORENSIC = "forensic"
    ACTIVE_PROBE = "active_probe"
    EDGE = "edge"


class DetectorStatus(str, Enum):
    """Outcome status for a single detection run."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """Standardized output every detector must return."""
    detector_name: str
    detector_type: DetectorType
    score: float                  # 0.0 (authentic) → 1.0 (manipulated)
    confidence: float             # 0.0 → 1.0
    status: DetectorStatus = DetectorStatus.PASS
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    data_quality: str = "GOOD"    # GOOD | LIMITED | INSUFFICIENT
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "detector_type": self.detector_type.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "details": self.details,
            "anomalies": self.anomalies,
            "data_quality": self.data_quality,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class DetectorInput:
    """Unified input payload for all detectors.

    Each detector picks the fields it needs and ignores the rest.
    """
    # Video-related
    frames: Optional[List[np.ndarray]] = None
    fps: float = 0.0
    video_path: Optional[str] = None
    video_profile: Optional[Any] = None   # core.forensic_types.VideoProfile

    # Audio-related
    audio_profile: Optional[Any] = None   # core.audio_analyzer.AudioProfile

    # Image-related
    image: Optional[np.ndarray] = None

    # Text-related
    text: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract base class for every Scanner detection engine.

    Subclasses MUST implement:
      - name (property)
      - detector_type (property)
      - capabilities (property)
      - detect(input) -> DetectorResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector identifier (e.g. 'BioSignal Detector')."""
        ...

    @property
    @abstractmethod
    def detector_type(self) -> DetectorType:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Set[DetectorCapability]:
        ...

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def enabled(self) -> bool:
        return True

    # ---- lifecycle ----

    def initialize(self) -> None:
        """Optional one-time setup (load models, warm caches)."""

    def shutdown(self) -> None:
        """Optional cleanup."""

    def health_check(self) -> Dict[str, Any]:
        """Return a health-check dict.  Override for richer status."""
        return {"name": self.name, "status": "ok", "enabled": self.enabled}

    # ---- detection ----

    @abstractmethod
    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        """Core detection logic.  Subclasses implement this."""
        ...

    def detect(self, inp: DetectorInput) -> DetectorResult:
        """Public entry-point: wraps _run_detection with timing and error handling."""
        start = time.perf_counter()
        try:
            result = self._run_detection(inp)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.ERROR,
                details={"error": str(exc)},
                data_quality="INSUFFICIENT",
                duration_ms=elapsed,
            )
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    # ---- dunder ----

    def __repr__(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities)
        return f"<{self.__class__.__name__} name={self.name!r} type={self.detector_type.value} caps=[{caps}]>"
