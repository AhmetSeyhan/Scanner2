"""
Scanner Prime - Forensic Type Definitions
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements shared dataclasses and type definitions for all core
forensic modules using standard open-source libraries (numpy, dataclasses).
All algorithms are based on publicly available academic research and
mathematical principles.

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import numpy as np

# Forward reference for AudioProfile (defined in audio_analyzer.py)
if TYPE_CHECKING:
    from core.audio_analyzer import AudioProfile


class ResolutionTier(Enum):
    """Video resolution classification for adaptive analysis."""
    ULTRA_LOW = "ultra_low"  # <= 360p
    LOW = "low"              # 361p - 480p
    MEDIUM = "medium"        # 481p - 720p
    HIGH = "high"            # 721p - 1080p
    ULTRA_HIGH = "ultra_hd"  # > 1080p


@dataclass
class VideoProfile:
    """Complete video profile with resolution analysis."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    resolution_tier: ResolutionTier
    pixel_count: int
    aspect_ratio: float

    # Adaptive flags
    rppg_viable: bool
    mesh_viable: bool
    recommended_analysis: str

    @property
    def resolution_label(self) -> str:
        """Human-readable resolution label."""
        if self.height <= 360:
            return f"{self.height}p (Ultra-Low)"
        elif self.height <= 480:
            return f"{self.height}p (Low)"
        elif self.height <= 720:
            return f"{self.height}p (Medium)"
        elif self.height <= 1080:
            return f"{self.height}p (HD)"
        else:
            return f"{self.height}p (Ultra-HD)"

    @property
    def is_low_res(self) -> bool:
        """Check if video is low resolution."""
        return self.resolution_tier in [ResolutionTier.ULTRA_LOW, ResolutionTier.LOW]


@dataclass
class ROIRegion:
    """Region of Interest for rPPG analysis."""
    x1: int
    y1: int
    x2: int
    y2: int
    weight: float = 1.0  # Importance weight (cheeks higher for pulse)
    name: str = ""       # Optional region identifier

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract this ROI from a frame."""
        return frame[self.y1:self.y2, self.x1:self.x2]


@dataclass
class BiologicalSignal:
    """Biological signal extracted from an ROI."""
    roi_index: int
    roi_name: str
    signal: np.ndarray        # Time series signal
    quality: float            # Signal quality 0-1
    estimated_hr: float       # Estimated heart rate in BPM
    signal_strength: float    # FFT peak strength
    is_valid: bool            # Whether signal is usable

    @property
    def length(self) -> int:
        return len(self.signal)


@dataclass
class CoreResult:
    """Base result class for all core forensic modules."""
    core_name: str
    score: float              # 0.0 (authentic) to 1.0 (manipulated)
    confidence: float         # How confident the core is (0-1)
    status: str               # "PASS", "WARN", "FAIL"
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    data_quality: str = "GOOD"  # "GOOD", "LIMITED", "INSUFFICIENT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "core_name": self.core_name,
            "score": self.score,
            "confidence": self.confidence,
            "status": self.status,
            "details": self.details,
            "anomalies": self.anomalies,
            "data_quality": self.data_quality
        }


@dataclass
class BioSignalCoreResult(CoreResult):
    """Result from BIOSIGNAL CORE - Biological integrity analysis."""
    roi_signals: List[BiologicalSignal] = field(default_factory=list)
    biological_sync_score: float = 0.0
    cross_correlation_matrix: Optional[np.ndarray] = None
    pulse_coverage: float = 0.0  # % of ROIs with valid pulse
    hr_consistency: float = 0.0  # Consistency of HR across ROIs

    def __post_init__(self):
        self.core_name = "BIOSIGNAL CORE"


@dataclass
class ArtifactCoreResult(CoreResult):
    """Result from ARTIFACT CORE - Generative model forensics."""
    gan_score: float = 0.0
    diffusion_score: float = 0.0
    vae_score: float = 0.0
    structural_integrity: float = 0.0
    detected_fingerprints: List[Dict[str, Any]] = field(default_factory=list)
    temporal_warping_detected: bool = False
    # v3.2.0: Spatial anomaly heatmap for visualization
    heatmap: Optional[Any] = None  # HeatmapAnalysis (Any to avoid forward reference issues)

    def __post_init__(self):
        self.core_name = "ARTIFACT CORE"


@dataclass
class AlignmentCoreResult(CoreResult):
    """Result from ALIGNMENT CORE - Multimodal cross-check."""
    av_alignment_score: float = 0.0
    phoneme_viseme_score: float = 0.0
    lip_closure_events: List[Dict[str, Any]] = field(default_factory=list)
    speech_rhythm_score: float = 0.0
    metadata_integrity: float = 0.0

    def __post_init__(self):
        self.core_name = "ALIGNMENT CORE"


@dataclass
class TransparencyReport:
    """
    Detailed explanation of the verdict for transparency and auditability.

    Provides per-engine explanations, environmental adjustments made,
    and the primary concern that drove the verdict.
    """
    # Short summary of the verdict reason
    summary: str

    # Per-engine explanations
    biosignal_explanation: str
    artifact_explanation: str
    alignment_explanation: str

    # Environmental factors that affected weight distribution
    environmental_factors: List[str] = field(default_factory=list)

    # Primary concern that drove the verdict (if MANIPULATED)
    primary_concern: Optional[str] = None

    # Supporting evidence (anomaly flags that contributed)
    supporting_evidence: List[str] = field(default_factory=list)

    # Weight justifications (why each weight was set)
    weight_justification: Dict[str, str] = field(default_factory=dict)

    # Fusion method used
    fusion_method: str = "weighted_average"

    # Audio quality impact
    audio_quality_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary,
            "biosignal_explanation": self.biosignal_explanation,
            "artifact_explanation": self.artifact_explanation,
            "alignment_explanation": self.alignment_explanation,
            "environmental_factors": self.environmental_factors,
            "primary_concern": self.primary_concern,
            "supporting_evidence": self.supporting_evidence,
            "weight_justification": self.weight_justification,
            "fusion_method": self.fusion_method,
            "audio_quality_note": self.audio_quality_note
        }


@dataclass
class FusionVerdict:
    """Final verdict from the FUSION ENGINE."""
    verdict: str              # "AUTHENTIC", "UNCERTAIN", "MANIPULATED", "INCONCLUSIVE"
    integrity_score: float    # 0-100 scale
    confidence: float         # 0-1 scale

    # Component scores
    biosignal_score: float
    artifact_score: float
    alignment_score: float

    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)

    # Decision explanation
    consensus_type: str = ""  # "CONSENSUS_FAIL", "CONSENSUS_PASS", "DEFER_TO_X", "INCONCLUSIVE"
    reason: str = ""

    # Component results
    biosignal_result: Optional[BioSignalCoreResult] = None
    artifact_result: Optional[ArtifactCoreResult] = None
    alignment_result: Optional[AlignmentCoreResult] = None

    # Meta
    leading_core: str = ""    # Which core drove the verdict
    conflicting_signals: bool = False

    # NEW: Transparency report for detailed explanations
    transparency_report: Optional[TransparencyReport] = None

    # NEW: Audio profile used in weight calculation
    audio_profile: Optional[Any] = None  # AudioProfile (Any to avoid circular import)

    # NEW: Fusion method used
    fusion_method: str = "weighted_average"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "verdict": self.verdict,
            "integrity_score": self.integrity_score,
            "confidence": self.confidence,
            "biosignal_score": self.biosignal_score,
            "artifact_score": self.artifact_score,
            "alignment_score": self.alignment_score,
            "weights": self.weights,
            "consensus_type": self.consensus_type,
            "reason": self.reason,
            "leading_core": self.leading_core,
            "conflicting_signals": self.conflicting_signals,
            "fusion_method": self.fusion_method,
            "biosignal_result": self.biosignal_result.to_dict() if self.biosignal_result else None,
            "artifact_result": self.artifact_result.to_dict() if self.artifact_result else None,
            "alignment_result": self.alignment_result.to_dict() if self.alignment_result else None,
        }

        # Add transparency report if available
        if self.transparency_report:
            result["transparency"] = self.transparency_report.to_dict()

        # Add audio profile if available
        if self.audio_profile and hasattr(self.audio_profile, 'to_dict'):
            result["audio_profile"] = self.audio_profile.to_dict()

        return result


# =============================================================================
# v3.2.0 NEW DATACLASSES
# =============================================================================

@dataclass
class HeatmapCell:
    """Single cell in the spatial anomaly heatmap."""
    x: int                    # Grid x position
    y: int                    # Grid y position
    anomaly_score: float      # 0.0 (clean) to 1.0 (manipulated)
    anomaly_type: str         # "GAN", "DIFFUSION", "VAE", "WARPING", "NONE"
    confidence: float         # Detection confidence
    pixel_coords: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)


@dataclass
class HeatmapAnalysis:
    """Complete spatial anomaly heatmap result for forensic visualization."""
    grid_size: Tuple[int, int]           # (rows, cols)
    cells: List[HeatmapCell] = field(default_factory=list)
    overall_anomaly_score: float = 0.0   # Aggregated score
    hotspot_regions: List[Dict[str, Any]] = field(default_factory=list)
    dominant_anomaly_type: str = "NONE"  # Most common type
    frame_index: int = 0                 # Frame this heatmap represents

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for visualization."""
        rows, cols = self.grid_size
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        for cell in self.cells:
            if 0 <= cell.y < rows and 0 <= cell.x < cols:
                heatmap[cell.y, cell.x] = cell.anomaly_score
        return heatmap

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "grid_size": self.grid_size,
            "cells": [
                {"x": c.x, "y": c.y, "score": c.anomaly_score,
                 "type": c.anomaly_type, "confidence": c.confidence}
                for c in self.cells
            ],
            "overall_anomaly_score": self.overall_anomaly_score,
            "hotspot_regions": self.hotspot_regions,
            "dominant_anomaly_type": self.dominant_anomaly_type,
            "frame_index": self.frame_index
        }


@dataclass
class SanityCheckResult:
    """Result from InputSanityGuard validation."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    frame_consistency_score: float = 0.0    # 0-1, how consistent frames are
    resolution_consistency: bool = True     # True if resolution is consistent
    content_integrity_score: float = 1.0    # 0-1, adversarial detection score
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "frame_consistency_score": self.frame_consistency_score,
            "resolution_consistency": self.resolution_consistency,
            "content_integrity_score": self.content_integrity_score,
            "rejection_reason": self.rejection_reason
        }


@dataclass
class ScanHistoryEntry:
    """Single entry in scan history for analytics."""
    id: int
    session_id: str
    filename: str
    verdict: str
    integrity_score: float
    biosignal_score: float
    artifact_score: float
    alignment_score: float
    resolution: str
    duration_seconds: float
    timestamp: datetime
    sha256_hash: str
    user: str = "anonymous"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "verdict": self.verdict,
            "integrity_score": self.integrity_score,
            "biosignal_score": self.biosignal_score,
            "artifact_score": self.artifact_score,
            "alignment_score": self.alignment_score,
            "resolution": self.resolution,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sha256_hash": self.sha256_hash,
            "user": self.user
        }


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Type aliases for convenience
FrameList = List[np.ndarray]
SignalArray = np.ndarray
CorrelationMatrix = np.ndarray
