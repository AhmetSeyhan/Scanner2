"""
Scanner Prime - Core Forensic Analysis Modules (v3.2.0)
Original Implementation by Scanner Prime Team based on Public Academic Research.

This package contains the three core forensic analysis engines and the unified
fusion engine for deepfake detection.

Modules:
- biosignal_core: BIOSIGNAL CORE - 32 ROI rPPG biological signal analysis
- artifact_core: ARTIFACT CORE - GAN/Diffusion/VAE fingerprint detection
- alignment_core: ALIGNMENT CORE - Phoneme-Viseme mapping and A/V alignment
- fusion_engine: FUSION ENGINE - Unified decision engine with conflict resolution
- audio_analyzer: Audio extraction and SNR analysis for adaptive weighting
- input_sanity_guard: INPUT SANITY GUARD - Adversarial input detection (v3.2.0)
- forensic_types: Shared dataclasses and type definitions

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

from core.forensic_types import (
    ROIRegion,
    BiologicalSignal,
    CoreResult,
    BioSignalCoreResult,
    ArtifactCoreResult,
    AlignmentCoreResult,
    FusionVerdict,
    VideoProfile,
    ResolutionTier,
    TransparencyReport,
    # v3.2.0 new types
    HeatmapCell,
    HeatmapAnalysis,
    SanityCheckResult,
    ScanHistoryEntry,
)

from core.biosignal_core import BioSignalCore
from core.artifact_core import ArtifactCore
from core.alignment_core import AlignmentCore
from core.fusion_engine import FusionEngine, FusionMode, create_fusion_verdict
from core.audio_analyzer import AudioAnalyzer, AudioProfile, analyze_video_audio
from core.input_sanity_guard import InputSanityGuard

__all__ = [
    # Types
    "ROIRegion",
    "BiologicalSignal",
    "CoreResult",
    "BioSignalCoreResult",
    "ArtifactCoreResult",
    "AlignmentCoreResult",
    "FusionVerdict",
    "VideoProfile",
    "ResolutionTier",
    "TransparencyReport",
    "AudioProfile",
    # v3.2.0 new types
    "HeatmapCell",
    "HeatmapAnalysis",
    "SanityCheckResult",
    "ScanHistoryEntry",
    # Core Modules
    "BioSignalCore",
    "ArtifactCore",
    "AlignmentCore",
    "FusionEngine",
    "AudioAnalyzer",
    "InputSanityGuard",
    # Utilities
    "FusionMode",
    "create_fusion_verdict",
    "analyze_video_audio",
]

__version__ = "3.2.0"
