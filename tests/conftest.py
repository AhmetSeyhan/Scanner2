"""
Scanner Test Suite - Shared Fixtures
Provides common test data and mocks used across all test modules.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure structured logging is configured for tests
os.environ.setdefault("SCANNER_LOG_JSON", "false")
os.environ.setdefault("SCANNER_LOG_LEVEL", "WARNING")
os.environ.setdefault("SCANNER_SECRET_KEY", "test-secret-key")
os.environ.setdefault("SCANNER_API_KEY", "test-api-key")
os.environ.setdefault("SCANNER_ADMIN_PASSWORD", "test-admin-pw")
os.environ.setdefault("SCANNER_VIEWER_PASSWORD", "test-viewer-pw")

from core.forensic_types import (
    AlignmentCoreResult,
    ArtifactCoreResult,
    BioSignalCoreResult,
    ResolutionTier,
    VideoProfile,
)


@pytest.fixture
def sample_frames_60():
    """Generate 60 synthetic video frames (480x640) with pulse-like variation."""
    frames = []
    for i in range(60):
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        variation = int(5 * np.sin(i * 0.2))
        frame[:, :, 1] = np.clip(
            frame[:, :, 1].astype(int) + variation, 0, 255
        ).astype(np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_frames_30():
    """Generate 30 synthetic video frames (480x640)."""
    frames = []
    for i in range(30):
        base = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        noise = np.random.randint(-5, 5, (480, 640, 3), dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise + i, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def low_res_frames():
    """Generate low-resolution frames (240x320)."""
    return [
        np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        for _ in range(60)
    ]


@pytest.fixture
def high_res_frames():
    """Generate high-resolution frames (1080x1920)."""
    return [
        np.random.randint(100, 200, (1080, 1920, 3), dtype=np.uint8)
        for _ in range(30)
    ]


@pytest.fixture
def video_profile_hd():
    """Standard HD video profile."""
    return VideoProfile(
        width=1920, height=1080, fps=30.0, frame_count=900,
        duration_seconds=30.0, resolution_tier=ResolutionTier.HIGH,
        pixel_count=1920 * 1080, aspect_ratio=16 / 9,
        rppg_viable=True, mesh_viable=True,
        recommended_analysis="PRIME HYBRID"
    )


@pytest.fixture
def video_profile_low():
    """Low-resolution video profile."""
    return VideoProfile(
        width=320, height=240, fps=30.0, frame_count=900,
        duration_seconds=30.0, resolution_tier=ResolutionTier.ULTRA_LOW,
        pixel_count=320 * 240, aspect_ratio=4 / 3,
        rppg_viable=False, mesh_viable=False,
        recommended_analysis="ARTIFACT_FOCUSED"
    )


@pytest.fixture
def authentic_core_results():
    """Core results indicating authentic media."""
    biosignal = BioSignalCoreResult(
        core_name="BIOSIGNAL CORE",
        score=0.15, confidence=0.85, status="PASS",
        details={}, anomalies=[], data_quality="GOOD",
        biological_sync_score=0.85, pulse_coverage=0.75, hr_consistency=0.90
    )
    artifact = ArtifactCoreResult(
        core_name="ARTIFACT CORE",
        score=0.10, confidence=0.80, status="PASS",
        details={"detected_model_type": "NONE"}, anomalies=[], data_quality="GOOD",
        gan_score=0.05, diffusion_score=0.08, vae_score=0.06,
        structural_integrity=0.92
    )
    alignment = AlignmentCoreResult(
        core_name="ALIGNMENT CORE",
        score=0.12, confidence=0.88, status="PASS",
        details={}, anomalies=[], data_quality="GOOD",
        av_alignment_score=0.90, phoneme_viseme_score=0.88,
        speech_rhythm_score=0.85
    )
    return biosignal, artifact, alignment


@pytest.fixture
def manipulated_core_results():
    """Core results indicating manipulated media."""
    biosignal = BioSignalCoreResult(
        core_name="BIOSIGNAL CORE",
        score=0.78, confidence=0.72, status="FAIL",
        details={}, anomalies=["LOW_BIOLOGICAL_SYNC", "HR_INCONSISTENCY"],
        data_quality="GOOD",
        biological_sync_score=0.22, pulse_coverage=0.30, hr_consistency=0.35
    )
    artifact = ArtifactCoreResult(
        core_name="ARTIFACT CORE",
        score=0.85, confidence=0.82, status="FAIL",
        details={"detected_model_type": "GAN"},
        anomalies=["GAN_GRID_ARTIFACT_DETECTED", "TEMPORAL_WARPING"],
        data_quality="GOOD",
        gan_score=0.82, diffusion_score=0.12, vae_score=0.08,
        structural_integrity=0.25, temporal_warping_detected=True
    )
    alignment = AlignmentCoreResult(
        core_name="ALIGNMENT CORE",
        score=0.72, confidence=0.65, status="FAIL",
        details={}, anomalies=["LIP_SYNC_MISMATCH"],
        data_quality="GOOD",
        av_alignment_score=0.28, phoneme_viseme_score=0.32,
        speech_rhythm_score=0.45
    )
    return biosignal, artifact, alignment
