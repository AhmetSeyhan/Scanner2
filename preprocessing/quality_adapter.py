"""
Scanner - Quality Adapter (v6.0.0)

Adaptive preprocessing that accounts for input quality degradation
from compression, low resolution, and noise. Adjusts detection
parameters and provides quality-normalized features.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class QualityLevel(str, Enum):
    """Input quality classification."""
    HIGH = "high"           # Minimal compression, high res
    MEDIUM = "medium"       # Standard web quality
    LOW = "low"             # Heavy compression, low res
    DEGRADED = "degraded"   # Severely degraded input


@dataclass
class QualityProfile:
    """Quality assessment of input media."""
    level: QualityLevel
    resolution_score: float     # 0-1, based on resolution
    compression_score: float    # 0-1, based on compression artifacts
    noise_score: float          # 0-1, based on noise level
    sharpness_score: float      # 0-1, based on edge sharpness
    overall_score: float        # Weighted combination
    recommended_detectors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class QualityAdapter:
    """Adaptive quality assessment and preprocessing.

    Assesses input quality and recommends which detectors will be
    most effective, adjusting sensitivity thresholds accordingly.
    """

    # Resolution tiers
    ULTRA_LOW_RES = 360
    LOW_RES = 480
    MEDIUM_RES = 720
    HIGH_RES = 1080

    def assess_quality(self, image: np.ndarray) -> QualityProfile:
        """Assess quality of a single image/frame."""
        resolution = self._resolution_score(image)
        compression = self._compression_score(image)
        noise = self._noise_score(image)
        sharpness = self._sharpness_score(image)

        overall = (resolution * 0.3 + compression * 0.25 +
                   noise * 0.2 + sharpness * 0.25)

        # Classify quality level
        if overall >= 0.7:
            level = QualityLevel.HIGH
        elif overall >= 0.5:
            level = QualityLevel.MEDIUM
        elif overall >= 0.3:
            level = QualityLevel.LOW
        else:
            level = QualityLevel.DEGRADED

        # Recommend detectors based on quality
        recommended = self._recommend_detectors(level, resolution, sharpness)

        return QualityProfile(
            level=level,
            resolution_score=resolution,
            compression_score=compression,
            noise_score=noise,
            sharpness_score=sharpness,
            overall_score=overall,
            recommended_detectors=recommended,
            details={
                "height": image.shape[0],
                "width": image.shape[1] if image.ndim > 1 else 0,
                "channels": image.shape[2] if image.ndim > 2 else 1,
            },
        )

    def adapt_frame(self, image: np.ndarray, profile: QualityProfile) -> np.ndarray:
        """Apply quality-adaptive preprocessing to a frame."""
        import cv2

        result = image.copy()

        if profile.level == QualityLevel.DEGRADED:
            # Aggressive denoising for degraded input
            result = cv2.bilateralFilter(result, 9, 75, 75)
        elif profile.level == QualityLevel.LOW:
            # Light denoising
            result = cv2.bilateralFilter(result, 5, 50, 50)

        # Adaptive histogram equalization for low-contrast images
        if profile.sharpness_score < 0.4:
            if result.ndim == 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(result)

        return result

    def _resolution_score(self, image: np.ndarray) -> float:
        """Score based on image resolution."""
        h = image.shape[0]
        if h >= self.HIGH_RES:
            return 1.0
        elif h >= self.MEDIUM_RES:
            return 0.75
        elif h >= self.LOW_RES:
            return 0.5
        elif h >= self.ULTRA_LOW_RES:
            return 0.3
        return 0.1

    def _compression_score(self, image: np.ndarray) -> float:
        """Estimate compression level via blockiness detection."""
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float64)
        h, w = gray.shape

        if h < 16 or w < 16:
            return 0.5

        # Block boundary discontinuity (8x8 JPEG blocks)
        block_size = 8
        h_diffs = []
        v_diffs = []

        for y in range(block_size, h - block_size, block_size):
            row_diff = np.mean(np.abs(gray[y, :] - gray[y-1, :]))
            h_diffs.append(row_diff)

        for x in range(block_size, w - block_size, block_size):
            col_diff = np.mean(np.abs(gray[:, x] - gray[:, x-1]))
            v_diffs.append(col_diff)

        if not h_diffs or not v_diffs:
            return 0.5

        blockiness = (np.mean(h_diffs) + np.mean(v_diffs)) / 2.0
        # Lower blockiness = better quality
        return float(np.clip(1.0 - blockiness / 20.0, 0.0, 1.0))

    def _noise_score(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance."""
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian))

        # Very high variance could indicate noise or genuine detail
        # Very low variance indicates smooth/blurry (possibly overcompressed)
        if variance < 10:
            return 0.3
        elif variance > 5000:
            return 0.5  # Could be noise

        return float(np.clip(variance / 2000.0, 0.0, 1.0))

    def _sharpness_score(self, image: np.ndarray) -> float:
        """Measure image sharpness using gradient magnitude."""
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        mean_grad = float(np.mean(magnitude))
        return float(np.clip(mean_grad / 50.0, 0.0, 1.0))

    @staticmethod
    def _recommend_detectors(
        level: QualityLevel,
        resolution: float,
        sharpness: float,
    ) -> List[str]:
        """Recommend detectors based on quality profile."""
        always = ["CLIP Detector", "ViT Detector", "Artifact Detector"]

        if level in (QualityLevel.HIGH, QualityLevel.MEDIUM):
            return always + ["BioSignal Detector", "Alignment Detector"]
        elif level == QualityLevel.LOW:
            return always  # Skip biological signal for low quality
        else:
            return ["CLIP Detector", "Artifact Detector"]  # Minimal set for degraded
