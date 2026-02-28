"""
Scanner - Vision Transformer Deepfake Detector (v6.0.0)

Patch-based Vision Transformer for deepfake detection.
Divides input into 16x16 patches and uses self-attention to
detect inter-patch inconsistencies caused by manipulation.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from typing import List, Set

import numpy as np

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


class PatchEmbedding:
    """Extract and embed image patches for ViT processing."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 512):
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches from image."""
        import cv2
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        resized = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        h, w, c = resized.shape
        ph = pw = self.patch_size
        num_h, num_w = h // ph, w // pw

        patches = []
        for i in range(num_h):
            for j in range(num_w):
                patch = resized[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
                patches.append(patch.flatten())

        return np.array(patches, dtype=np.float32)

    def compute_attention_scores(self, patches: np.ndarray) -> np.ndarray:
        """Compute simplified self-attention between patches.

        Uses dot-product similarity as a proxy for learned attention.
        Real ViT would use multi-head self-attention with learned Q/K/V.
        """
        # Normalize patches
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
        normalized = patches / norms

        # Compute attention matrix (scaled dot-product)
        d_k = patches.shape[1]
        attention = normalized @ normalized.T / np.sqrt(d_k)

        # Softmax
        attention = np.exp(attention - np.max(attention, axis=1, keepdims=True))
        attention = attention / (np.sum(attention, axis=1, keepdims=True) + 1e-10)

        return attention


class ViTDetector(BaseDetector):
    """Vision Transformer-based deepfake detector.

    Analyzes inter-patch relationships using self-attention to detect
    spatial inconsistencies characteristic of face manipulation.
    """

    ATTENTION_UNIFORMITY_THRESHOLD = 0.85
    PATCH_VARIANCE_THRESHOLD = 0.01

    def __init__(self, patch_size: int = 16):
        self._embedder = PatchEmbedding(patch_size=patch_size)

    @property
    def name(self) -> str:
        return "ViT Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.SINGLE_IMAGE, DetectorCapability.VIDEO_FRAMES}

    @property
    def version(self) -> str:
        return "6.0.0"

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.image is not None:
            return self._analyze_image(inp.image)
        elif inp.frames and len(inp.frames) > 0:
            return self._analyze_frames(inp.frames)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=0.5,
            confidence=0.0,
            status=DetectorStatus.SKIPPED,
            data_quality="INSUFFICIENT",
        )

    def _analyze_image(self, image: np.ndarray) -> DetectorResult:
        patches = self._embedder.extract_patches(image)
        attention = self._embedder.compute_attention_scores(patches)

        # Analyze attention patterns
        attention_entropy = self._attention_entropy(attention)
        attention_uniformity = self._attention_uniformity(attention)
        patch_variance = float(np.var(np.mean(patches, axis=1)))

        # Boundary consistency (adjacent patches should be similar)
        boundary_score = self._boundary_consistency(patches)

        anomalies = []
        score_parts = []

        if attention_uniformity > self.ATTENTION_UNIFORMITY_THRESHOLD:
            anomalies.append("Overly uniform attention pattern - synthetic indicator")
            score_parts.append(0.7)

        if patch_variance < self.PATCH_VARIANCE_THRESHOLD:
            anomalies.append("Low inter-patch variance - possible generated content")
            score_parts.append(0.6)

        if boundary_score > 0.3:
            anomalies.append("Patch boundary discontinuities detected")
            score_parts.append(boundary_score)

        if score_parts:
            score = float(np.mean(score_parts))
        else:
            score = float(np.clip(0.3 * (1 - attention_entropy / 5.0) + 0.2 * boundary_score, 0.0, 1.0))

        confidence = min(0.5 + len(anomalies) * 0.15 + abs(score - 0.5) * 0.3, 1.0)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=float(np.clip(score, 0.0, 1.0)),
            confidence=confidence,
            status=DetectorStatus.PASS,
            anomalies=anomalies,
            details={
                "num_patches": len(patches),
                "attention_entropy": round(attention_entropy, 4),
                "attention_uniformity": round(attention_uniformity, 4),
                "patch_variance": round(patch_variance, 6),
                "boundary_score": round(boundary_score, 4),
            },
        )

    def _analyze_frames(self, frames: List[np.ndarray]) -> DetectorResult:
        max_sample = 6
        if len(frames) > max_sample:
            indices = np.linspace(0, len(frames) - 1, max_sample, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        results = [self._analyze_image(f) for f in sampled]
        scores = [r.score for r in results]

        avg_score = float(np.mean(scores))
        all_anomalies = list({a for r in results for a in r.anomalies})

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=float(np.clip(avg_score, 0.0, 1.0)),
            confidence=float(np.mean([r.confidence for r in results])),
            status=DetectorStatus.PASS,
            anomalies=all_anomalies,
            details={
                "frames_analyzed": len(sampled),
                "per_frame_scores": [round(s, 4) for s in scores],
            },
        )

    @staticmethod
    def _attention_entropy(attention: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        flat = attention.flatten()
        flat = flat[flat > 0]
        return float(-np.sum(flat * np.log(flat + 1e-10)))

    @staticmethod
    def _attention_uniformity(attention: np.ndarray) -> float:
        """Measure how uniform the attention is (1.0 = perfectly uniform)."""
        n = attention.shape[0]
        uniform = np.ones_like(attention) / n
        diff = np.abs(attention - uniform)
        return float(1.0 - np.mean(diff) * n)

    @staticmethod
    def _boundary_consistency(patches: np.ndarray) -> float:
        """Check consistency at patch boundaries."""
        n = int(np.sqrt(len(patches)))
        if n < 2:
            return 0.0

        diffs = []
        for i in range(n):
            for j in range(n - 1):
                idx = i * n + j
                if idx + 1 < len(patches):
                    diff = np.mean(np.abs(patches[idx] - patches[idx + 1]))
                    diffs.append(diff)

        if not diffs:
            return 0.0

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        # High std relative to mean indicates inconsistent boundaries
        if mean_diff < 1e-8:
            return 0.0
        return float(np.clip(std_diff / (mean_diff + 1e-8) - 0.5, 0.0, 1.0))
