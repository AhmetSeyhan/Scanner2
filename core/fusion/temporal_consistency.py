"""
Scanner - Temporal Consistency Analyzer (v6.0.0)

Analyzes frame-to-frame temporal consistency of detection scores
and visual features. Real videos have smooth, physically-consistent
transitions, while deepfakes may exhibit temporal artifacts.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class TemporalConsistencyResult:
    """Result from temporal consistency analysis."""
    consistency_score: float      # 0.0 (inconsistent) -> 1.0 (consistent)
    temporal_smoothness: float    # Score smoothness over time
    flicker_score: float          # Amount of score flickering
    trend_stability: float        # Trend stability metric
    anomalies: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class TemporalConsistency:
    """Analyzes temporal consistency of detection results across frames.

    Deepfake videos often exhibit:
    - Score flickering between adjacent frames
    - Abrupt score transitions at manipulation boundaries
    - Periodic patterns from frame-by-frame generation
    """

    SMOOTHNESS_WINDOW = 5
    FLICKER_THRESHOLD = 0.15
    TREND_CHANGE_THRESHOLD = 0.3

    def analyze(self, frame_scores: List[float], fps: float = 30.0) -> TemporalConsistencyResult:
        """Analyze temporal consistency of per-frame scores."""
        if len(frame_scores) < 3:
            return TemporalConsistencyResult(
                consistency_score=0.5,
                temporal_smoothness=0.5,
                flicker_score=0.0,
                trend_stability=0.5,
            )

        scores = np.array(frame_scores, dtype=np.float64)

        smoothness = self._temporal_smoothness(scores)
        flicker = self._flicker_detection(scores)
        trend = self._trend_stability(scores)

        anomalies = []

        if smoothness < 0.4:
            anomalies.append("Low temporal smoothness - possible frame-level manipulation")
        if flicker > self.FLICKER_THRESHOLD:
            anomalies.append(f"Score flickering detected (flicker={flicker:.3f})")
        if trend < 0.3:
            anomalies.append("Unstable score trend - manipulation boundary suspected")

        # Periodic pattern detection
        periodic = self._detect_periodicity(scores)
        if periodic > 0.6:
            anomalies.append(f"Periodic score pattern detected (strength={periodic:.3f})")

        # Overall consistency: high is more authentic
        consistency = (smoothness * 0.4 + (1.0 - flicker) * 0.3 + trend * 0.3)
        consistency = float(np.clip(consistency, 0.0, 1.0))

        return TemporalConsistencyResult(
            consistency_score=consistency,
            temporal_smoothness=smoothness,
            flicker_score=flicker,
            trend_stability=trend,
            anomalies=anomalies,
            details={
                "num_frames": len(frame_scores),
                "fps": fps,
                "score_mean": round(float(np.mean(scores)), 4),
                "score_std": round(float(np.std(scores)), 4),
                "periodic_strength": round(periodic, 4),
            },
        )

    def _temporal_smoothness(self, scores: np.ndarray) -> float:
        """Measure how smoothly scores change over time."""
        diffs = np.abs(np.diff(scores))
        if len(diffs) == 0:
            return 1.0

        avg_diff = float(np.mean(diffs))
        max_diff = float(np.max(diffs))

        # Low average difference = smooth = high score
        smoothness = 1.0 - np.clip(avg_diff * 3.0, 0.0, 1.0)

        # Penalize large jumps
        if max_diff > 0.3:
            smoothness *= 0.7

        return float(np.clip(smoothness, 0.0, 1.0))

    def _flicker_detection(self, scores: np.ndarray) -> float:
        """Detect rapid score oscillation (flickering).

        Flickering is characterized by alternating high/low scores
        in adjacent frames, common in frame-by-frame deepfakes.
        """
        if len(scores) < 3:
            return 0.0

        diffs = np.diff(scores)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        max_changes = len(diffs) - 1

        if max_changes == 0:
            return 0.0

        flicker_ratio = sign_changes / max_changes
        amplitude = float(np.std(diffs))

        return float(np.clip(flicker_ratio * amplitude * 4, 0.0, 1.0))

    def _trend_stability(self, scores: np.ndarray) -> float:
        """Measure stability of the score trend.

        Stable trend = consistent detection, unstable = suspicious.
        """
        if len(scores) < self.SMOOTHNESS_WINDOW:
            return 0.5

        # Moving average
        kernel = np.ones(self.SMOOTHNESS_WINDOW) / self.SMOOTHNESS_WINDOW
        smoothed = np.convolve(scores, kernel, mode='valid')

        if len(smoothed) < 2:
            return 0.5

        # Measure trend changes
        trend_diffs = np.abs(np.diff(smoothed))
        avg_trend_change = float(np.mean(trend_diffs))

        stability = 1.0 - np.clip(avg_trend_change / self.TREND_CHANGE_THRESHOLD, 0.0, 1.0)
        return float(stability)

    def _detect_periodicity(self, scores: np.ndarray) -> float:
        """Detect periodic patterns in scores using autocorrelation."""
        if len(scores) < 10:
            return 0.0

        centered = scores - np.mean(scores)
        if np.std(centered) < 1e-8:
            return 0.0

        # Autocorrelation
        corr = np.correlate(centered, centered, mode='full')
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-10)

        # Find peaks in autocorrelation (skip lag 0)
        if len(corr) < 4:
            return 0.0

        peaks = []
        for i in range(2, len(corr) - 1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > 0.3:
                peaks.append(corr[i])

        if not peaks:
            return 0.0

        return float(np.max(peaks))
