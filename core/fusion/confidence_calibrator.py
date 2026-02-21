"""
Scanner - Confidence Calibrator (v6.0.0)

Calibrates detector confidence scores using Platt scaling and
isotonic regression. Ensures that a confidence of 0.9 actually
means the detector is correct ~90% of the time.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    """Result from confidence calibration."""
    calibrated_score: float
    calibrated_confidence: float
    reliability: float
    calibration_method: str


class ConfidenceCalibrator:
    """Calibrate detection scores for reliable probability estimates.

    Uses Platt scaling (logistic regression on scores) and
    temperature scaling for post-hoc calibration.
    """

    def __init__(self):
        self._platt_a: float = -1.0  # Logistic slope
        self._platt_b: float = 0.0   # Logistic intercept
        self._temperature: float = 1.0
        self._calibrated = False
        # ECE bins for reliability measurement
        self._n_bins = 10

    def fit(self, scores: List[float], labels: List[int]) -> None:
        """Fit Platt scaling parameters from validation data.

        Args:
            scores: Detector output scores (0-1)
            labels: Ground truth labels (0=real, 1=fake)
        """
        if len(scores) < 10:
            return

        scores_arr = np.array(scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.float64)

        # Platt scaling: fit sigmoid(a*score + b) to labels
        # Using gradient descent on log-loss
        a, b = -1.0, 0.0
        lr = 0.01

        for _ in range(1000):
            z = a * scores_arr + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)

            # Gradients of log-loss
            grad_a = np.mean((p - labels_arr) * scores_arr)
            grad_b = np.mean(p - labels_arr)

            a -= lr * grad_a
            b -= lr * grad_b

        self._platt_a = float(a)
        self._platt_b = float(b)
        self._calibrated = True

        # Fit temperature
        self._fit_temperature(scores_arr, labels_arr)

    def _fit_temperature(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature scaling parameter."""
        best_t = 1.0
        best_ece = float('inf')

        for t in np.arange(0.1, 5.0, 0.1):
            calibrated = self._apply_temperature(scores, t)
            ece = self._compute_ece(calibrated, labels)
            if ece < best_ece:
                best_ece = ece
                best_t = t

        self._temperature = float(best_t)

    def calibrate(self, score: float, confidence: float) -> CalibrationResult:
        """Calibrate a single score and confidence."""
        if self._calibrated:
            cal_score = self._platt_transform(score)
            cal_conf = self._apply_temperature(
                np.array([confidence]), self._temperature
            )[0]
            method = "platt_scaling"
        else:
            cal_score = score
            cal_conf = confidence
            method = "uncalibrated"

        reliability = self._estimate_reliability(cal_score, cal_conf)

        return CalibrationResult(
            calibrated_score=float(np.clip(cal_score, 0.0, 1.0)),
            calibrated_confidence=float(np.clip(cal_conf, 0.0, 1.0)),
            reliability=reliability,
            calibration_method=method,
        )

    def calibrate_batch(
        self,
        scores: List[float],
        confidences: List[float],
    ) -> List[CalibrationResult]:
        """Calibrate a batch of scores."""
        return [
            self.calibrate(s, c) for s, c in zip(scores, confidences)
        ]

    def _platt_transform(self, score: float) -> float:
        """Apply Platt scaling transform."""
        z = self._platt_a * score + self._platt_b
        return float(1.0 / (1.0 + np.exp(-z)))

    @staticmethod
    def _apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling."""
        logits = np.log(scores / (1.0 - scores + 1e-10) + 1e-10)
        scaled = logits / temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self._n_bins + 1)
        ece = 0.0
        n = len(probs)

        for i in range(self._n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if np.sum(mask) == 0:
                continue
            bin_conf = np.mean(probs[mask])
            bin_acc = np.mean(labels[mask])
            ece += np.sum(mask) / n * abs(bin_conf - bin_acc)

        return float(ece)

    @staticmethod
    def _estimate_reliability(score: float, confidence: float) -> float:
        """Estimate reliability of the calibrated result."""
        # Reliability is higher when score is decisive and confidence is high
        decisiveness = abs(score - 0.5) * 2  # 0 at 0.5, 1 at extremes
        return float(np.clip(decisiveness * 0.5 + confidence * 0.5, 0.0, 1.0))
