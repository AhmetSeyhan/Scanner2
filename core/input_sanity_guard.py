"""
Scanner Prime - Input Sanity Guard
Adversarial input detection and frame sequence validation.

This module provides a pre-analysis defense layer to detect:
- Adversarial perturbations designed to fool deepfake detectors
- Frame sequence anomalies (duplicates, shuffling, injection)
- Resolution inconsistencies
- Content integrity issues

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

from core.forensic_types import SanityCheckResult, FrameList


class InputSanityGuard:
    """
    Adversarial Defense Layer

    Validates input before forensic analysis to:
    1. Detect adversarial perturbations
    2. Verify frame sequence consistency
    3. Check resolution consistency
    4. Validate content integrity
    """

    # Thresholds
    MAX_FRAME_DIFF_RATIO = 0.95  # Max % pixels that can change between frames
    MIN_FRAME_DIFF_RATIO = 0.001  # Min % to detect identical frames
    ADVERSARIAL_GRADIENT_THRESHOLD = 200  # Max avg gradient magnitude
    MIN_FRAMES_REQUIRED = 2

    def __init__(self):
        self.name = "INPUT SANITY GUARD"

    def validate(
        self,
        frames: FrameList,
        expected_fps: Optional[float] = None
    ) -> SanityCheckResult:
        """
        Run complete input validation.

        Args:
            frames: List of video frames
            expected_fps: Expected FPS if known

        Returns:
            SanityCheckResult with validation outcome
        """
        checks_passed = []
        checks_failed = []
        warnings = []

        if len(frames) < self.MIN_FRAMES_REQUIRED:
            return SanityCheckResult(
                is_valid=False,
                checks_passed=[],
                checks_failed=["INSUFFICIENT_FRAMES"],
                warnings=[],
                frame_consistency_score=0.0,
                resolution_consistency=False,
                content_integrity_score=0.0,
                rejection_reason="Need at least 2 frames for validation"
            )

        # Check 1: Frame sequence consistency
        consistency_score, consistency_warnings = self._check_frame_consistency(frames)
        if consistency_score > 0.5:
            checks_passed.append("FRAME_SEQUENCE_VALID")
        else:
            checks_failed.append("FRAME_SEQUENCE_INVALID")
        warnings.extend(consistency_warnings)

        # Check 2: Resolution consistency
        res_consistent = self._check_resolution_consistency(frames)
        if res_consistent:
            checks_passed.append("RESOLUTION_CONSISTENT")
        else:
            checks_failed.append("RESOLUTION_MISMATCH")
            warnings.append("Frame resolutions vary - possible splicing")

        # Check 3: Adversarial perturbation detection
        integrity_score, adv_warnings = self._check_adversarial_patterns(frames)
        if integrity_score > 0.6:
            checks_passed.append("NO_ADVERSARIAL_PATTERNS")
        else:
            checks_failed.append("ADVERSARIAL_PATTERN_DETECTED")
        warnings.extend(adv_warnings)

        # Check 4: Content integrity (extreme values, NaN, etc.)
        content_valid, content_warnings = self._check_content_integrity(frames)
        if content_valid:
            checks_passed.append("CONTENT_INTEGRITY_OK")
        else:
            checks_failed.append("CONTENT_INTEGRITY_FAILED")
        warnings.extend(content_warnings)

        # Determine overall validity
        critical_failures = {"FRAME_SEQUENCE_INVALID", "ADVERSARIAL_PATTERN_DETECTED"}
        is_valid = not any(f in critical_failures for f in checks_failed)

        rejection_reason = None
        if not is_valid:
            if "ADVERSARIAL_PATTERN_DETECTED" in checks_failed:
                rejection_reason = "Potential adversarial attack detected"
            elif "FRAME_SEQUENCE_INVALID" in checks_failed:
                rejection_reason = "Frame sequence appears corrupted or manipulated"

        return SanityCheckResult(
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            frame_consistency_score=consistency_score,
            resolution_consistency=res_consistent,
            content_integrity_score=integrity_score,
            rejection_reason=rejection_reason
        )

    def _check_frame_consistency(
        self,
        frames: FrameList
    ) -> Tuple[float, List[str]]:
        """Check temporal consistency between adjacent frames."""
        warnings = []
        diff_ratios = []

        for i in range(1, min(len(frames), 30)):  # Check first 30 frame pairs
            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)

            if prev.shape != curr.shape:
                continue

            diff = np.abs(prev - curr)
            changed_pixels = np.sum(diff > 10) / diff.size
            diff_ratios.append(changed_pixels)

        if not diff_ratios:
            return 0.0, ["Could not compute frame differences"]

        avg_diff = np.mean(diff_ratios)
        std_diff = np.std(diff_ratios)

        # Check for suspicious patterns
        if avg_diff > self.MAX_FRAME_DIFF_RATIO:
            warnings.append("Extreme frame-to-frame changes detected")
        if avg_diff < self.MIN_FRAME_DIFF_RATIO:
            warnings.append("Frames appear nearly identical (possible loop)")
        if std_diff > 0.3:
            warnings.append("Inconsistent frame changes (possible splicing)")

        # Score: penalize extremes
        score = 1.0
        if avg_diff > 0.5 or avg_diff < 0.01:
            score *= 0.5
        if std_diff > 0.2:
            score *= 0.7

        return score, warnings

    def _check_resolution_consistency(self, frames: FrameList) -> bool:
        """Check that all frames have same resolution."""
        if not frames:
            return False

        base_shape = frames[0].shape
        return all(f.shape == base_shape for f in frames)

    def _check_adversarial_patterns(
        self,
        frames: FrameList
    ) -> Tuple[float, List[str]]:
        """Detect adversarial perturbation patterns."""
        warnings = []
        high_grad_count = 0

        for frame in frames[:10]:  # Check first 10 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # High-frequency gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Adversarial attacks often have unnaturally high gradients
            if np.mean(grad_mag) > self.ADVERSARIAL_GRADIENT_THRESHOLD:
                high_grad_count += 1

            # Check for periodic noise patterns (FGSM signature)
            fft = np.fft.fft2(gray.astype(np.float32))
            fft_mag = np.abs(np.fft.fftshift(fft))

            # Adversarial noise often has specific frequency spikes
            h, w = fft_mag.shape
            high_freq_energy = np.sum(fft_mag[h // 4:3 * h // 4, w // 4:3 * w // 4])
            total_energy = np.sum(fft_mag) + 1e-6

            if high_freq_energy / total_energy > 0.9:
                warnings.append("Unusual frequency distribution detected")

        if high_grad_count > len(frames[:10]) * 0.3:
            warnings.append("Multiple frames show adversarial gradient patterns")

        # Score based on warnings
        num_checked = max(len(frames[:10]), 1)
        score = 1.0 - (high_grad_count / num_checked) * 0.5
        score = max(0.0, min(1.0, score))

        return score, warnings

    def _check_content_integrity(
        self,
        frames: FrameList
    ) -> Tuple[bool, List[str]]:
        """Check for corrupted content (NaN, extreme values)."""
        warnings = []

        for i, frame in enumerate(frames[:5]):  # Check first 5
            # Check for NaN or Inf
            if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
                return False, [f"Frame {i} contains NaN/Inf values"]

            # Check for extreme values
            if frame.max() > 255 or frame.min() < 0:
                warnings.append(f"Frame {i} has out-of-range pixel values")

            # Check for mostly black/white frames
            mean_val = np.mean(frame)
            if mean_val < 5 or mean_val > 250:
                warnings.append(f"Frame {i} appears nearly black/white")

        return True, warnings

    def quick_check(self, frames: FrameList) -> bool:
        """
        Quick validation check (faster, less thorough).

        Args:
            frames: List of video frames

        Returns:
            True if input passes basic checks
        """
        if len(frames) < self.MIN_FRAMES_REQUIRED:
            return False

        if not self._check_resolution_consistency(frames):
            return False

        # Quick content check
        for frame in frames[:3]:
            if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
                return False

        return True
