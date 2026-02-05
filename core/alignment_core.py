"""
Scanner Prime - ALIGNMENT CORE
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements multimodal cross-check focusing on phoneme-viseme mapping
and audio-visual alignment using standard open-source libraries (numpy, scipy, OpenCV).
All algorithms are based on publicly available academic research on lip-sync
detection and audio-visual alignment.

Key Features:
- Phoneme-Viseme mapping (bilabial phonemes P, B, M)
- Lip closure event detection and timing analysis
- Speech rhythm analysis via FFT
- Metadata and compression artifact forensics

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

# FFT import with fallback
try:
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    from numpy.fft import fft
    SCIPY_AVAILABLE = False

from core.forensic_types import (
    AlignmentCoreResult,
    VideoProfile,
    ResolutionTier,
    FrameList,
)


class AlignmentCore:
    """
    ALIGNMENT CORE - Multimodal Cross-Check

    Analyzes audio-visual alignment through phoneme-viseme mapping.
    Bilabial phonemes (P, B, M) require complete lip closure, making
    them excellent markers for lip-sync verification.
    """

    # Bilabial phonemes that require lip contact
    BILABIAL_PHONEMES = ['P', 'B', 'M']

    # Expected lip closure interval range (seconds)
    # Natural speech has lip closures every 0.3-2.0 seconds
    LIP_CLOSURE_MIN_INTERVAL = 0.3
    LIP_CLOSURE_MAX_INTERVAL = 2.0

    # Speech rhythm frequency range (Hz)
    SPEECH_FREQ_LOW = 2.0   # ~2 syllables per second
    SPEECH_FREQ_HIGH = 8.0  # ~8 syllables per second

    # Minimum frames required
    MIN_FRAMES_REQUIRED = 10

    def __init__(self):
        self.name = "ALIGNMENT CORE"

    def _extract_mouth_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the mouth region from a frame.

        Assumes face is roughly centered (typical for deepfake content).
        Mouth is typically in the lower third of the face region.
        """
        h, w = frame.shape[:2]

        # Face region bounds
        face_top = h // 6
        face_bottom = 5 * h // 6
        face_left = w // 4
        face_right = 3 * w // 4

        # Mouth region (lower third of face)
        mouth_top = face_top + 2 * (face_bottom - face_top) // 3
        mouth_bottom = face_bottom
        mouth_left = face_left + (face_right - face_left) // 4
        mouth_right = face_right - (face_right - face_left) // 4

        return frame[mouth_top:mouth_bottom, mouth_left:mouth_right]

    def _analyze_lip_movement(
        self,
        frames: FrameList,
        fps: float
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Analyze lip movement across frames.

        Returns:
            Tuple of (movement_signal, movement_magnitudes)
        """
        movements = []
        magnitudes = []

        prev_mouth = None
        for frame in frames:
            mouth_region = self._extract_mouth_region(frame)

            if mouth_region.size == 0:
                movements.append(0.0)
                magnitudes.append(0.0)
                continue

            if len(mouth_region.shape) == 3:
                gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_mouth = mouth_region

            # Compute gradient to measure lip edge activity
            grad = cv2.Laplacian(gray_mouth, cv2.CV_64F)
            movement = np.var(grad)
            movements.append(movement)

            # Compute inter-frame difference
            if prev_mouth is not None and prev_mouth.shape == gray_mouth.shape:
                diff = np.abs(gray_mouth.astype(float) - prev_mouth.astype(float))
                magnitude = np.mean(diff)
                magnitudes.append(magnitude)
            else:
                magnitudes.append(0.0)

            prev_mouth = gray_mouth.copy()

        return np.array(movements), magnitudes

    def _detect_lip_closure_events(
        self,
        frames: FrameList,
        fps: float
    ) -> List[Dict]:
        """
        Detect frames where lips close (bilabial candidates).

        Lip closure is detected by looking for local minima in the
        vertical extent of the mouth opening.

        Returns:
            List of lip closure event dictionaries
        """
        closure_events = []
        mouth_openings = []

        for i, frame in enumerate(frames):
            mouth_region = self._extract_mouth_region(frame)

            if mouth_region.size == 0:
                mouth_openings.append(0.0)
                continue

            if len(mouth_region.shape) == 3:
                gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_mouth = mouth_region

            # Vertical projection to find mouth opening
            vertical_proj = np.sum(gray_mouth, axis=1)

            # Find the gap (dark area = mouth opening)
            threshold = np.mean(vertical_proj)
            below_threshold = vertical_proj < threshold
            opening_width = np.sum(below_threshold)

            mouth_openings.append(opening_width)

        # Detect local minima (lip closures)
        mouth_openings = np.array(mouth_openings)
        if len(mouth_openings) < 5:
            return []

        # Smooth the signal
        window = min(5, len(mouth_openings) // 2)
        if window > 0:
            smoothed = np.convolve(mouth_openings, np.ones(window) / window, mode='same')
        else:
            smoothed = mouth_openings

        # Find local minima
        for i in range(2, len(smoothed) - 2):
            if (smoothed[i] < smoothed[i - 1] and
                smoothed[i] < smoothed[i + 1] and
                smoothed[i] < smoothed[i - 2] and
                smoothed[i] < smoothed[i + 2]):

                # Check if this is a significant closure
                local_max = max(smoothed[max(0, i - 5):min(len(smoothed), i + 5)])
                if smoothed[i] < local_max * 0.7:  # 30% reduction
                    closure_events.append({
                        "frame": i,
                        "time": i / fps,
                        "opening_ratio": float(smoothed[i] / (local_max + 1e-6)),
                        "confidence": float(1.0 - smoothed[i] / (local_max + 1e-6))
                    })

        return closure_events

    def verify_av_alignment(
        self,
        frames: FrameList,
        fps: float
    ) -> Tuple[float, Dict]:
        """
        Verify audio-visual alignment via lip movement FFT.

        Natural speech has characteristic rhythm in the 2-8 Hz range.
        Deepfakes often lack this natural rhythm or have robotic patterns.

        Returns:
            Tuple of (alignment_score, analysis_details)
        """
        movements, _ = self._analyze_lip_movement(frames, fps)

        if len(movements) < self.MIN_FRAMES_REQUIRED:
            return 0.5, {"reason": "Insufficient frames for alignment analysis"}

        # Detrend
        movements = movements - np.mean(movements)

        # FFT analysis
        n = len(movements)
        freqs = np.fft.fftfreq(n, 1 / fps)
        fft_vals = np.abs(fft(movements))

        # Find speech rhythm frequency content
        speech_mask = (freqs > self.SPEECH_FREQ_LOW) & (freqs < self.SPEECH_FREQ_HIGH)
        if np.any(speech_mask):
            speech_power = np.mean(fft_vals[speech_mask])
            total_power = np.mean(fft_vals[freqs > 0]) + 1e-6
            speech_ratio = speech_power / total_power
        else:
            speech_ratio = 0.0

        # Analyze movement variance
        movement_variance = np.var(movements)

        # Detect if movements are too regular (robotic)
        # Natural speech has some randomness in rhythm
        if len(fft_vals) > 5:
            dominant_peak = np.max(fft_vals[1:n // 2])
            mean_power = np.mean(fft_vals[1:n // 2]) + 1e-6
            peak_ratio = dominant_peak / mean_power
        else:
            peak_ratio = 1.0

        # Scoring
        score = 0.0

        # Low speech rhythm ratio
        if speech_ratio < 0.2:
            score += 0.4

        # Low movement variance (static lips)
        if movement_variance < 10:
            score += 0.3

        # Very high peak ratio (robotic lip movement)
        if peak_ratio > 10:
            score += 0.3

        score = min(score, 1.0)

        details = {
            "speech_rhythm_ratio": round(speech_ratio, 4),
            "movement_variance": round(movement_variance, 2),
            "peak_ratio": round(peak_ratio, 2),
            "frames_analyzed": len(movements)
        }

        return score, details

    def _analyze_phoneme_viseme_mapping(
        self,
        frames: FrameList,
        fps: float
    ) -> Tuple[float, List[Dict]]:
        """
        Check lip closure interval naturalness.

        Natural speech has bilabial closures (P, B, M) every 0.3-2.0 seconds.
        Too few or too regular closures suggest manipulation.

        Returns:
            Tuple of (score, closure_events)
        """
        closure_events = self._detect_lip_closure_events(frames, fps)

        if len(closure_events) < 2:
            # No meaningful closure analysis possible
            return 0.5, closure_events

        # Analyze closure intervals
        intervals = []
        for i in range(1, len(closure_events)):
            interval = closure_events[i]["time"] - closure_events[i - 1]["time"]
            intervals.append(interval)

        if not intervals:
            return 0.5, closure_events

        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)

        # Check if intervals are in natural range
        in_range_count = sum(
            1 for i in intervals
            if self.LIP_CLOSURE_MIN_INTERVAL <= i <= self.LIP_CLOSURE_MAX_INTERVAL
        )
        natural_ratio = in_range_count / len(intervals)

        # Check for unnaturally regular intervals (robotic)
        interval_cv = interval_std / (avg_interval + 1e-6)

        score = 0.0

        # Few natural interval closures
        if natural_ratio < 0.5:
            score += 0.35

        # Too regular (low coefficient of variation)
        if interval_cv < 0.2 and len(intervals) > 3:
            score += 0.25

        # Too few closures overall
        video_duration = len(frames) / fps
        expected_closures = video_duration / 1.0  # ~1 closure per second on average
        if len(closure_events) < expected_closures * 0.3:
            score += 0.3

        score = min(score, 1.0)

        # Add analysis details to events
        for event in closure_events:
            event["natural_range"] = (
                self.LIP_CLOSURE_MIN_INTERVAL <= event.get("time", 0) <=
                self.LIP_CLOSURE_MAX_INTERVAL
            )

        return score, closure_events

    def _analyze_metadata_integrity(
        self,
        frames: FrameList,
        video_path: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Analyze compression artifacts and brightness consistency.

        Looks for encoding anomalies that might indicate splicing or
        post-processing.

        Returns:
            Tuple of (score, details)
        """
        brightness_values = []
        compression_scores = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Brightness analysis
            brightness = np.mean(gray)
            brightness_values.append(brightness)

            # Compression artifact analysis (8x8 block boundaries)
            h, w = gray.shape
            if h >= 16 and w >= 16:
                # Check 8x8 block boundary discontinuities
                boundary_grads = []
                for i in range(8, h - 8, 8):
                    boundary_grads.append(
                        np.mean(np.abs(
                            gray[i - 1, :].astype(float) - gray[i, :].astype(float)
                        ))
                    )
                for j in range(8, w - 8, 8):
                    boundary_grads.append(
                        np.mean(np.abs(
                            gray[:, j - 1].astype(float) - gray[:, j].astype(float)
                        ))
                    )

                if boundary_grads:
                    compression_scores.append(np.mean(boundary_grads))

        if not brightness_values:
            return 0.5, {"reason": "No frames analyzed"}

        # Detect brightness jumps
        brightness_diff = np.diff(brightness_values)
        jump_threshold = np.std(brightness_values) * 2
        brightness_jumps = np.sum(np.abs(brightness_diff) > jump_threshold)

        # Compression artifact consistency
        if compression_scores:
            compression_mean = np.mean(compression_scores)
            compression_std = np.std(compression_scores)
        else:
            compression_mean = 0
            compression_std = 0

        score = 0.0

        # Many brightness jumps suggest splicing
        if brightness_jumps > len(frames) * 0.1:
            score += 0.4

        # Inconsistent compression artifacts
        if compression_std > compression_mean * 0.5 and compression_mean > 0:
            score += 0.3

        score = min(score, 1.0)

        details = {
            "brightness_mean": round(float(np.mean(brightness_values)), 2),
            "brightness_std": round(float(np.std(brightness_values)), 2),
            "brightness_jumps": int(brightness_jumps),
            "compression_score": round(compression_mean, 2),
            "compression_consistency": round(1.0 - min(compression_std / (compression_mean + 1e-6), 1.0), 2)
        }

        return score, details

    def analyze(
        self,
        frames: FrameList,
        fps: float,
        video_path: Optional[str] = None,
        video_profile: Optional[VideoProfile] = None
    ) -> AlignmentCoreResult:
        """
        Run complete multimodal cross-check analysis.

        Args:
            frames: List of video frames
            fps: Frames per second
            video_path: Optional path to video file
            video_profile: Optional video profile for adaptive processing

        Returns:
            AlignmentCoreResult with multimodal analysis
        """
        if len(frames) < self.MIN_FRAMES_REQUIRED:
            return AlignmentCoreResult(
                core_name=self.name,
                score=0.5,
                confidence=0.3,
                status="WARN",
                details={"reason": f"Insufficient frames ({len(frames)} < {self.MIN_FRAMES_REQUIRED})"},
                anomalies=["LOW_FRAME_COUNT"],
                data_quality="INSUFFICIENT"
            )

        # A/V alignment analysis
        av_score, av_details = self.verify_av_alignment(frames, fps)

        # Phoneme-Viseme mapping analysis
        pv_score, closure_events = self._analyze_phoneme_viseme_mapping(frames, fps)

        # Metadata integrity analysis
        metadata_score, metadata_details = self._analyze_metadata_integrity(frames, video_path)

        # Collect anomalies
        anomalies = []

        if av_score > 0.5:
            if av_details.get("speech_rhythm_ratio", 1) < 0.2:
                anomalies.append("MISSING_SPEECH_RHYTHM")
            if av_details.get("movement_variance", 100) < 10:
                anomalies.append("STATIC_LIP_MOVEMENT")
            if av_details.get("peak_ratio", 1) > 10:
                anomalies.append("ROBOTIC_LIP_SYNC")

        if pv_score > 0.5:
            anomalies.append("PHONEME_VISEME_MISMATCH")

        if metadata_score > 0.4:
            if metadata_details.get("brightness_jumps", 0) > len(frames) * 0.1:
                anomalies.append("BRIGHTNESS_DISCONTINUITY")
            anomalies.append("ENCODING_ANOMALY")

        # Overall score: weighted combination
        overall_score = (av_score * 0.4 + pv_score * 0.35 + metadata_score * 0.25)
        overall_score = min(overall_score, 1.0)

        # Details
        details = {
            "av_alignment": av_details,
            "phoneme_viseme": {
                "score": round(pv_score, 4),
                "closure_count": len(closure_events),
            },
            "metadata": metadata_details,
        }

        # Confidence
        confidence = 0.6 if len(frames) >= 30 else 0.4
        if video_profile and video_profile.is_low_res:
            confidence *= 0.8  # Lower confidence for low-res

        # Determine status
        if overall_score > 0.6:
            status = "FAIL"
        elif overall_score > 0.3:
            status = "WARN"
        else:
            status = "PASS"

        return AlignmentCoreResult(
            core_name=self.name,
            score=overall_score,
            confidence=confidence,
            status=status,
            details=details,
            anomalies=anomalies,
            data_quality="GOOD" if len(frames) >= 30 else "LIMITED",
            av_alignment_score=av_score,
            phoneme_viseme_score=pv_score,
            lip_closure_events=closure_events[:10],  # First 10 events
            speech_rhythm_score=av_details.get("speech_rhythm_ratio", 0.0),
            metadata_integrity=1.0 - metadata_score
        )
