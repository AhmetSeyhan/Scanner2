"""
Scanner - Frequency Domain Forensic Analyzer (v5.1.0)

Analyzes images and video frames in the frequency domain to detect
spectral anomalies characteristic of AI-generated content.

Techniques:
- 2D FFT power spectrum with azimuthal averaging
- DCT coefficient distribution analysis
- Mid-to-high frequency energy ratio
- Spectral flatness measurement

Reference: Frank et al. (2020) "Leveraging Frequency Analysis for Deep Fake
Image Recognition" - uses publicly available academic research.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class FrequencyAnalysisResult:
    """Result from frequency domain analysis."""
    score: float                    # 0.0 (natural) -> 1.0 (generated)
    confidence: float               # 0.0 -> 1.0
    spectral_flatness: float        # Geometric/arithmetic mean ratio
    mid_freq_ratio: float           # Energy in mid-frequency band
    high_freq_ratio: float          # Energy in high-frequency band
    azimuthal_std: float            # Std of azimuthal average (uniformity indicator)
    dct_kurtosis: float             # DCT coefficient kurtosis
    anomalies: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class FrequencyAnalyzer:
    """Frequency-domain forensic analysis engine.

    AI-generated images exhibit distinct spectral characteristics:
    - GAN upsampling creates periodic artifacts at specific frequencies
    - Diffusion models produce abnormally uniform spectral distributions
    - Natural images follow a characteristic 1/f power spectral density falloff
    """

    # Frequency band boundaries (as fraction of Nyquist frequency)
    LOW_FREQ_UPPER = 0.2
    MID_FREQ_UPPER = 0.5

    # Thresholds for anomaly detection
    FLATNESS_THRESHOLD = 0.35       # Above this = suspiciously uniform spectrum
    HIGH_FREQ_THRESHOLD = 0.25      # Above this = unusual high-frequency content
    KURTOSIS_THRESHOLD_LOW = 1.5    # Below this = too uniform (diffusion)
    KURTOSIS_THRESHOLD_HIGH = 20.0  # Above this = spectral spikes (GAN)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale float64."""
        if image.ndim == 2:
            return image.astype(np.float64)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        # BGR -> Grayscale via luminance
        return (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] +
                0.299 * image[:, :, 2]).astype(np.float64)

    def _compute_power_spectrum(self, gray: np.ndarray) -> np.ndarray:
        """Compute 2D FFT magnitude spectrum (shifted)."""
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        # Log scale for better dynamic range
        power = np.log1p(magnitude)
        return power

    def _azimuthal_average(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Compute azimuthal (radial) average of the power spectrum.

        This averages the power at each radius from the center,
        producing a 1D profile that characterizes the spectral falloff.
        """
        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        profile = np.zeros(max_r)
        for radius in range(max_r):
            mask = r == radius
            values = power_spectrum[mask]
            if len(values) > 0:
                profile[radius] = np.mean(values)

        return profile

    def _spectral_flatness(self, profile: np.ndarray) -> float:
        """Compute spectral flatness: ratio of geometric to arithmetic mean.

        Flatness close to 1.0 indicates uniform spectrum (suspicious for
        AI-generated content). Natural images typically have lower flatness
        due to the 1/f falloff characteristic.
        """
        profile = profile[profile > 0]
        if len(profile) == 0:
            return 0.0

        geo_mean = np.exp(np.mean(np.log(profile + 1e-10)))
        arith_mean = np.mean(profile)
        if arith_mean < 1e-10:
            return 0.0

        return float(min(geo_mean / arith_mean, 1.0))

    def _frequency_band_ratios(self, profile: np.ndarray) -> tuple:
        """Compute energy ratios in low/mid/high frequency bands."""
        n = len(profile)
        if n == 0:
            return 0.33, 0.33, 0.34

        total_energy = np.sum(profile ** 2)
        if total_energy < 1e-10:
            return 0.33, 0.33, 0.34

        low_end = int(n * self.LOW_FREQ_UPPER)
        mid_end = int(n * self.MID_FREQ_UPPER)

        low_energy = np.sum(profile[:low_end] ** 2) / total_energy
        mid_energy = np.sum(profile[low_end:mid_end] ** 2) / total_energy
        high_energy = np.sum(profile[mid_end:] ** 2) / total_energy

        return float(low_energy), float(mid_energy), float(high_energy)

    def _dct_analysis(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze DCT coefficient distribution.

        Natural images have a characteristic heavy-tailed distribution
        of DCT coefficients. AI-generated images often differ.
        """
        try:
            from scipy.fft import dctn
            dct_coeffs = dctn(gray, type=2, norm='ortho')
        except ImportError:
            # Fallback: use FFT-based approximation
            dct_coeffs = np.real(np.fft.fft2(gray))

        # Flatten and remove DC component
        flat = np.abs(dct_coeffs.flatten())
        flat = flat[1:]  # Remove DC

        if len(flat) == 0 or np.std(flat) < 1e-10:
            return {"kurtosis": 3.0, "skewness": 0.0, "entropy": 0.0}

        mean_val = np.mean(flat)
        std_val = np.std(flat)
        centered = flat - mean_val

        # Kurtosis (excess)
        kurtosis = float(np.mean(centered ** 4) / (std_val ** 4) - 3.0)

        # Skewness
        skewness = float(np.mean(centered ** 3) / (std_val ** 3))

        # Normalized entropy
        probs = flat / (np.sum(flat) + 1e-10)
        probs = probs[probs > 0]
        max_entropy = np.log(len(probs)) if len(probs) > 0 else 1.0
        entropy = float(-np.sum(probs * np.log(probs)) / max(max_entropy, 1e-10))

        return {"kurtosis": kurtosis, "skewness": skewness, "entropy": entropy}

    def analyze_frame(self, frame: np.ndarray) -> FrequencyAnalysisResult:
        """Analyze a single frame's frequency characteristics."""
        gray = self._to_grayscale(frame)

        # Resize to standard size for consistent analysis
        target_size = 256
        if gray.shape[0] != target_size or gray.shape[1] != target_size:
            import cv2
            gray = cv2.resize(gray, (target_size, target_size))

        # Power spectrum analysis
        power = self._compute_power_spectrum(gray)
        profile = self._azimuthal_average(power)

        # Metrics
        flatness = self._spectral_flatness(profile)
        low_ratio, mid_ratio, high_ratio = self._frequency_band_ratios(profile)
        azimuthal_std = float(np.std(profile)) if len(profile) > 0 else 0.0

        # DCT analysis
        dct_stats = self._dct_analysis(gray)
        kurtosis = dct_stats["kurtosis"]

        # Anomaly detection
        anomalies = []
        score_components = []

        # Check spectral flatness
        if flatness > self.FLATNESS_THRESHOLD:
            anomalies.append(f"High spectral flatness ({flatness:.3f}) - uniform spectrum suggests generation")
            score_components.append(min((flatness - self.FLATNESS_THRESHOLD) / 0.3, 1.0))

        # Check high-frequency content
        if high_ratio > self.HIGH_FREQ_THRESHOLD:
            anomalies.append(f"Elevated high-frequency energy ({high_ratio:.3f}) - possible upsampling artifacts")
            score_components.append(min((high_ratio - self.HIGH_FREQ_THRESHOLD) / 0.3, 1.0))

        # Check DCT kurtosis
        if kurtosis < self.KURTOSIS_THRESHOLD_LOW:
            anomalies.append(f"Low DCT kurtosis ({kurtosis:.2f}) - unnaturally uniform coefficient distribution")
            score_components.append(min((self.KURTOSIS_THRESHOLD_LOW - kurtosis) / 3.0, 1.0))
        elif kurtosis > self.KURTOSIS_THRESHOLD_HIGH:
            anomalies.append(f"High DCT kurtosis ({kurtosis:.2f}) - spectral spikes detected")
            score_components.append(min((kurtosis - self.KURTOSIS_THRESHOLD_HIGH) / 30.0, 1.0))

        # Compute final score
        if score_components:
            score = min(np.mean(score_components) * 0.8 + 0.2, 1.0)
        else:
            score = max(flatness * 0.3 + high_ratio * 0.3, 0.0)

        score = float(np.clip(score, 0.0, 1.0))

        # Confidence based on signal quality
        confidence = min(0.5 + len(anomalies) * 0.15 + azimuthal_std * 0.1, 1.0)

        return FrequencyAnalysisResult(
            score=score,
            confidence=confidence,
            spectral_flatness=flatness,
            mid_freq_ratio=mid_ratio,
            high_freq_ratio=high_ratio,
            azimuthal_std=azimuthal_std,
            dct_kurtosis=kurtosis,
            anomalies=anomalies,
            details={
                "low_freq_ratio": round(low_ratio, 4),
                "mid_freq_ratio": round(mid_ratio, 4),
                "high_freq_ratio": round(high_ratio, 4),
                "spectral_flatness": round(flatness, 4),
                "dct_kurtosis": round(kurtosis, 4),
                "dct_skewness": round(dct_stats["skewness"], 4),
                "dct_entropy": round(dct_stats["entropy"], 4),
            },
        )

    def analyze(self, frames: List[np.ndarray]) -> FrequencyAnalysisResult:
        """Analyze multiple frames and aggregate results.

        Samples up to 10 frames evenly for efficiency.
        """
        if not frames:
            return FrequencyAnalysisResult(
                score=0.5, confidence=0.0,
                spectral_flatness=0.0, mid_freq_ratio=0.33,
                high_freq_ratio=0.34, azimuthal_std=0.0,
                dct_kurtosis=3.0,
                anomalies=["No frames provided"],
            )

        # Sample frames evenly
        max_sample = 10
        if len(frames) > max_sample:
            indices = np.linspace(0, len(frames) - 1, max_sample, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        results = [self.analyze_frame(f) for f in sampled]

        # Aggregate
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        flatnesses = [r.spectral_flatness for r in results]

        avg_score = float(np.mean(scores))
        avg_confidence = float(np.mean(confidences))

        # Temporal consistency: if scores are very consistent, boost confidence
        score_std = float(np.std(scores))
        if score_std < 0.05:
            avg_confidence = min(avg_confidence + 0.1, 1.0)

        # Collect unique anomalies
        all_anomalies = []
        seen = set()
        for r in results:
            for a in r.anomalies:
                key = a.split(" - ")[0] if " - " in a else a
                if key not in seen:
                    seen.add(key)
                    all_anomalies.append(a)

        return FrequencyAnalysisResult(
            score=avg_score,
            confidence=avg_confidence,
            spectral_flatness=float(np.mean(flatnesses)),
            mid_freq_ratio=float(np.mean([r.mid_freq_ratio for r in results])),
            high_freq_ratio=float(np.mean([r.high_freq_ratio for r in results])),
            azimuthal_std=float(np.mean([r.azimuthal_std for r in results])),
            dct_kurtosis=float(np.mean([r.dct_kurtosis for r in results])),
            anomalies=all_anomalies,
            details={
                "frames_analyzed": len(sampled),
                "score_std": round(score_std, 4),
                "per_frame_scores": [round(s, 4) for s in scores],
            },
        )
