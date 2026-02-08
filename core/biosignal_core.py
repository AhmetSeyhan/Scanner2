"""
Scanner Prime - BIOSIGNAL CORE
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements biological integrity analysis via 32-region rPPG
(remote photoplethysmography) using standard open-source libraries
(numpy, scipy, OpenCV). All algorithms are based on publicly available
academic research on rPPG signal extraction and analysis.

Key Features:
- 32 ROI grid (8x4) over face region with weighted importance
- Cross-correlation analysis between ROI signals for biological sync
- Multi-order Butterworth bandpass filter with fallback
- Spatial averaging for low-resolution signal recovery

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

# Scipy imports with fallback
try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    from numpy.fft import fft

from core.forensic_types import (
    ROIRegion,
    BiologicalSignal,
    BioSignalCoreResult,
    VideoProfile,
    ResolutionTier,
    FrameList,
)


class BioSignalCore:
    """
    BIOSIGNAL CORE - Biological Integrity Analysis

    Analyzes 32 facial regions for rPPG signals to detect authentic
    biological presence. Deepfakes often lack coherent biological
    signals across facial regions.
    """

    # ROI Grid Configuration
    ROI_GRID_ROWS = 8
    ROI_GRID_COLS = 4

    # Heart rate frequency range (Hz)
    HR_LOW_FREQ = 0.7   # ~42 BPM
    HR_HIGH_FREQ = 4.0  # ~240 BPM

    # Expected heart rate range (BPM)
    EXPECTED_HR_MIN = 50
    EXPECTED_HR_MAX = 120

    # Minimum frames required
    MIN_FRAMES_REQUIRED = 30

    # ROI weights (cheeks and forehead are best for pulse detection)
    # Grid position weights: higher values for cheeks (rows 2-4, cols 1-2)
    ROI_WEIGHT_MAP = {
        # Row 0-1: Forehead (good for pulse)
        (0, 0): 0.8, (0, 1): 0.9, (0, 2): 0.9, (0, 3): 0.8,
        (1, 0): 0.9, (1, 1): 1.0, (1, 2): 1.0, (1, 3): 0.9,
        # Row 2-3: Upper cheeks (best for pulse)
        (2, 0): 1.0, (2, 1): 1.2, (2, 2): 1.2, (2, 3): 1.0,
        (3, 0): 1.0, (3, 1): 1.2, (3, 2): 1.2, (3, 3): 1.0,
        # Row 4-5: Lower cheeks (good for pulse)
        (4, 0): 0.9, (4, 1): 1.0, (4, 2): 1.0, (4, 3): 0.9,
        (5, 0): 0.8, (5, 1): 0.9, (5, 2): 0.9, (5, 3): 0.8,
        # Row 6-7: Lower face (less reliable)
        (6, 0): 0.5, (6, 1): 0.6, (6, 2): 0.6, (6, 3): 0.5,
        (7, 0): 0.3, (7, 1): 0.4, (7, 2): 0.4, (7, 3): 0.3,
    }

    def __init__(self):
        self.name = "BIOSIGNAL CORE"

    def _generate_roi_grid(self, frame_shape: Tuple[int, int]) -> List[ROIRegion]:
        """
        Generate 32 ROI regions (8x4 grid) over the face region.

        Args:
            frame_shape: (height, width) of the frame

        Returns:
            List of 32 ROIRegion objects
        """
        h, w = frame_shape[:2]

        # Face region (assuming face is roughly centered)
        face_top = h // 6
        face_bottom = 5 * h // 6
        face_left = w // 4
        face_right = 3 * w // 4

        face_height = face_bottom - face_top
        face_width = face_right - face_left

        cell_height = face_height // self.ROI_GRID_ROWS
        cell_width = face_width // self.ROI_GRID_COLS

        regions = []
        roi_index = 0

        for row in range(self.ROI_GRID_ROWS):
            for col in range(self.ROI_GRID_COLS):
                y1 = face_top + row * cell_height
                y2 = y1 + cell_height
                x1 = face_left + col * cell_width
                x2 = x1 + cell_width

                weight = self.ROI_WEIGHT_MAP.get((row, col), 0.5)
                name = f"ROI_{row}_{col}"

                regions.append(ROIRegion(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    weight=weight, name=name
                ))
                roi_index += 1

        return regions

    def _bandpass_filter(self, data: np.ndarray, fps: float) -> np.ndarray:
        """
        Apply bandpass filter to isolate heart rate frequencies.
        Includes multi-order fallback for stability.

        Args:
            data: Time series signal
            fps: Frames per second

        Returns:
            Filtered signal
        """
        if not SCIPY_AVAILABLE:
            # Simple moving average fallback
            window = max(3, int(fps / 4))
            return np.convolve(data, np.ones(window) / window, mode='same')

        try:
            nyquist = fps / 2
            if nyquist <= self.HR_LOW_FREQ:
                return data

            low = max(0.01, self.HR_LOW_FREQ / nyquist)
            high = min(0.99, self.HR_HIGH_FREQ / nyquist)

            if low >= high:
                return data

            # Try higher order filter first, fall back if unstable
            for order in [4, 3, 2, 1]:
                try:
                    b, a = scipy_signal.butter(order, [low, high], btype='band')
                    padlen = min(len(data) - 1, 3 * order)
                    filtered = scipy_signal.filtfilt(b, a, data, padlen=padlen)
                    if not np.any(np.isnan(filtered)) and not np.any(np.isinf(filtered)):
                        return filtered
                except Exception:
                    continue

            return data
        except Exception:
            return data

    def _extract_roi_signal(
        self,
        frames: FrameList,
        roi: ROIRegion,
        is_low_res: bool
    ) -> BiologicalSignal:
        """
        Extract rPPG signal from a single ROI across all frames.

        Args:
            frames: List of video frames
            roi: Region of interest
            is_low_res: Whether to use spatial averaging for low-res

        Returns:
            BiologicalSignal with extracted signal
        """
        green_signals = []

        for frame in frames:
            region = roi.extract_from_frame(frame)
            if region.size == 0:
                green_signals.append(0.0)
                continue

            if len(region.shape) == 3:
                if is_low_res:
                    # Spatial averaging: use block means for low-SNR recovery
                    block_size = 8 if region.shape[0] >= 16 else 4
                    roi_h, roi_w = region.shape[:2]

                    block_means = []
                    for bi in range(0, roi_h - block_size + 1, block_size):
                        for bj in range(0, roi_w - block_size + 1, block_size):
                            block = region[bi:bi + block_size, bj:bj + block_size, 1]
                            block_means.append(np.mean(block))

                    green_mean = np.median(block_means) if block_means else np.mean(region[:, :, 1])
                else:
                    # Standard: mean of green channel
                    green_mean = np.mean(region[:, :, 1])
            else:
                green_mean = np.mean(region)

            green_signals.append(green_mean)

        signal = np.array(green_signals)

        # Check for valid signal
        if len(signal) < self.MIN_FRAMES_REQUIRED:
            return BiologicalSignal(
                roi_index=0, roi_name=roi.name,
                signal=signal, quality=0.0,
                estimated_hr=0.0, signal_strength=0.0,
                is_valid=False
            )

        # Detrend and normalize
        signal = signal - np.mean(signal)
        std = np.std(signal)
        if std < 1e-6:
            return BiologicalSignal(
                roi_index=0, roi_name=roi.name,
                signal=signal, quality=0.0,
                estimated_hr=0.0, signal_strength=0.0,
                is_valid=False
            )
        signal = signal / std

        return BiologicalSignal(
            roi_index=0, roi_name=roi.name,
            signal=signal, quality=1.0,
            estimated_hr=0.0, signal_strength=0.0,
            is_valid=True
        )

    def calculate_biological_sync(
        self,
        roi_signals: List[BiologicalSignal],
        fps: float
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate cross-correlation between ROI signals.
        High synchronization indicates authentic biological signal.

        Args:
            roi_signals: List of BiologicalSignal from each ROI
            fps: Frames per second

        Returns:
            Tuple of (sync_score, correlation_matrix)
        """
        valid_signals = [s for s in roi_signals if s.is_valid and len(s.signal) > 0]

        if len(valid_signals) < 2:
            return 0.0, np.array([[]])

        # Filter all signals
        filtered_signals = []
        for sig in valid_signals:
            filtered = self._bandpass_filter(sig.signal, fps)
            filtered_signals.append(filtered)

        # Ensure equal length
        min_len = min(len(s) for s in filtered_signals)
        filtered_signals = [s[:min_len] for s in filtered_signals]

        n_signals = len(filtered_signals)
        correlation_matrix = np.zeros((n_signals, n_signals))

        # Calculate pairwise correlations
        for i in range(n_signals):
            for j in range(i, n_signals):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    sig_i = filtered_signals[i]
                    sig_j = filtered_signals[j]

                    std_i = np.std(sig_i)
                    std_j = np.std(sig_j)

                    if std_i > 0.01 and std_j > 0.01:
                        corr = np.corrcoef(sig_i, sig_j)[0, 1]
                        if not np.isnan(corr):
                            correlation_matrix[i, j] = corr
                            correlation_matrix[j, i] = corr

        # Calculate sync score as mean of upper triangle correlations
        upper_tri = correlation_matrix[np.triu_indices(n_signals, k=1)]
        if len(upper_tri) > 0:
            sync_score = np.mean(upper_tri)
        else:
            sync_score = 0.0

        return float(sync_score), correlation_matrix

    def _analyze_heart_rates(
        self,
        roi_signals: List[BiologicalSignal],
        fps: float
    ) -> Tuple[List[float], float, float]:
        """
        Analyze heart rates across all ROIs.

        Returns:
            Tuple of (hr_list, hr_consistency, pulse_coverage)
        """
        heart_rates = []
        valid_count = 0

        for sig in roi_signals:
            if not sig.is_valid:
                continue

            # Apply bandpass filter
            filtered = self._bandpass_filter(sig.signal, fps)

            # FFT analysis
            n = len(filtered)
            freqs = np.fft.fftfreq(n, 1 / fps)
            fft_vals = np.abs(fft(filtered))

            # Find dominant frequency in HR range
            hr_mask = (freqs > self.HR_LOW_FREQ) & (freqs < self.HR_HIGH_FREQ)
            if np.any(hr_mask):
                hr_freqs = freqs[hr_mask]
                hr_fft = fft_vals[hr_mask]
                dominant_freq = hr_freqs[np.argmax(hr_fft)]
                estimated_hr = dominant_freq * 60
                signal_strength = np.max(hr_fft) / (np.mean(hr_fft) + 1e-6)

                if signal_strength > 1.5:  # Minimum signal strength
                    heart_rates.append(estimated_hr)
                    valid_count += 1

        if not heart_rates:
            return [], 0.0, 0.0

        # Consistency: inverse of coefficient of variation
        hr_mean = np.mean(heart_rates)
        hr_std = np.std(heart_rates)
        hr_consistency = 1.0 - min(hr_std / (hr_mean + 1e-6), 1.0) if hr_mean > 0 else 0.0

        # Pulse coverage: % of ROIs with valid pulse
        pulse_coverage = valid_count / len(roi_signals) if roi_signals else 0.0

        return heart_rates, hr_consistency, pulse_coverage

    # =========================================================================
    # v4.0.0 PPG MAP GENERATION (Intel FakeCatcher style)
    # =========================================================================

    def generate_ppg_map(
        self,
        frames: List[np.ndarray],
        fps: float,
        grid_size: Tuple[int, int] = (8, 4),
    ) -> Dict[str, Any]:
        """
        Generate a PPG (Photoplethysmography) map for visualization.

        Similar to Intel FakeCatcher's PPG map output - shows spatial
        distribution of blood flow signal quality across the face.

        Args:
            frames: Video frames (BGR).
            fps: Video frame rate.
            grid_size: Grid dimensions for the PPG map (rows, cols).

        Returns:
            Dict with ppg_map, quality_map, hr_map, coherence_map and metadata.
        """
        rows, cols = grid_size
        if len(frames) < 30:
            return {
                "ppg_map": np.zeros((rows, cols), dtype=np.float32).tolist(),
                "quality_map": np.zeros((rows, cols), dtype=np.float32).tolist(),
                "hr_map": np.zeros((rows, cols), dtype=np.float32).tolist(),
                "coherence_map": np.zeros((rows, cols), dtype=np.float32).tolist(),
                "grid_size": list(grid_size),
                "mean_ppg_strength": 0.0,
                "mean_quality": 0.0,
                "ppg_coverage": 0.0,
                "status": "INSUFFICIENT_FRAMES",
            }

        h, w = frames[0].shape[:2]
        cell_h, cell_w = h // rows, w // cols

        ppg_map = np.zeros((rows, cols), dtype=np.float32)
        quality_map = np.zeros((rows, cols), dtype=np.float32)
        hr_map = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, min((r + 1) * cell_h, h)
                x1, x2 = c * cell_w, min((c + 1) * cell_w, w)

                # Extract green channel signal for this ROI
                signals = []
                for frame in frames:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        green_mean = np.mean(roi[:, :, 1])  # Green channel
                        signals.append(green_mean)

                if len(signals) < 30:
                    continue

                signal_arr = np.array(signals, dtype=np.float64)
                signal_arr = signal_arr - np.mean(signal_arr)  # Detrend

                # FFT for pulse detection
                fft_vals = np.fft.rfft(signal_arr)
                freqs = np.fft.rfftfreq(len(signal_arr), d=1.0 / fps)

                # Bandpass: 0.7-4.0 Hz (42-240 BPM)
                mask = (freqs >= 0.7) & (freqs <= 4.0)
                if not np.any(mask):
                    continue

                magnitudes = np.abs(fft_vals)
                bandpass_mags = magnitudes.copy()
                bandpass_mags[~mask] = 0

                # Peak frequency -> heart rate
                peak_idx = np.argmax(bandpass_mags)
                peak_freq = freqs[peak_idx]
                estimated_hr = peak_freq * 60

                # Signal quality: ratio of peak energy to total energy
                peak_energy = bandpass_mags[peak_idx] ** 2
                total_energy = np.sum(magnitudes ** 2) + 1e-10
                quality = peak_energy / total_energy

                # PPG strength: normalised peak magnitude
                ppg_strength = bandpass_mags[peak_idx] / (np.max(magnitudes) + 1e-10)

                ppg_map[r, c] = ppg_strength
                quality_map[r, c] = min(quality * 5, 1.0)
                hr_map[r, c] = estimated_hr

        # Coherence map: consistency between adjacent cells
        coherence_map = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(ppg_map[nr, nc])
                if neighbors and ppg_map[r, c] > 0:
                    coherence_map[r, c] = 1.0 - min(
                        np.std([ppg_map[r, c]] + neighbors) / (np.mean([ppg_map[r, c]] + neighbors) + 1e-10),
                        1.0,
                    )

        return {
            "ppg_map": ppg_map.tolist(),
            "quality_map": quality_map.tolist(),
            "hr_map": hr_map.tolist(),
            "coherence_map": coherence_map.tolist(),
            "grid_size": list(grid_size),
            "mean_ppg_strength": float(np.mean(ppg_map)),
            "mean_quality": float(np.mean(quality_map)),
            "ppg_coverage": float(np.mean(ppg_map > 0.1)),
            "status": "OK",
        }

    def analyze(
        self,
        frames: FrameList,
        fps: float,
        video_profile: Optional[VideoProfile] = None
    ) -> BioSignalCoreResult:
        """
        Analyze frames for biological integrity via 32-ROI rPPG.

        Args:
            frames: List of video frames
            fps: Frames per second
            video_profile: Optional video profile for adaptive processing

        Returns:
            BioSignalCoreResult with biological analysis
        """
        # Determine processing mode
        is_low_res = False
        data_quality = "GOOD"

        if video_profile:
            is_low_res = video_profile.is_low_res
            if not video_profile.rppg_viable:
                data_quality = "INSUFFICIENT"
            elif is_low_res:
                data_quality = "LIMITED"

        # Check minimum frames
        if len(frames) < self.MIN_FRAMES_REQUIRED:
            return BioSignalCoreResult(
                core_name=self.name,
                score=0.5,
                confidence=0.2,
                status="WARN",
                details={"reason": f"Insufficient frames ({len(frames)} < {self.MIN_FRAMES_REQUIRED})"},
                anomalies=["LOW_FRAME_COUNT"],
                data_quality="INSUFFICIENT"
            )

        # Generate ROI grid from first frame
        if not frames:
            return BioSignalCoreResult(
                core_name=self.name,
                score=0.5,
                confidence=0.1,
                status="WARN",
                details={"reason": "No frames provided"},
                anomalies=["NO_FRAMES"],
                data_quality="INSUFFICIENT"
            )

        roi_regions = self._generate_roi_grid(frames[0].shape)

        # Extract signals from all ROIs
        roi_signals = []
        for i, roi in enumerate(roi_regions):
            sig = self._extract_roi_signal(frames, roi, is_low_res)
            sig.roi_index = i
            roi_signals.append(sig)

        # Calculate biological sync
        sync_score, corr_matrix = self.calculate_biological_sync(roi_signals, fps)

        # Analyze heart rates
        heart_rates, hr_consistency, pulse_coverage = self._analyze_heart_rates(roi_signals, fps)

        # Build details
        details = {
            "roi_count": len(roi_regions),
            "valid_roi_count": sum(1 for s in roi_signals if s.is_valid),
            "biological_sync": round(sync_score, 4),
            "pulse_coverage": round(pulse_coverage, 4),
            "hr_consistency": round(hr_consistency, 4),
            "spatial_averaging_used": is_low_res,
        }

        if heart_rates:
            details["heart_rates_bpm"] = [round(hr, 1) for hr in heart_rates[:5]]  # First 5
            details["mean_hr_bpm"] = round(np.mean(heart_rates), 1)
            details["hr_std_bpm"] = round(np.std(heart_rates), 1)

        # Scoring logic
        anomalies = []
        score = 0.0

        # Low biological sync (signals not correlated)
        if sync_score < 0.3:
            score += 0.4
            anomalies.append("LOW_BIOLOGICAL_SYNC")
        elif sync_score < 0.5:
            score += 0.2

        # Poor pulse coverage
        if pulse_coverage < 0.3:
            score += 0.3
            anomalies.append("WEAK_PULSE_COVERAGE")
        elif pulse_coverage < 0.5:
            score += 0.15

        # Inconsistent heart rates across ROIs
        if hr_consistency < 0.5:
            score += 0.2
            anomalies.append("INCONSISTENT_HR_ACROSS_ROIS")

        # Abnormal heart rate range
        if heart_rates:
            mean_hr = np.mean(heart_rates)
            if mean_hr < self.EXPECTED_HR_MIN or mean_hr > self.EXPECTED_HR_MAX:
                score += 0.1
                anomalies.append("ABNORMAL_HR_RANGE")

        score = min(score, 1.0)

        # Confidence adjustment for low-res
        if is_low_res:
            confidence = min(sync_score * 0.8, 0.6)  # Cap at 60% for low-res
        else:
            confidence = min(0.4 + sync_score * 0.4 + pulse_coverage * 0.2, 1.0)

        # Determine status
        if score > 0.6:
            status = "FAIL"
        elif score > 0.3:
            status = "WARN"
        else:
            status = "PASS"

        return BioSignalCoreResult(
            core_name=self.name,
            score=score,
            confidence=confidence,
            status=status,
            details=details,
            anomalies=anomalies,
            data_quality=data_quality,
            roi_signals=roi_signals,
            biological_sync_score=sync_score,
            cross_correlation_matrix=corr_matrix,
            pulse_coverage=pulse_coverage,
            hr_consistency=hr_consistency
        )
