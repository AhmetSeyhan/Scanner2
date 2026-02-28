"""
Scanner Prime - ARTIFACT CORE
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements generative model fingerprint detection and structural
integrity analysis using standard open-source libraries (numpy, OpenCV).
All algorithms are based on publicly available academic research on
GAN, Diffusion, and VAE artifact detection.

Key Features:
- GAN fingerprint detection via FFT grid artifact analysis
- Diffusion model detection via uniform noise patterns
- VAE detection via blur signatures from reconstruction loss
- Temporal warping analysis for structural integrity

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.forensic_types import (
    ArtifactCoreResult,
    FrameList,
    HeatmapAnalysis,
    HeatmapCell,
    VideoProfile,
)


class ArtifactCore:
    """
    ARTIFACT CORE - Generative Model Forensics

    Detects fingerprints left by different generative model architectures:
    - GANs leave periodic grid artifacts in frequency domain
    - Diffusion models leave uniform noise patterns in smooth regions
    - VAEs leave blur signatures from reconstruction loss

    Also analyzes structural integrity through optical flow.
    """

    # GAN detection frequencies (common upsampling patterns)
    GAN_ARTIFACT_FREQUENCIES = [8, 16, 32, 64]

    # Minimum frames required
    MIN_FRAMES_REQUIRED = 5

    # v3.2.0: Heatmap grid size for spatial analysis
    HEATMAP_GRID_SIZE = (8, 8)

    def __init__(self):
        self.name = "ARTIFACT CORE"

    # =========================================================================
    # v3.2.0 HEATMAP GENERATION METHODS
    # =========================================================================

    def generate_spatial_heatmap(
        self,
        frame: np.ndarray,
        frame_index: int = 0
    ) -> HeatmapAnalysis:
        """
        Generate spatial anomaly heatmap for a single frame.

        Divides frame into 8x8 grid and analyzes each cell for:
        - GAN grid artifacts (FFT analysis)
        - Diffusion noise patterns
        - VAE blur signatures

        Args:
            frame: Single video frame (BGR)
            frame_index: Frame index for reference

        Returns:
            HeatmapAnalysis with per-cell anomaly scores
        """
        rows, cols = self.HEATMAP_GRID_SIZE
        h, w = frame.shape[:2]
        cell_h, cell_w = h // rows, w // cols

        cells = []
        anomaly_types_count = {"GAN": 0, "DIFFUSION": 0, "VAE": 0, "NONE": 0}

        for row in range(rows):
            for col in range(cols):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = min(x1 + cell_w, w)
                y2 = min(y1 + cell_h, h)

                cell_region = frame[y1:y2, x1:x2]

                if cell_region.size < 64:
                    cells.append(HeatmapCell(
                        x=col, y=row, anomaly_score=0.0,
                        anomaly_type="NONE", confidence=0.0,
                        pixel_coords=(x1, y1, x2, y2)
                    ))
                    anomaly_types_count["NONE"] += 1
                    continue

                # Analyze cell for each artifact type
                gan_score = self._analyze_cell_gan(cell_region)
                diff_score = self._analyze_cell_diffusion(cell_region)
                vae_score = self._analyze_cell_vae(cell_region)

                # Determine dominant anomaly type
                max_score = max(gan_score, diff_score, vae_score)
                if max_score < 0.3:
                    anomaly_type = "NONE"
                elif gan_score == max_score:
                    anomaly_type = "GAN"
                elif diff_score == max_score:
                    anomaly_type = "DIFFUSION"
                else:
                    anomaly_type = "VAE"

                anomaly_types_count[anomaly_type] += 1

                cells.append(HeatmapCell(
                    x=col,
                    y=row,
                    anomaly_score=max_score,
                    anomaly_type=anomaly_type,
                    confidence=min(max_score * 1.5, 1.0),
                    pixel_coords=(x1, y1, x2, y2)
                ))

        # Find hotspot regions (score > 0.5)
        hotspots = [
            {"x": c.x, "y": c.y, "score": c.anomaly_score, "type": c.anomaly_type}
            for c in cells if c.anomaly_score > 0.5
        ]

        # Determine dominant type
        dominant_type = max(anomaly_types_count, key=anomaly_types_count.get)
        if anomaly_types_count[dominant_type] == anomaly_types_count["NONE"]:
            dominant_type = "NONE"

        overall_score = np.mean([c.anomaly_score for c in cells]) if cells else 0.0

        return HeatmapAnalysis(
            grid_size=(rows, cols),
            cells=cells,
            overall_anomaly_score=float(overall_score),
            hotspot_regions=hotspots,
            dominant_anomaly_type=dominant_type,
            frame_index=frame_index
        )

    def _analyze_cell_gan(self, cell: np.ndarray) -> float:
        """Analyze single cell for GAN artifacts using FFT."""
        if cell.size < 64:
            return 0.0

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
        f_transform = np.fft.fft2(gray.astype(np.float32))
        magnitude = np.abs(np.fft.fftshift(f_transform))

        h, w = magnitude.shape
        if h < 16 or w < 16:
            return 0.0

        center_y, center_x = h // 2, w // 2
        artifact_energy = 0
        total_energy = np.sum(magnitude) + 1e-6

        for freq in [4, 8, 16]:
            if freq < min(h, w) // 2:
                artifact_energy += magnitude[center_y, center_x + freq]
                artifact_energy += magnitude[center_y + freq, center_x]

        return min(artifact_energy / total_energy * 10, 1.0)

    def _analyze_cell_diffusion(self, cell: np.ndarray) -> float:
        """Analyze single cell for diffusion noise patterns."""
        if cell.size < 64:
            return 0.0

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell

        # Check noise uniformity
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        noise_kurtosis = self._compute_kurtosis(noise.flatten())

        # Diffusion leaves uniform (low kurtosis) noise
        uniformity = 1.0 / (1.0 + abs(noise_kurtosis - 3.0))
        return min(uniformity * 1.5, 1.0) if uniformity > 0.5 else 0.0

    def _analyze_cell_vae(self, cell: np.ndarray) -> float:
        """Analyze single cell for VAE blur signatures."""
        if cell.size < 64:
            return 0.0

        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell

        # Laplacian variance for blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)

        # Low variance = blurry (VAE signature)
        if blur_score < 50:
            return 0.8
        elif blur_score < 150:
            return 0.4
        return 0.0

    def generate_video_heatmaps(
        self,
        frames: FrameList,
        sample_rate: int = 10
    ) -> List[HeatmapAnalysis]:
        """
        Generate heatmaps for multiple frames.

        Args:
            frames: List of video frames
            sample_rate: Analyze every Nth frame

        Returns:
            List of HeatmapAnalysis for sampled frames
        """
        heatmaps = []
        for i in range(0, len(frames), sample_rate):
            heatmap = self.generate_spatial_heatmap(frames[i], frame_index=i)
            heatmaps.append(heatmap)
        return heatmaps

    # =========================================================================
    # ORIGINAL ARTIFACT DETECTION METHODS
    # =========================================================================

    def _detect_gan_fingerprint(self, frames: FrameList) -> Tuple[float, List[str]]:
        """
        Detect GAN-specific artifacts via FFT analysis.

        GANs using upsampling (transposed convolutions) often leave
        periodic artifacts at specific frequencies corresponding to
        their upsampling factors.

        Args:
            frames: List of video frames

        Returns:
            Tuple of (score, anomaly_flags)
        """
        anomalies = []
        grid_scores = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Compute 2D FFT
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2

            # Check for peaks at GAN artifact frequencies
            artifact_energy = 0
            total_energy = np.sum(magnitude) + 1e-6

            for freq in self.GAN_ARTIFACT_FREQUENCIES:
                if freq < min(h, w) // 2:
                    # Check horizontal and vertical lines at this frequency
                    h_line = magnitude[center_y, center_x - freq:center_x + freq + 1]
                    v_line = magnitude[center_y - freq:center_y + freq + 1, center_x]

                    # Also check diagonal patterns
                    diag_energy = 0
                    for offset in range(-2, 3):
                        if 0 <= center_y + freq + offset < h and 0 <= center_x + freq + offset < w:
                            diag_energy += magnitude[center_y + freq + offset, center_x + freq + offset]
                        if 0 <= center_y - freq + offset < h and 0 <= center_x + freq + offset < w:
                            diag_energy += magnitude[center_y - freq + offset, center_x + freq + offset]

                    freq_energy = np.sum(h_line) + np.sum(v_line) + diag_energy
                    artifact_energy += freq_energy

            # Normalize by total energy
            grid_score = artifact_energy / total_energy
            grid_scores.append(grid_score)

        if not grid_scores:
            return 0.0, []

        avg_grid_score = np.mean(grid_scores)
        grid_consistency = 1.0 - min(np.std(grid_scores) / (avg_grid_score + 1e-6), 1.0)

        # Score based on artifact presence
        score = 0.0
        if avg_grid_score > 0.15:
            score = 0.7
            anomalies.append("GAN_GRID_ARTIFACT_DETECTED")
        elif avg_grid_score > 0.08:
            score = 0.4
            anomalies.append("POSSIBLE_GAN_ARTIFACT")

        # High consistency of artifacts suggests GAN
        if grid_consistency > 0.8 and avg_grid_score > 0.05:
            score = min(score + 0.2, 1.0)
            if "GAN_GRID_ARTIFACT_DETECTED" not in anomalies:
                anomalies.append("CONSISTENT_FREQUENCY_PATTERN")

        return score, anomalies

    def _detect_diffusion_fingerprint(self, frames: FrameList) -> Tuple[float, List[str]]:
        """
        Detect Diffusion model-specific patterns.

        Diffusion models often leave uniform noise patterns in smooth
        regions due to their iterative denoising process. The noise
        tends to be more uniform than natural camera noise.

        Args:
            frames: List of video frames

        Returns:
            Tuple of (score, anomaly_flags)
        """
        anomalies = []
        uniformity_scores = []
        noise_patterns = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            h, w = gray.shape

            # Find smooth regions (low gradient areas)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Create mask for smooth regions
            smooth_mask = gradient_mag < np.percentile(gradient_mag, 30)

            if np.sum(smooth_mask) < 100:
                continue

            # Analyze noise in smooth regions
            high_pass = cv2.Laplacian(gray, cv2.CV_64F)
            smooth_noise = high_pass.flatten()[smooth_mask.flatten()]

            # Diffusion models tend to have more uniform noise distribution
            noise_std = np.std(smooth_noise)
            noise_kurtosis = self._compute_kurtosis(smooth_noise)

            # Uniformity: low kurtosis suggests uniform (non-Gaussian) noise
            uniformity = 1.0 / (1.0 + abs(noise_kurtosis - 3.0))  # Gaussian has kurtosis ~3
            uniformity_scores.append(uniformity)
            noise_patterns.append(noise_std)

        if not uniformity_scores:
            return 0.0, []

        avg_uniformity = np.mean(uniformity_scores)
        avg_noise = np.mean(noise_patterns)

        score = 0.0

        # High uniformity in smooth regions suggests diffusion
        if avg_uniformity > 0.7:
            score = 0.6
            anomalies.append("DIFFUSION_NOISE_SIGNATURE")
        elif avg_uniformity > 0.5:
            score = 0.3
            anomalies.append("POSSIBLE_DIFFUSION_PATTERN")

        # Very low noise in smooth regions is also suspicious
        if avg_noise < 2.0:
            score = min(score + 0.2, 1.0)
            anomalies.append("UNNATURALLY_SMOOTH_REGIONS")

        return score, anomalies

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        n = len(data)
        if n < 4:
            return 3.0  # Default Gaussian

        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-6:
            return 3.0

        m4 = np.mean((data - mean) ** 4)
        kurtosis = m4 / (std ** 4)
        return kurtosis

    def _detect_vae_fingerprint(self, frames: FrameList) -> Tuple[float, List[str]]:
        """
        Detect VAE-specific artifacts.

        VAEs using reconstruction loss often produce slightly blurred
        outputs, especially around fine details. This can be detected
        by analyzing high-frequency content loss.

        Args:
            frames: List of video frames

        Returns:
            Tuple of (score, anomaly_flags)
        """
        anomalies = []
        blur_scores = []
        detail_losses = []

        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Laplacian variance (blur detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = np.var(laplacian)
            blur_scores.append(blur_score)

            # High-frequency content via DFT
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2

            # High frequency region (outer ring)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), min(h, w) // 2, 1, -1)
            cv2.circle(mask, (center_x, center_y), min(h, w) // 4, 0, -1)

            high_freq_energy = np.sum(magnitude * mask)
            low_freq_energy = np.sum(magnitude * (1 - mask)) + 1e-6
            detail_ratio = high_freq_energy / low_freq_energy
            detail_losses.append(detail_ratio)

        if not blur_scores:
            return 0.0, []

        avg_blur = np.mean(blur_scores)
        avg_detail = np.mean(detail_losses)

        score = 0.0

        # Low blur score (high variance = sharp, low variance = blurry)
        if avg_blur < 100:
            score = 0.5
            anomalies.append("VAE_BLUR_SIGNATURE")
        elif avg_blur < 300:
            score = 0.25
            anomalies.append("POSSIBLE_VAE_BLUR")

        # Low high-frequency content
        if avg_detail < 0.3:
            score = min(score + 0.3, 1.0)
            anomalies.append("HIGH_FREQUENCY_LOSS")

        return score, anomalies

    def analyze_structural_integrity(self, frames: FrameList) -> Tuple[float, List[Dict]]:
        """
        Analyze structural integrity via optical flow.

        Looks for temporal warping in facial boundary regions
        (jawline, ears) which are common failure points in deepfakes.

        Args:
            frames: List of video frames

        Returns:
            Tuple of (warping_score, warping_events)
        """
        if len(frames) < 3:
            return 0.0, []

        warping_events = []
        boundary_inconsistencies = []

        prev_gray = None
        for i, frame in enumerate(frames):
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            if prev_gray is None:
                prev_gray = gray
                continue

            h, w = gray.shape

            # Focus on face boundary regions (jawline, sides)
            # Left boundary region
            left_region = (h // 4, 3 * h // 4, 0, w // 5)
            # Right boundary region
            right_region = (h // 4, 3 * h // 4, 4 * w // 5, w)
            # Bottom (jawline) region
            bottom_region = (3 * h // 4, h, w // 4, 3 * w // 4)

            for region_name, (y1, y2, x1, x2) in [
                ("left", left_region),
                ("right", right_region),
                ("jawline", bottom_region)
            ]:
                prev_roi = prev_gray[y1:y2, x1:x2]
                curr_roi = gray[y1:y2, x1:x2]

                if prev_roi.size == 0 or curr_roi.size == 0:
                    continue

                try:
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_roi, curr_roi, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )

                    # Analyze flow for anomalies
                    flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                    flow_dir = np.arctan2(flow[:, :, 1], flow[:, :, 0])

                    # Warping detection: high variance in flow direction
                    dir_variance = np.var(flow_dir)
                    mag_mean = np.mean(flow_mag)

                    # High directional variance with significant motion = warping
                    if dir_variance > 1.5 and mag_mean > 1.0:
                        warping_events.append({
                            "frame": i,
                            "region": region_name,
                            "direction_variance": float(dir_variance),
                            "magnitude": float(mag_mean)
                        })
                        boundary_inconsistencies.append(dir_variance)

                except Exception:
                    continue

            prev_gray = gray

        # Calculate warping score
        if boundary_inconsistencies:
            avg_inconsistency = np.mean(boundary_inconsistencies)
            warping_score = min(avg_inconsistency / 3.0, 1.0)
        else:
            warping_score = 0.0

        return warping_score, warping_events

    def analyze(
        self,
        frames: FrameList,
        video_profile: Optional[VideoProfile] = None
    ) -> ArtifactCoreResult:
        """
        Run complete generative model forensics analysis.

        Args:
            frames: List of video frames
            video_profile: Optional video profile for adaptive processing

        Returns:
            ArtifactCoreResult with fingerprint analysis
        """
        if len(frames) < self.MIN_FRAMES_REQUIRED:
            return ArtifactCoreResult(
                core_name=self.name,
                score=0.5,
                confidence=0.3,
                status="WARN",
                details={"reason": "Insufficient frames"},
                anomalies=["LOW_FRAME_COUNT"],
                data_quality="INSUFFICIENT"
            )

        # Detect fingerprints from different model types
        gan_score, gan_anomalies = self._detect_gan_fingerprint(frames)
        diffusion_score, diffusion_anomalies = self._detect_diffusion_fingerprint(frames)
        vae_score, vae_anomalies = self._detect_vae_fingerprint(frames)

        # Analyze structural integrity
        warping_score, warping_events = self.analyze_structural_integrity(frames)

        # Combine all anomalies
        all_anomalies = gan_anomalies + diffusion_anomalies + vae_anomalies
        if warping_score > 0.4:
            all_anomalies.append("TEMPORAL_WARPING_DETECTED")

        # Build detected fingerprints list
        detected_fingerprints = []
        if gan_score > 0.3:
            detected_fingerprints.append({
                "type": "GAN",
                "score": round(gan_score, 4),
                "anomalies": gan_anomalies
            })
        if diffusion_score > 0.3:
            detected_fingerprints.append({
                "type": "DIFFUSION",
                "score": round(diffusion_score, 4),
                "anomalies": diffusion_anomalies
            })
        if vae_score > 0.3:
            detected_fingerprints.append({
                "type": "VAE",
                "score": round(vae_score, 4),
                "anomalies": vae_anomalies
            })

        # Overall score: max of individual scores + warping
        base_score = max(gan_score, diffusion_score, vae_score)
        overall_score = min(base_score + warping_score * 0.3, 1.0)

        # Details
        details = {
            "gan_score": round(gan_score, 4),
            "diffusion_score": round(diffusion_score, 4),
            "vae_score": round(vae_score, 4),
            "structural_integrity_score": round(1.0 - warping_score, 4),
            "warping_events_count": len(warping_events),
            "detected_model_type": self._determine_model_type(gan_score, diffusion_score, vae_score),
        }

        if warping_events:
            details["warping_events"] = warping_events[:5]  # First 5

        # Confidence based on signal strength
        confidence = min(0.5 + overall_score * 0.4, 0.9)
        if len(frames) < 20:
            confidence *= 0.8

        # Adjust for low-res (neural fingerprinting is actually more reliable for low-res)
        if video_profile and video_profile.is_low_res:
            confidence = min(confidence * 1.1, 0.95)  # Slight boost

        # Determine status
        if overall_score > 0.6:
            status = "FAIL"
        elif overall_score > 0.3:
            status = "WARN"
        else:
            status = "PASS"

        return ArtifactCoreResult(
            core_name=self.name,
            score=overall_score,
            confidence=confidence,
            status=status,
            details=details,
            anomalies=all_anomalies,
            data_quality="GOOD",  # Works at any resolution
            gan_score=gan_score,
            diffusion_score=diffusion_score,
            vae_score=vae_score,
            structural_integrity=1.0 - warping_score,
            detected_fingerprints=detected_fingerprints,
            temporal_warping_detected=warping_score > 0.4
        )

    def _determine_model_type(
        self,
        gan_score: float,
        diffusion_score: float,
        vae_score: float
    ) -> str:
        """Determine most likely generative model type based on scores."""
        max_score = max(gan_score, diffusion_score, vae_score)
        if max_score < 0.3:
            return "NONE_DETECTED"

        if gan_score == max_score:
            return "GAN"
        elif diffusion_score == max_score:
            return "DIFFUSION"
        else:
            return "VAE"
