"""
Scanner - Gaze & Eye Movement Forensic Analyzer (v5.1.0)

Analyzes eye movement patterns and gaze behavior to detect deepfakes.
Deepfakes often exhibit unnatural gaze characteristics:
- Abnormal blink rates (too fast, too slow, or absent)
- Asymmetric blinking (one eye blinks without the other)
- Unnatural saccade velocities
- Gaze direction discontinuities between frames

Uses MediaPipe Face Mesh for landmark extraction.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GazeAnalysisResult:
    """Result from gaze and eye movement analysis."""
    score: float                    # 0.0 (natural) -> 1.0 (anomalous)
    confidence: float               # 0.0 -> 1.0
    blink_rate: float               # Blinks per minute
    blink_symmetry: float           # 0.0 (asymmetric) -> 1.0 (symmetric)
    gaze_stability: float           # 0.0 (unstable) -> 1.0 (stable)
    saccade_velocity_score: float   # Naturalness of eye movements
    anomalies: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class GazeAnalyzer:
    """Eye movement and gaze pattern forensic analysis engine.

    Uses MediaPipe Face Mesh (468 landmarks) to extract eye and iris
    positions, then analyzes temporal patterns for anomalies.
    """

    # MediaPipe Face Mesh landmark indices for Eye Aspect Ratio (EAR)
    # Based on Soukupova & Cech (2016) real-time eye blink detection
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Iris landmarks (MediaPipe 478-point mesh)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # Normal physiological ranges
    NORMAL_BLINK_RATE_MIN = 10      # blinks per minute
    NORMAL_BLINK_RATE_MAX = 25
    EAR_BLINK_THRESHOLD = 0.21      # Below this = eye closed
    EAR_CONSEC_FRAMES = 2           # Minimum consecutive frames for blink
    MIN_FRAMES_REQUIRED = 15

    def __init__(self) -> None:
        self._face_mesh = None

    def _ensure_mesh(self) -> None:
        """Lazy-load MediaPipe Face Mesh."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,  # Enables iris landmarks (478 total)
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._face_mesh = None

    def _compute_ear(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Returns a value near 0.3 for open eyes and near 0.05 for closed.
        """
        try:
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            p3 = landmarks[indices[2]]
            p4 = landmarks[indices[3]]
            p5 = landmarks[indices[4]]
            p6 = landmarks[indices[5]]

            # Vertical distances
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            # Horizontal distance
            h = np.linalg.norm(p1 - p4)

            if h < 1e-6:
                return 0.0
            return float((v1 + v2) / (2.0 * h))
        except (IndexError, ValueError):
            return 0.3  # Default open eye

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract face mesh landmarks from a frame."""
        import cv2

        self._ensure_mesh()
        if self._face_mesh is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face.landmark
        ])
        return landmarks

    def _compute_gaze_direction(self, landmarks: np.ndarray) -> Optional[Tuple[float, float]]:
        """Estimate gaze direction from iris position relative to eye corners.

        Returns (horizontal_ratio, vertical_ratio) where:
        - horizontal: 0.0 = looking left, 0.5 = center, 1.0 = looking right
        - vertical: 0.0 = looking up, 0.5 = center, 1.0 = looking down
        """
        try:
            if len(landmarks) < 478:
                return None

            # Left iris center
            left_iris = np.mean(landmarks[self.LEFT_IRIS], axis=0)
            # Right iris center
            right_iris = np.mean(landmarks[self.RIGHT_IRIS], axis=0)

            # Eye corners for reference
            left_outer = landmarks[33]
            left_inner = landmarks[133]
            right_inner = landmarks[362]
            right_outer = landmarks[263]

            # Horizontal ratio (left eye)
            eye_width = np.linalg.norm(left_inner[:2] - left_outer[:2])
            if eye_width < 1e-6:
                return None

            h_ratio_l = float(np.linalg.norm(left_iris[:2] - left_outer[:2]) / eye_width)

            # Horizontal ratio (right eye)
            eye_width_r = np.linalg.norm(right_outer[:2] - right_inner[:2])
            if eye_width_r < 1e-6:
                return None

            h_ratio_r = float(np.linalg.norm(right_iris[:2] - right_inner[:2]) / eye_width_r)

            h_ratio = (h_ratio_l + h_ratio_r) / 2.0

            # Vertical ratio (using upper/lower eye landmarks)
            upper = landmarks[159]  # Upper eyelid
            lower = landmarks[145]  # Lower eyelid
            eye_height = np.linalg.norm(upper[:2] - lower[:2])
            if eye_height < 1e-6:
                return None

            v_ratio = float(np.linalg.norm(left_iris[:2] - upper[:2]) / eye_height)

            return (np.clip(h_ratio, 0, 1), np.clip(v_ratio, 0, 1))
        except (IndexError, ValueError):
            return None

    def analyze(self, frames: List[np.ndarray], fps: float) -> GazeAnalysisResult:
        """Analyze eye movement patterns across video frames.

        Args:
            frames: List of video frames (BGR format)
            fps: Frames per second

        Returns:
            GazeAnalysisResult with blink rate, symmetry, and gaze metrics
        """
        if not frames or len(frames) < self.MIN_FRAMES_REQUIRED:
            return GazeAnalysisResult(
                score=0.5, confidence=0.0,
                blink_rate=0.0, blink_symmetry=1.0,
                gaze_stability=1.0, saccade_velocity_score=1.0,
                anomalies=["Insufficient frames for gaze analysis"],
            )

        if fps <= 0:
            fps = 30.0

        # Sample frames evenly (max 120 for performance)
        max_analyze = 120
        if len(frames) > max_analyze:
            indices = np.linspace(0, len(frames) - 1, max_analyze, dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = frames

        # Extract landmarks and compute metrics per frame
        left_ears = []
        right_ears = []
        gaze_directions = []
        frames_with_face = 0

        for frame in sampled:
            landmarks = self._extract_landmarks(frame)
            if landmarks is None:
                left_ears.append(None)
                right_ears.append(None)
                continue

            frames_with_face += 1
            left_ear = self._compute_ear(landmarks, self.LEFT_EYE)
            right_ear = self._compute_ear(landmarks, self.RIGHT_EYE)
            left_ears.append(left_ear)
            right_ears.append(right_ear)

            gaze = self._compute_gaze_direction(landmarks)
            if gaze is not None:
                gaze_directions.append(gaze)

        if frames_with_face < self.MIN_FRAMES_REQUIRED:
            return GazeAnalysisResult(
                score=0.5, confidence=0.2,
                blink_rate=0.0, blink_symmetry=1.0,
                gaze_stability=1.0, saccade_velocity_score=1.0,
                anomalies=["Insufficient face detections for gaze analysis"],
                details={"frames_with_face": frames_with_face, "total_frames": len(sampled)},
            )

        # --- Blink Analysis ---
        valid_left = np.array([e for e in left_ears if e is not None])
        valid_right = np.array([e for e in right_ears if e is not None])

        blink_count, blink_symmetry = self._analyze_blinks(
            valid_left, valid_right, fps
        )

        duration_sec = len(sampled) / fps
        blink_rate = (blink_count / duration_sec) * 60 if duration_sec > 0 else 0

        # --- Gaze Analysis ---
        gaze_stability = 1.0
        saccade_score = 1.0
        if len(gaze_directions) >= 5:
            gaze_stability, saccade_score = self._analyze_gaze(gaze_directions, fps)

        # --- Anomaly Detection ---
        anomalies = []
        score_components = []

        # Blink rate check
        if blink_rate < self.NORMAL_BLINK_RATE_MIN and duration_sec > 3:
            anomalies.append(f"Low blink rate ({blink_rate:.1f}/min, normal: {self.NORMAL_BLINK_RATE_MIN}-{self.NORMAL_BLINK_RATE_MAX})")
            deficit = (self.NORMAL_BLINK_RATE_MIN - blink_rate) / self.NORMAL_BLINK_RATE_MIN
            score_components.append(min(deficit, 1.0))
        elif blink_rate > self.NORMAL_BLINK_RATE_MAX * 2:
            anomalies.append(f"Abnormally high blink rate ({blink_rate:.1f}/min)")
            score_components.append(0.6)

        # Blink symmetry check
        if blink_symmetry < 0.6:
            anomalies.append(f"Asymmetric blinking (symmetry: {blink_symmetry:.2f})")
            score_components.append((1.0 - blink_symmetry) * 0.8)

        # Gaze stability check
        if gaze_stability < 0.4:
            anomalies.append(f"Unstable gaze pattern (stability: {gaze_stability:.2f})")
            score_components.append((1.0 - gaze_stability) * 0.6)

        # Saccade velocity check
        if saccade_score < 0.5:
            anomalies.append(f"Unnatural eye movement velocity (score: {saccade_score:.2f})")
            score_components.append((1.0 - saccade_score) * 0.7)

        # Final score
        if score_components:
            score = float(np.clip(np.mean(score_components), 0.0, 1.0))
        else:
            score = 0.1  # Low score if no anomalies

        # Confidence based on data quality
        face_ratio = frames_with_face / len(sampled)
        confidence = min(face_ratio * 0.7 + 0.3, 1.0)

        return GazeAnalysisResult(
            score=score,
            confidence=confidence,
            blink_rate=round(blink_rate, 1),
            blink_symmetry=round(blink_symmetry, 3),
            gaze_stability=round(gaze_stability, 3),
            saccade_velocity_score=round(saccade_score, 3),
            anomalies=anomalies,
            details={
                "frames_analyzed": len(sampled),
                "frames_with_face": frames_with_face,
                "blink_count": blink_count,
                "duration_seconds": round(duration_sec, 2),
                "gaze_points": len(gaze_directions),
            },
        )

    def _analyze_blinks(
        self, left_ear: np.ndarray, right_ear: np.ndarray, fps: float
    ) -> Tuple[int, float]:
        """Analyze blink patterns and symmetry.

        Returns:
            (blink_count, symmetry_score)
        """
        if len(left_ear) < 5 or len(right_ear) < 5:
            return 0, 1.0

        # Detect blinks as consecutive frames below EAR threshold
        avg_ear = (left_ear + right_ear) / 2
        below_threshold = avg_ear < self.EAR_BLINK_THRESHOLD

        blink_count = 0
        in_blink = False
        consec = 0
        for is_closed in below_threshold:
            if is_closed:
                consec += 1
                if consec >= self.EAR_CONSEC_FRAMES and not in_blink:
                    blink_count += 1
                    in_blink = True
            else:
                in_blink = False
                consec = 0

        # Symmetry: correlation between left and right EAR sequences
        if np.std(left_ear) > 1e-6 and np.std(right_ear) > 1e-6:
            correlation = np.corrcoef(left_ear, right_ear)[0, 1]
            symmetry = max(0.0, float(correlation))
        else:
            symmetry = 1.0

        return blink_count, symmetry

    def _analyze_gaze(
        self, directions: List[Tuple[float, float]], fps: float
    ) -> Tuple[float, float]:
        """Analyze gaze stability and saccade velocity.

        Returns:
            (stability_score, saccade_score) both 0-1
        """
        if len(directions) < 3:
            return 1.0, 1.0

        h_positions = np.array([d[0] for d in directions])
        v_positions = np.array([d[1] for d in directions])

        # Stability: inverse of position variance
        h_std = float(np.std(h_positions))
        v_std = float(np.std(v_positions))
        stability = float(np.clip(1.0 - (h_std + v_std) * 2, 0.0, 1.0))

        # Saccade velocity: frame-to-frame gaze shifts
        h_diff = np.diff(h_positions)
        v_diff = np.diff(v_positions)
        velocities = np.sqrt(h_diff ** 2 + v_diff ** 2) * fps

        if len(velocities) == 0:
            return stability, 1.0

        # Natural saccades have specific velocity distributions
        # Very high velocities or no velocity variation are suspicious
        mean_vel = float(np.mean(velocities))
        std_vel = float(np.std(velocities))

        # Score: penalize unnaturally uniform velocities or extreme velocities
        if std_vel < 0.01 and mean_vel > 0.1:
            saccade_score = 0.3  # Too uniform
        elif mean_vel > 5.0:
            saccade_score = 0.2  # Too fast
        else:
            saccade_score = float(np.clip(1.0 - mean_vel * 0.1, 0.3, 1.0))

        return stability, saccade_score
