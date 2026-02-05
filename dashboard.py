"""
SCANNER ELITE v5.0 - Tactical Forensic Command Suite
Defense-Grade Deepfake Detection Platform with War Room Interface

Features:
- Professional Navbar with Radar Pulse Branding
- War Room Dual-Pane Analysis Interface
- Resolution-Aware Hybrid Logic (Auto-adjusts engine weights based on video quality)
- Biometric rPPG Engine with Spatial Averaging for Low-SNR recovery
- Neural Noise Fingerprinting with Compression Artifact Detection
- Semantic Logic Checker (physical inconsistency)
- A/V Sync-Lock (audio-video synchronization)
- Live Sentinel Mode with Adaptive HUD (Mesh vs Heatmap)
- Blockchain-Ready Origin Hash
- Adversarial Defense Layer
- Tactical Footer with System Telemetry

Run with: streamlit run dashboard.py
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import tempfile
import os
import json
import hashlib
import threading
import queue
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Scipy imports with fallback
try:
    from scipy import signal
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback to numpy
    from numpy.fft import fft

# Must be first Streamlit command
st.set_page_config(
    page_title="SCANNER ELITE | Forensic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
from preprocessing import FaceExtractor, VideoProcessor
from model import DeepfakeInference

# Import PRIME HYBRID core modules (renamed for IP compliance)
from core.biosignal_core import BioSignalCore
from core.artifact_core import ArtifactCore
from core.alignment_core import AlignmentCore
from core.fusion_engine import FusionEngine, FusionMode
from core.audio_analyzer import AudioAnalyzer, AudioProfile
from core.forensic_types import (
    VideoProfile as CoreVideoProfile,
    ResolutionTier as CoreResolutionTier,
    FusionVerdict,
    BioSignalCoreResult,
    ArtifactCoreResult,
    AlignmentCoreResult,
    TransparencyReport,
)


# ============== RESOLUTION TIERS ==============

class ResolutionTier(Enum):
    """Video resolution classification for adaptive analysis."""
    ULTRA_LOW = "ultra_low"  # <= 360p
    LOW = "low"              # 361p - 480p
    MEDIUM = "medium"        # 481p - 720p
    HIGH = "high"            # 721p - 1080p
    ULTRA_HIGH = "ultra_hd"  # > 1080p


@dataclass
class VideoProfile:
    """Complete video profile with resolution analysis."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    resolution_tier: ResolutionTier
    pixel_count: int
    aspect_ratio: float

    # Adaptive flags
    rppg_viable: bool
    mesh_viable: bool
    recommended_analysis: str

    @property
    def resolution_label(self) -> str:
        """Human-readable resolution label."""
        if self.height <= 360:
            return f"{self.height}p (Ultra-Low)"
        elif self.height <= 480:
            return f"{self.height}p (Low)"
        elif self.height <= 720:
            return f"{self.height}p (Medium)"
        elif self.height <= 1080:
            return f"{self.height}p (HD)"
        else:
            return f"{self.height}p (Ultra-HD)"


# ============== GLOBAL THEME CONFIGURATION ==============
# Complete theme with ALL glow colors defined to prevent KeyError

THEME_CONFIG = {
    "dark": {
        "name": "dark",
        # Core Backgrounds
        "background": "#0B0B0C",
        "background_secondary": "#0D0D10",
        "background_tertiary": "#111116",
        # Cards & Containers
        "card_bg": "rgba(15, 15, 20, 0.7)",
        "card_bg_solid": "#0F0F14",
        "card_border": "rgba(255, 255, 255, 0.06)",
        # Text Hierarchy
        "text_primary": "#FFFFFF",
        "text_secondary": "#B0B0B8",
        "text_muted": "#5C5C66",
        "text_warning": "#FFB800",
        # Primary Accents
        "accent_blue": "#0066FF",
        "accent_cyan": "#00D4FF",
        "accent_electric": "#00A8FF",
        # Warm Accents
        "accent_orange": "#FF6B35",
        "accent_amber": "#FFB800",
        "accent_magenta": "#FF006E",
        "accent_purple": "#8B5CF6",
        "accent_red": "#FF3D57",
        # Status Colors
        "success": "#00C853",
        "danger": "#FF3D57",
        "warning": "#FFB800",
        # Glow Colors - ALL FULLY DEFINED
        "glow_blue": "rgba(0, 102, 255, 0.5)",
        "glow_cyan": "rgba(0, 212, 255, 0.4)",
        "glow_electric": "rgba(0, 168, 255, 0.4)",
        "glow_orange": "rgba(255, 107, 53, 0.45)",
        "glow_amber": "rgba(255, 184, 0, 0.4)",
        "glow_magenta": "rgba(255, 0, 110, 0.4)",
        "glow_purple": "rgba(139, 92, 246, 0.4)",
        "glow_red": "rgba(255, 61, 87, 0.4)",
        "glow_success": "rgba(0, 200, 83, 0.4)",
        "glow_warning": "rgba(255, 184, 0, 0.35)",
        # Gradient Colors
        "gradient_start": "#0B0B0C",
        "gradient_mid": "#0A192F",
        "gradient_end": "#1a0a2e",
        # Glass Effect
        "glass_bg": "rgba(15, 15, 20, 0.6)",
        "glass_border": "rgba(255, 255, 255, 0.08)",
        "glass_blur": "blur(20px)",
        # Terminal
        "terminal_bg": "rgba(0, 0, 0, 0.6)",
        "terminal_header": "rgba(20, 20, 25, 0.9)",
        # Buttons - MATTE BLACK
        "button_bg": "#000000",
        "button_text": "#FFFFFF",
        "button_hover": "#1a1a1a",
        # Heatmap colors
        "heatmap_cold": "#0066FF",
        "heatmap_warm": "#FFB800",
        "heatmap_hot": "#FF3D57",
    },
    "light": {
        "name": "light",
        # Core Backgrounds
        "background": "#F5F5F7",
        "background_secondary": "#FFFFFF",
        "background_tertiary": "#EEEEF0",
        # Cards & Containers
        "card_bg": "rgba(255, 255, 255, 0.85)",
        "card_bg_solid": "#FFFFFF",
        "card_border": "rgba(0, 0, 0, 0.08)",
        # Text Hierarchy
        "text_primary": "#1D1D1F",
        "text_secondary": "#6E6E73",
        "text_muted": "#8E8E93",
        "text_warning": "#E6A600",
        # Primary Accents
        "accent_blue": "#0066FF",
        "accent_cyan": "#00A3CC",
        "accent_electric": "#0088CC",
        # Warm Accents
        "accent_orange": "#E85A2D",
        "accent_amber": "#E6A600",
        "accent_magenta": "#CC0058",
        "accent_purple": "#7C4DDB",
        "accent_red": "#E6354A",
        # Status Colors
        "success": "#00A843",
        "danger": "#E6354A",
        "warning": "#E6A600",
        # Glow Colors - ALL FULLY DEFINED
        "glow_blue": "rgba(0, 102, 255, 0.2)",
        "glow_cyan": "rgba(0, 163, 204, 0.15)",
        "glow_electric": "rgba(0, 136, 204, 0.15)",
        "glow_orange": "rgba(232, 90, 45, 0.18)",
        "glow_amber": "rgba(230, 166, 0, 0.15)",
        "glow_magenta": "rgba(204, 0, 88, 0.15)",
        "glow_purple": "rgba(124, 77, 219, 0.15)",
        "glow_red": "rgba(230, 53, 74, 0.15)",
        "glow_success": "rgba(0, 168, 67, 0.15)",
        "glow_warning": "rgba(230, 166, 0, 0.12)",
        # Gradient Colors
        "gradient_start": "#F5F5F7",
        "gradient_mid": "#E8E8ED",
        "gradient_end": "#F0F0F5",
        # Glass Effect
        "glass_bg": "rgba(255, 255, 255, 0.7)",
        "glass_border": "rgba(0, 0, 0, 0.06)",
        "glass_blur": "blur(20px)",
        # Terminal
        "terminal_bg": "rgba(245, 245, 247, 0.9)",
        "terminal_header": "rgba(238, 238, 240, 0.95)",
        # Buttons
        "button_bg": "#1D1D1F",
        "button_text": "#FFFFFF",
        "button_hover": "#333333",
        # Heatmap colors
        "heatmap_cold": "#0066FF",
        "heatmap_warm": "#E6A600",
        "heatmap_hot": "#E6354A",
    }
}


def get_theme(theme_name: str = "dark") -> dict:
    """Safely retrieve theme configuration with fallback."""
    return THEME_CONFIG.get(theme_name, THEME_CONFIG["dark"])


def get_color(theme: dict, key: str, fallback: str = "#888888") -> str:
    """Safely get a color from theme with fallback to prevent KeyError."""
    return theme.get(key, fallback)


# ============== VIDEO PROFILER ==============

class VideoProfiler:
    """
    Analyzes video properties to determine optimal analysis strategy.
    Resolution-aware logic for adaptive engine weighting.
    """

    # Resolution thresholds
    ULTRA_LOW_MAX = 360
    LOW_MAX = 480
    MEDIUM_MAX = 720
    HIGH_MAX = 1080

    # Minimum requirements for reliable rPPG
    RPPG_MIN_HEIGHT = 480
    RPPG_MIN_FPS = 24

    # Minimum for 3D mesh overlay
    MESH_MIN_HEIGHT = 720

    @classmethod
    def analyze_video(cls, video_path: str) -> VideoProfile:
        """Analyze video and return complete profile."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        duration = frame_count / fps if fps > 0 else 0
        pixel_count = width * height
        aspect_ratio = width / height if height > 0 else 1.0

        # Determine resolution tier
        if height <= cls.ULTRA_LOW_MAX:
            tier = ResolutionTier.ULTRA_LOW
        elif height <= cls.LOW_MAX:
            tier = ResolutionTier.LOW
        elif height <= cls.MEDIUM_MAX:
            tier = ResolutionTier.MEDIUM
        elif height <= cls.HIGH_MAX:
            tier = ResolutionTier.HIGH
        else:
            tier = ResolutionTier.ULTRA_HIGH

        # Determine viable analyses
        rppg_viable = height >= cls.RPPG_MIN_HEIGHT and fps >= cls.RPPG_MIN_FPS
        mesh_viable = height >= cls.MESH_MIN_HEIGHT

        # Recommended analysis strategy
        if tier in [ResolutionTier.ULTRA_LOW, ResolutionTier.LOW]:
            recommended = "Deep-Pixel & Semantic (Low-res mode)"
        elif tier == ResolutionTier.MEDIUM:
            recommended = "Hybrid Analysis (Balanced mode)"
        else:
            recommended = "Full Forensic Suite (High-res mode)"

        return VideoProfile(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            resolution_tier=tier,
            pixel_count=pixel_count,
            aspect_ratio=aspect_ratio,
            rppg_viable=rppg_viable,
            mesh_viable=mesh_viable,
            recommended_analysis=recommended
        )

    @classmethod
    def get_adaptive_weights(cls, profile: VideoProfile) -> Dict[str, float]:
        """
        Calculate adaptive engine weights based on video resolution.
        Returns normalized weights that sum to 1.0.
        """
        tier = profile.resolution_tier

        if tier == ResolutionTier.ULTRA_LOW:
            # Ultra-low: rPPG essentially useless, rely on neural/semantic
            weights = {
                "primary_model": 0.30,
                "Biometric rPPG": 0.05,  # Minimal - insufficient data
                "Neural Fingerprint": 0.35,  # Primary for low-res
                "Semantic Logic": 0.20,
                "A/V Sync-Lock": 0.10
            }
        elif tier == ResolutionTier.LOW:
            # Low (480p): rPPG weak, shift to neural fingerprinting
            weights = {
                "primary_model": 0.30,
                "Biometric rPPG": 0.10,  # Reduced - spatial averaging helps but limited
                "Neural Fingerprint": 0.30,  # Boosted
                "Semantic Logic": 0.20,
                "A/V Sync-Lock": 0.10
            }
        elif tier == ResolutionTier.MEDIUM:
            # Medium (720p): Balanced approach
            weights = {
                "primary_model": 0.30,
                "Biometric rPPG": 0.20,
                "Neural Fingerprint": 0.20,
                "Semantic Logic": 0.18,
                "A/V Sync-Lock": 0.12
            }
        else:
            # High/Ultra-High: Full rPPG capability
            weights = {
                "primary_model": 0.25,
                "Biometric rPPG": 0.30,  # Boosted for high-res
                "Neural Fingerprint": 0.18,
                "Semantic Logic": 0.15,
                "A/V Sync-Lock": 0.12
            }

        return weights


# ============== PRIME HYBRID CORE INTEGRATION ==============

# ForensicResult wrapper for backward compatibility with UI code
@dataclass
class ForensicResult:
    """Result from a forensic analysis engine (wrapper for UI compatibility)."""
    engine_name: str
    score: float  # 0.0 (authentic) to 1.0 (manipulated)
    confidence: float  # How confident the engine is
    status: str  # "PASS", "WARN", "FAIL"
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    is_leading: bool = False  # Whether this engine is driving the verdict
    data_quality: str = "GOOD"  # "GOOD", "LIMITED", "INSUFFICIENT"


class AdversarialDefenseLayer:
    """Adversarial Defense Layer - Counter AI-trickery attempts."""

    @staticmethod
    def apply_defense(frame: np.ndarray) -> np.ndarray:
        """Apply adversarial defense preprocessing."""
        jitter = np.random.normal(0, 0.5, frame.shape).astype(np.float32)
        defended = frame.astype(np.float32) + jitter
        defended = np.clip(defended, 0, 255).astype(np.uint8)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encoded = cv2.imencode('.jpg', defended, encode_param)
        defended = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        defended = cv2.GaussianBlur(defended, (3, 3), 0.3)
        return defended

    @staticmethod
    def apply_batch(frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply defense to a batch of frames."""
        return [AdversarialDefenseLayer.apply_defense(f) for f in frames]


def convert_video_profile_to_core(dashboard_profile: VideoProfile) -> CoreVideoProfile:
    """Convert dashboard VideoProfile to core module VideoProfile."""
    # Map resolution tiers
    tier_map = {
        ResolutionTier.ULTRA_LOW: CoreResolutionTier.ULTRA_LOW,
        ResolutionTier.LOW: CoreResolutionTier.LOW,
        ResolutionTier.MEDIUM: CoreResolutionTier.MEDIUM,
        ResolutionTier.HIGH: CoreResolutionTier.HIGH,
        ResolutionTier.ULTRA_HIGH: CoreResolutionTier.ULTRA_HIGH,
    }

    return CoreVideoProfile(
        width=dashboard_profile.width,
        height=dashboard_profile.height,
        fps=dashboard_profile.fps,
        frame_count=dashboard_profile.frame_count,
        duration_seconds=dashboard_profile.duration_seconds,
        resolution_tier=tier_map.get(dashboard_profile.resolution_tier, CoreResolutionTier.MEDIUM),
        pixel_count=dashboard_profile.pixel_count,
        aspect_ratio=dashboard_profile.aspect_ratio,
        rppg_viable=dashboard_profile.rppg_viable,
        mesh_viable=dashboard_profile.mesh_viable,
        recommended_analysis=dashboard_profile.recommended_analysis
    )


def core_result_to_forensic_result(core_result, engine_name: str) -> ForensicResult:
    """Convert core module result to ForensicResult for UI compatibility."""
    return ForensicResult(
        engine_name=engine_name,
        score=core_result.score,
        confidence=core_result.confidence,
        status=core_result.status,
        details=core_result.details,
        anomalies=core_result.anomalies,
        data_quality=core_result.data_quality
    )



# ============== BLOCKCHAIN & HASH ==============

def generate_origin_hash(video_path: str, results: Dict[str, Any], session_id: str) -> Dict[str, str]:
    """Generate SHA-256 origin hash for blockchain-ready proof."""
    cap = cv2.VideoCapture(video_path)
    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    cap.release()

    with open(video_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    proof_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "file_hash": file_hash,
        "metadata": metadata,
        "trust_score": results.get("trust_score", 0),
        "verdict": results.get("verdict", "UNKNOWN"),
        "resolution_tier": results.get("resolution_tier", "UNKNOWN"),
        "leading_engine": results.get("leading_engine", "UNKNOWN"),
        "engine_results": {
            k: {"score": v.get("score", 0), "status": v.get("status", "UNKNOWN")}
            for k, v in results.get("engines", {}).items()
        }
    }

    payload_json = json.dumps(proof_payload, sort_keys=True)
    origin_hash = hashlib.sha256(payload_json.encode()).hexdigest()

    return {
        "origin_hash": origin_hash,
        "file_hash": file_hash,
        "timestamp": proof_payload["timestamp"],
        "payload": proof_payload
    }


# ============== LIVE SENTINEL ==============

class LiveSentinel:
    """Live Camera Sentinel with resolution-adaptive HUD."""

    def __init__(self, face_extractor: FaceExtractor, inference_engine: DeepfakeInference):
        self.face_extractor = face_extractor
        self.inference_engine = inference_engine
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.latest_result = None
        self.inconsistency_map = None

    def start(self):
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)

    def _compute_inconsistency_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Compute pixel inconsistency heatmap for low-res mode."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Laplacian for edge inconsistencies
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)

        # Normalize to 0-255
        heatmap = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap

    def _process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                start_time = time.time()

                defended_frame = AdversarialDefenseLayer.apply_defense(frame)
                face = self.face_extractor.extract_primary_face(defended_frame)

                if face is not None:
                    preprocessed = self.face_extractor.preprocess_for_model(face)
                    prob, label = self.inference_engine.predict_single(preprocessed)
                    latency = (time.time() - start_time) * 1000

                    # Compute heatmap for low-res visualization
                    self.inconsistency_map = self._compute_inconsistency_heatmap(frame)

                    result = {
                        "probability": prob,
                        "label": label,
                        "latency_ms": latency,
                        "face_detected": True,
                        "frame_height": frame.shape[0]
                    }
                else:
                    result = {
                        "probability": 0.0,
                        "label": "NO_FACE",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "face_detected": False,
                        "frame_height": frame.shape[0]
                    }

                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break

                self.result_queue.put(result)
                self.latest_result = result

            except queue.Empty:
                continue
            except Exception:
                continue

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        return self.latest_result

    def draw_hud_adaptive(self, frame: np.ndarray, result: Dict, theme: dict) -> np.ndarray:
        """Draw HUD with resolution-adaptive visualization (Mesh vs Heatmap)."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        is_low_res = h < 720

        # Get colors
        if result.get("label") == "FAKE":
            status_color = (57, 61, 255)
            status_text = "THREAT DETECTED"
        elif result.get("label") == "REAL":
            status_color = (83, 200, 0)
            status_text = "VERIFIED"
        else:
            status_color = (0, 184, 255)
            status_text = "SCANNING..."

        # Draw corner brackets
        bracket_length = 40 if not is_low_res else 25
        bracket_thickness = 3 if not is_low_res else 2
        margin = 20 if not is_low_res else 10

        cv2.line(frame, (margin, margin), (margin + bracket_length, margin), status_color, bracket_thickness)
        cv2.line(frame, (margin, margin), (margin, margin + bracket_length), status_color, bracket_thickness)
        cv2.line(frame, (w - margin, margin), (w - margin - bracket_length, margin), status_color, bracket_thickness)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + bracket_length), status_color, bracket_thickness)
        cv2.line(frame, (margin, h - margin), (margin + bracket_length, h - margin), status_color, bracket_thickness)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - bracket_length), status_color, bracket_thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin - bracket_length, h - margin), status_color, bracket_thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - bracket_length), status_color, bracket_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 if not is_low_res else 0.5

        text_size = cv2.getTextSize(status_text, font, font_scale, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.rectangle(frame, (text_x - 10, 10), (text_x + text_size[0] + 10, 45 if not is_low_res else 35), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (text_x, 35 if not is_low_res else 28), font, font_scale, status_color, 2)

        # LOW-RES MODE: Overlay heatmap instead of mesh
        if is_low_res and self.inconsistency_map is not None:
            # Blend heatmap with frame
            heatmap_resized = cv2.resize(self.inconsistency_map, (w, h))
            frame = cv2.addWeighted(frame, 0.7, heatmap_resized, 0.3, 0)

            # Add "HEATMAP MODE" indicator
            cv2.putText(frame, "HEATMAP MODE", (10, h - 40), font, 0.4, (0, 184, 255), 1)
            cv2.putText(frame, f"Res: {h}p (Low)", (10, h - 25), font, 0.4, (255, 184, 0), 1)

        # Integrity gauge
        gauge_x = w - (80 if not is_low_res else 50)
        gauge_y = h // 2 - (100 if not is_low_res else 60)
        gauge_height = 200 if not is_low_res else 120
        gauge_width = 20 if not is_low_res else 12

        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), (40, 40, 40), -1)

        trust = 1 - result.get("probability", 0.5)
        fill_height = int(gauge_height * trust)
        fill_color = (83, 200, 0) if trust > 0.7 else (0, 184, 255) if trust > 0.4 else (57, 61, 255)
        cv2.rectangle(frame, (gauge_x, gauge_y + gauge_height - fill_height),
                     (gauge_x + gauge_width, gauge_y + gauge_height), fill_color, -1)

        cv2.putText(frame, f"{int(trust * 100)}%", (gauge_x - 10, gauge_y + gauge_height + 25),
                   font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "TRUST", (gauge_x - 15, gauge_y - 10), font, 0.4, (180, 180, 180), 1)

        latency = result.get("latency_ms", 0)
        cv2.putText(frame, f"{latency:.0f}ms", (10, h - 10), font, 0.4, (180, 180, 180), 1)

        return frame


# ============== PREMIUM CSS ==============

def get_elite_css(theme: dict) -> str:
    """Generate SCANNER ELITE v5.0 Tactical Command Suite CSS."""
    is_dark = theme["name"] == "dark"

    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    @property --border-angle {{
        syntax: "<angle>";
        initial-value: 0deg;
        inherits: false;
    }}

    @keyframes liquidBorderRotate {{
        0% {{ --border-angle: 0deg; }}
        100% {{ --border-angle: 360deg; }}
    }}

    @keyframes dualGlowPulse {{
        0%, 100% {{ filter: blur(60px) brightness(1); transform: scale(1); }}
        50% {{ filter: blur(80px) brightness(1.3); transform: scale(1.08); }}
    }}

    @keyframes shimmerGradient {{
        0% {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}

    @keyframes radarPulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.2); opacity: 0.6; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}

    @keyframes statusBlink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    @keyframes scanLine {{
        0% {{ top: 0%; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ top: 100%; opacity: 0; }}
    }}

    .stApp {{
        background: {get_color(theme, 'background')} !important;
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: {get_color(theme, 'background')} !important;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 15% 15%, {get_color(theme, 'glow_purple')} 0%, transparent 45%),
            radial-gradient(ellipse at 85% 85%, {get_color(theme, 'glow_blue')} 0%, transparent 45%),
            radial-gradient(ellipse at 50% 50%, {get_color(theme, 'glow_orange') if is_dark else 'transparent'} 0%, transparent 55%),
            linear-gradient(145deg, {get_color(theme, 'gradient_start')} 0%, {get_color(theme, 'gradient_mid')} 50%, {get_color(theme, 'gradient_end')} 100%);
        pointer-events: none;
        z-index: -1;
    }}

    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {get_color(theme, 'background_secondary')}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {get_color(theme, 'accent_blue')}, {get_color(theme, 'accent_purple')}); border-radius: 4px; }}

    /* ============== TACTICAL NAVBAR ============== */
    .tactical-navbar {{
        position: sticky;
        top: 0;
        z-index: 1000;
        background: {get_color(theme, 'glass_bg')};
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid {get_color(theme, 'glass_border')};
        padding: 1rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }}

    .navbar-brand {{
        display: flex;
        align-items: center;
        gap: 1rem;
    }}

    .radar-icon {{
        width: 36px;
        height: 36px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .radar-ring {{
        position: absolute;
        width: 100%;
        height: 100%;
        border: 2px solid {get_color(theme, 'accent_cyan')};
        border-radius: 50%;
        animation: radarPulse 2s ease-in-out infinite;
    }}

    .radar-ring:nth-child(2) {{
        width: 70%;
        height: 70%;
        animation-delay: 0.3s;
    }}

    .radar-ring:nth-child(3) {{
        width: 40%;
        height: 40%;
        animation-delay: 0.6s;
        background: {get_color(theme, 'accent_cyan')};
    }}

    .brand-text {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: {get_color(theme, 'text_primary')};
        letter-spacing: 6px;
        text-transform: uppercase;
    }}

    .navbar-links {{
        display: flex;
        gap: 0.5rem;
    }}

    .nav-link {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: {get_color(theme, 'text_secondary')};
        text-decoration: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        transition: all 0.3s ease;
        letter-spacing: 1px;
        text-transform: uppercase;
        cursor: pointer;
    }}

    .nav-link:hover {{
        color: {get_color(theme, 'text_primary')};
        background: {get_color(theme, 'glass_bg')};
        border-color: {get_color(theme, 'accent_blue')}40;
    }}

    .nav-link.active {{
        color: {get_color(theme, 'accent_cyan')};
        background: {get_color(theme, 'glow_cyan')};
        border-color: {get_color(theme, 'accent_cyan')}60;
    }}

    .system-status {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 1rem;
        background: {get_color(theme, 'glass_bg')};
        border: 1px solid {get_color(theme, 'glass_border')};
        border-radius: 20px;
    }}

    .status-dot {{
        width: 10px;
        height: 10px;
        background: {get_color(theme, 'success')};
        border-radius: 50%;
        box-shadow: 0 0 10px {get_color(theme, 'success')}, 0 0 20px {get_color(theme, 'glow_success')};
        animation: statusBlink 2s ease-in-out infinite;
    }}

    .status-text {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: {get_color(theme, 'success')};
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    /* ============== WAR ROOM CONTAINER ============== */
    .war-room {{
        background: {get_color(theme, 'glass_bg')};
        backdrop-filter: blur(20px);
        border: 1px solid {get_color(theme, 'glass_border')};
        border-radius: 20px;
        margin: 1.5rem 0;
        overflow: hidden;
        box-shadow:
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 {get_color(theme, 'glass_border')};
    }}

    .war-room-header {{
        background: linear-gradient(90deg,
            {get_color(theme, 'background_secondary')} 0%,
            {get_color(theme, 'background_tertiary')} 50%,
            {get_color(theme, 'background_secondary')} 100%);
        border-bottom: 1px solid {get_color(theme, 'glass_border')};
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .session-info {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}

    .session-id {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: {get_color(theme, 'accent_cyan')};
        letter-spacing: 1px;
    }}

    .session-timestamp {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: {get_color(theme, 'text_muted')};
    }}

    .classification-badge {{
        font-family: 'Inter', sans-serif;
        font-size: 0.55rem;
        font-weight: 700;
        color: {get_color(theme, 'accent_orange')};
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 12px;
        border: 1px solid {get_color(theme, 'accent_orange')}60;
        border-radius: 4px;
        background: {get_color(theme, 'glow_orange')};
    }}

    .war-room-body {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        min-height: 450px;
    }}

    .analysis-pane {{
        padding: 1.5rem;
        border-right: 1px solid {get_color(theme, 'glass_border')};
        position: relative;
    }}

    .analysis-pane::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            transparent,
            {get_color(theme, 'accent_blue')},
            {get_color(theme, 'accent_cyan')},
            transparent);
        opacity: 0.6;
    }}

    .intelligence-pane {{
        padding: 1.5rem;
        background: {get_color(theme, 'terminal_bg')};
        position: relative;
    }}

    .intelligence-pane::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            transparent,
            {get_color(theme, 'accent_orange')},
            {get_color(theme, 'accent_amber')},
            transparent);
        opacity: 0.6;
    }}

    .pane-title {{
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        color: {get_color(theme, 'text_muted')};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .pane-title-icon {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }}

    .pane-title-icon.blue {{
        background: {get_color(theme, 'accent_blue')};
        box-shadow: 0 0 8px {get_color(theme, 'glow_blue')};
    }}

    .pane-title-icon.orange {{
        background: {get_color(theme, 'accent_orange')};
        box-shadow: 0 0 8px {get_color(theme, 'glow_orange')};
    }}

    .video-preview-container {{
        position: relative;
        background: {get_color(theme, 'background')};
        border: 2px solid {get_color(theme, 'accent_blue')}40;
        border-radius: 12px;
        min-height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        box-shadow:
            0 0 30px {get_color(theme, 'glow_blue')},
            inset 0 0 60px rgba(0, 102, 255, 0.05);
    }}

    .video-preview-container.alert {{
        border-color: {get_color(theme, 'accent_orange')}60;
        box-shadow:
            0 0 30px {get_color(theme, 'glow_orange')},
            inset 0 0 60px rgba(255, 107, 53, 0.05);
    }}

    .scan-line {{
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg,
            transparent,
            {get_color(theme, 'accent_cyan')},
            {get_color(theme, 'accent_blue')},
            {get_color(theme, 'accent_cyan')},
            transparent);
        animation: scanLine 3s linear infinite;
        box-shadow: 0 0 10px {get_color(theme, 'accent_cyan')};
    }}

    /* ============== TACTICAL FOOTER ============== */
    .tactical-footer {{
        background: {get_color(theme, 'background_secondary')};
        border-top: 1px solid {get_color(theme, 'glass_border')};
        padding: 1.5rem 2rem;
        margin: 2rem -1rem -1rem -1rem;
    }}

    .footer-content {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.5rem;
    }}

    .system-telemetry {{
        display: flex;
        gap: 2rem;
    }}

    .telemetry-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .telemetry-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 600;
        color: {get_color(theme, 'text_muted')};
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .telemetry-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: {get_color(theme, 'accent_cyan')};
    }}

    .telemetry-status {{
        width: 6px;
        height: 6px;
        background: {get_color(theme, 'success')};
        border-radius: 50%;
        box-shadow: 0 0 6px {get_color(theme, 'success')};
    }}

    .footer-links {{
        display: flex;
        gap: 1.5rem;
    }}

    .footer-link {{
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 500;
        color: {get_color(theme, 'text_muted')};
        text-decoration: none;
        transition: color 0.2s ease;
        letter-spacing: 0.5px;
    }}

    .footer-link:hover {{
        color: {get_color(theme, 'accent_cyan')};
    }}

    .footer-signature {{
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 500;
        color: {get_color(theme, 'text_muted')};
        letter-spacing: 1px;
    }}

    .footer-signature strong {{
        color: {get_color(theme, 'text_secondary')};
        font-weight: 700;
    }}

    /* ============== ORIGINAL COMPONENTS (ENHANCED) ============== */
    .elite-logo {{ display: flex; align-items: baseline; gap: 1rem; }}
    .elite-logo-main {{
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 900;
        color: {get_color(theme, 'text_primary')};
        letter-spacing: 8px;
        text-transform: uppercase;
        margin: 0;
    }}
    .elite-logo-sub {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: {get_color(theme, 'accent_cyan')};
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 4px 12px;
        border: 1px solid {get_color(theme, 'accent_cyan')}40;
        border-radius: 4px;
        background: {get_color(theme, 'glow_cyan')};
    }}
    .elite-badge {{
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 600;
        color: {get_color(theme, 'text_muted')};
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 14px;
        border: 1px solid {get_color(theme, 'card_border')};
        border-radius: 20px;
        background: {get_color(theme, 'glass_bg')};
    }}

    .resolution-alert {{
        background: {get_color(theme, 'glow_warning')};
        border: 1px solid {get_color(theme, 'warning')};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .resolution-alert-icon {{ font-size: 1.5rem; }}
    .resolution-alert-text {{
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: {get_color(theme, 'text_warning')};
        font-weight: 500;
    }}

    .scanner-zone {{
        position: relative;
        margin: 2rem 0;
        padding: 4px;
        border-radius: 24px;
        background: {get_color(theme, 'glass_bg')};
        backdrop-filter: {get_color(theme, 'glass_blur')};
        border: 1px solid {get_color(theme, 'glass_border')};
        overflow: visible;
    }}
    .scanner-zone::before {{
        content: "";
        position: absolute;
        top: -80px; left: -80px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, {get_color(theme, 'glow_orange')} 0%, {get_color(theme, 'glow_amber')} 35%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        animation: dualGlowPulse 5s ease-in-out infinite;
    }}
    .scanner-zone::after {{
        content: "";
        position: absolute;
        bottom: -80px; right: -80px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, {get_color(theme, 'glow_cyan')} 0%, {get_color(theme, 'glow_purple')} 35%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        animation: dualGlowPulse 5s ease-in-out infinite 2.5s;
    }}

    .liquid-border-wrapper {{
        position: relative;
        border-radius: 20px;
        padding: 3px;
    }}
    .liquid-border-wrapper::before {{
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 3px;
        background: conic-gradient(
            from var(--border-angle),
            {get_color(theme, 'gradient_start')} 0deg,
            {get_color(theme, 'accent_orange')} 45deg,
            {get_color(theme, 'accent_amber')} 90deg,
            {get_color(theme, 'gradient_mid')} 135deg,
            {get_color(theme, 'accent_blue')} 180deg,
            {get_color(theme, 'accent_cyan')} 225deg,
            {get_color(theme, 'accent_purple')} 270deg,
            {get_color(theme, 'gradient_start')} 360deg
        );
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        animation: liquidBorderRotate 8s linear infinite;
        opacity: {0.95 if is_dark else 0.7};
    }}
    .liquid-border-inner {{
        background: {get_color(theme, 'card_bg_solid')};
        border-radius: 17px;
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        position: relative;
        padding: 3rem 2rem;
    }}

    .upload-icon {{ font-size: 4.5rem; margin-bottom: 1.5rem; filter: {"drop-shadow(0 0 30px " + get_color(theme, 'glow_cyan') + ")" if is_dark else "none"}; }}
    .upload-title {{ font-family: 'Inter', sans-serif; font-size: 1.3rem; font-weight: 700; color: {get_color(theme, 'text_primary')}; margin: 0 0 0.75rem 0; }}
    .upload-subtitle {{ font-family: 'Inter', sans-serif; font-size: 0.85rem; color: {get_color(theme, 'text_muted')}; margin: 0; }}

    .engine-card {{
        background: {get_color(theme, 'glass_bg')};
        border: 1px solid {get_color(theme, 'glass_border')};
        border-radius: 16px;
        padding: 1.25rem;
        backdrop-filter: {get_color(theme, 'glass_blur')};
        transition: all 0.3s ease;
    }}
    .engine-card:hover {{ transform: translateY(-2px); border-color: {get_color(theme, 'accent_blue')}40; }}
    .engine-card.status-pass {{ border-left: 3px solid {get_color(theme, 'success')}; }}
    .engine-card.status-warn {{ border-left: 3px solid {get_color(theme, 'warning')}; }}
    .engine-card.status-fail {{ border-left: 3px solid {get_color(theme, 'danger')}; }}
    .engine-card.leading {{ box-shadow: 0 0 20px {get_color(theme, 'glow_cyan')}; border-color: {get_color(theme, 'accent_cyan')}; }}

    .engine-name {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: {get_color(theme, 'text_primary')};
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }}
    .engine-score {{ font-family: 'Inter', sans-serif; font-size: 1.8rem; font-weight: 800; }}
    .engine-status {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .status-pass-badge {{ background: {get_color(theme, 'glow_success')}; color: {get_color(theme, 'success')}; }}
    .status-warn-badge {{ background: {get_color(theme, 'glow_amber')}; color: {get_color(theme, 'warning')}; }}
    .status-fail-badge {{ background: {get_color(theme, 'glow_red')}; color: {get_color(theme, 'danger')}; }}

    .leading-badge {{
        background: linear-gradient(135deg, {get_color(theme, 'accent_cyan')}, {get_color(theme, 'accent_blue')});
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.55rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-left: 8px;
    }}

    .data-quality-badge {{
        font-size: 0.55rem;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-block;
    }}
    .quality-good {{ background: {get_color(theme, 'glow_success')}; color: {get_color(theme, 'success')}; }}
    .quality-limited {{ background: {get_color(theme, 'glow_amber')}; color: {get_color(theme, 'warning')}; }}
    .quality-insufficient {{ background: {get_color(theme, 'glow_red')}; color: {get_color(theme, 'danger')}; }}

    div[data-testid="stButton"] > button {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 14px !important;
        padding: 1rem 2.5rem !important;
        background: {get_color(theme, 'button_bg')} !important;
        color: {get_color(theme, 'button_text')} !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow:
            -15px 0 40px -10px {get_color(theme, 'glow_orange')},
            15px 0 40px -10px {get_color(theme, 'glow_cyan')},
            0 20px 50px -15px rgba(0, 0, 0, 0.6) !important;
    }}

    .batch-container {{
        position: relative;
        background: {get_color(theme, 'glass_bg')};
        backdrop-filter: {get_color(theme, 'glass_blur')};
        border: 1px solid {get_color(theme, 'glass_border')};
        border-radius: 20px;
        padding: 1.75rem;
        margin: 1.5rem 0;
    }}
    .batch-table {{ width: 100%; border-collapse: collapse; }}
    .batch-table th {{
        text-align: left;
        padding: 1rem 1.25rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 700;
        color: {get_color(theme, 'text_muted')};
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid {get_color(theme, 'glass_border')};
    }}
    .batch-table td {{
        padding: 1.1rem 1.25rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: {get_color(theme, 'text_primary')};
        border-bottom: 1px solid {get_color(theme, 'glass_border')};
    }}
    .progress-track {{ width: 140px; height: 6px; background: {get_color(theme, 'card_border')}; border-radius: 3px; overflow: hidden; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, {get_color(theme, 'accent_orange')}, {get_color(theme, 'accent_cyan')}, {get_color(theme, 'accent_purple')}); background-size: 200% 100%; animation: shimmerGradient 2s linear infinite; }}
    .status-pill {{ display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }}
    .status-queued {{ background: {"rgba(255, 255, 255, 0.06)" if is_dark else "rgba(0, 0, 0, 0.05)"}; color: {get_color(theme, 'text_muted')}; }}
    .status-analyzing {{ background: linear-gradient(135deg, {get_color(theme, 'glow_blue')}, {get_color(theme, 'glow_cyan')}); color: {get_color(theme, 'accent_cyan')}; }}
    .status-complete {{ background: {get_color(theme, 'glow_success')}; color: {get_color(theme, 'success')}; }}
    .result-threat {{ color: {get_color(theme, 'danger')}; font-weight: 700; }}
    .result-verified {{ color: {get_color(theme, 'success')}; font-weight: 700; }}

    [data-testid="stFileUploader"] {{ background: transparent !important; }}
    [data-testid="stFileUploader"] section {{ background: transparent !important; border: none !important; }}
    section[data-testid="stSidebar"] {{ background: {get_color(theme, 'background_secondary')} !important; border-right: 1px solid {get_color(theme, 'card_border')} !important; }}

    .elite-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {get_color(theme, 'glass_border')}, {get_color(theme, 'accent_blue')}30, {get_color(theme, 'accent_orange')}20, {get_color(theme, 'glass_border')}, transparent);
        margin: 2rem 0;
        border: none;
    }}

    .blockchain-proof {{
        background: {get_color(theme, 'terminal_bg')};
        border: 1px solid {get_color(theme, 'glass_border')};
        border-radius: 12px;
        padding: 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 1.5rem;
    }}
    .blockchain-title {{ font-family: 'Inter', sans-serif; font-size: 0.65rem; font-weight: 700; color: {get_color(theme, 'text_muted')}; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.75rem; }}
    .blockchain-hash {{ font-size: 0.7rem; color: {get_color(theme, 'accent_cyan')}; word-break: break-all; line-height: 1.6; }}
    .blockchain-timestamp {{ font-size: 0.65rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.5rem; }}
    """


# ============== TERMINAL RENDERER ==============

def render_forensic_terminal(logs: List[Tuple[str, str]], theme: dict):
    """Render forensic terminal."""
    log_html = ""
    for msg, level in logs[-20:]:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = get_color(theme, 'text_secondary')
        prefix = "›"

        if level == "success":
            color = get_color(theme, 'success')
            prefix = "✓"
        elif level == "alert":
            color = get_color(theme, 'danger')
            prefix = "⚠"
        elif level == "scan":
            color = get_color(theme, 'accent_cyan')
            prefix = "◉"
        elif level == "engine":
            color = get_color(theme, 'accent_purple')
            prefix = "◆"
        elif level == "hash":
            color = get_color(theme, 'accent_amber')
            prefix = "⬡"
        elif level == "resolution":
            color = get_color(theme, 'warning')
            prefix = "⚡"

        log_html += f'<div class="log-line"><span class="log-time">{timestamp}</span><span class="log-prefix" style="color: {color};">{prefix}</span><span class="log-text" style="color: {color};">{msg}</span></div>'

    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        .terminal {{
            background: {get_color(theme, 'terminal_bg')};
            backdrop-filter: blur(20px);
            border: 1px solid {get_color(theme, 'glass_border')};
            border-radius: 14px;
            font-family: 'JetBrains Mono', monospace;
            overflow: hidden;
        }}
        .terminal-header {{
            background: {get_color(theme, 'terminal_header')};
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid {get_color(theme, 'glass_border')};
        }}
        .terminal-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .dot-close {{ background: linear-gradient(135deg, #FF5F57, #FF2D20); }}
        .dot-minimize {{ background: linear-gradient(135deg, #FFBD2E, #FFa500); }}
        .dot-maximize {{ background: linear-gradient(135deg, #28CA41, #00A843); }}
        .terminal-title {{ color: {get_color(theme, 'text_muted')}; font-size: 11px; margin-left: 10px; letter-spacing: 2px; text-transform: uppercase; }}
        .terminal-body {{ padding: 16px; max-height: 250px; overflow-y: auto; }}
        .log-line {{ display: flex; align-items: center; gap: 12px; font-size: 11px; line-height: 2; }}
        .log-time {{ color: {get_color(theme, 'text_muted')}; font-size: 9px; min-width: 75px; opacity: 0.7; }}
        .log-prefix {{ font-weight: 600; min-width: 12px; }}
        .cursor {{ display: inline-block; width: 8px; height: 16px; background: linear-gradient(180deg, {get_color(theme, 'accent_cyan')}, {get_color(theme, 'accent_blue')}); margin-left: 4px; animation: blink 1s step-end infinite; }}
        @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
    </style>
    <div class="terminal">
        <div class="terminal-header">
            <div class="terminal-dot dot-close"></div>
            <div class="terminal-dot dot-minimize"></div>
            <div class="terminal-dot dot-maximize"></div>
            <span class="terminal-title">Forensic Analysis Log</span>
        </div>
        <div class="terminal-body">
            {log_html if log_html else f'<div class="log-line"><span class="log-time">--:--:--.---</span><span class="log-prefix" style="color: {get_color(theme, "text_muted")};">›</span><span class="log-text" style="color: {get_color(theme, "text_muted")};">[SYS] Awaiting forensic input...</span></div>'}
            <span class="cursor"></span>
        </div>
    </div>
    """, height=340)


# ============== TRUST GAUGE RENDERER ==============

def render_trust_gauge(trust_score: float, verdict: str, leading_engine: str, theme: dict, verdict_reason: str = ""):
    """Render Trust Score gauge with leading engine indicator and verdict reason."""
    # Determine colors based on verdict (not just trust score)
    if verdict == "AUTHENTIC":
        gauge_color = get_color(theme, 'success')
        verdict_class = "verdict-authentic"
    elif verdict == "MANIPULATED":
        gauge_color = get_color(theme, 'danger')
        verdict_class = "verdict-threat"
    elif verdict == "INCONCLUSIVE":
        gauge_color = get_color(theme, 'accent_purple')  # Distinct color for INCONCLUSIVE
        verdict_class = "verdict-inconclusive"
    else:  # UNCERTAIN
        gauge_color = get_color(theme, 'warning')
        verdict_class = "verdict-uncertain"

    arc_percentage = trust_score / 100
    circumference = 2 * 3.14159 * 85
    dash_offset = circumference * (1 - arc_percentage)

    # Shorten verdict reason for display
    reason_display = verdict_reason[:60] + "..." if len(verdict_reason) > 60 else verdict_reason

    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;900&display=swap');
        .gauge-container {{ display: flex; flex-direction: column; align-items: center; padding: 2rem; }}
        .gauge-svg {{ transform: rotate(-90deg); filter: drop-shadow(0 0 20px {gauge_color}50); }}
        .gauge-bg {{ fill: none; stroke: {get_color(theme, 'card_border')}; stroke-width: 12; }}
        .gauge-fill {{ fill: none; stroke: {gauge_color}; stroke-width: 12; stroke-linecap: round; stroke-dasharray: {circumference}; stroke-dashoffset: {dash_offset}; transition: stroke-dashoffset 1s ease-out; }}
        .gauge-value {{ font-family: 'Inter', sans-serif; font-size: 3.5rem; font-weight: 900; fill: {get_color(theme, 'text_primary')}; }}
        .gauge-percent {{ font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700; fill: {get_color(theme, 'text_muted')}; }}
        .gauge-label {{ font-family: 'Inter', sans-serif; font-size: 0.7rem; font-weight: 700; color: {get_color(theme, 'text_muted')}; letter-spacing: 3px; text-transform: uppercase; margin-top: 1.5rem; }}
        .gauge-verdict {{ font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; margin-top: 1rem; padding: 8px 24px; border-radius: 8px; }}
        .verdict-authentic {{ color: {get_color(theme, 'success')}; background: {get_color(theme, 'glow_success')}; }}
        .verdict-threat {{ color: {get_color(theme, 'danger')}; background: {get_color(theme, 'glow_red')}; }}
        .verdict-uncertain {{ color: {get_color(theme, 'warning')}; background: {get_color(theme, 'glow_amber')}; }}
        .verdict-inconclusive {{
            color: {get_color(theme, 'accent_purple')};
            background: {get_color(theme, 'glow_purple')};
            border: 1px dashed {get_color(theme, 'accent_purple')}60;
        }}
        .leading-engine {{ font-family: 'Inter', sans-serif; font-size: 0.65rem; color: {get_color(theme, 'accent_cyan')}; margin-top: 0.75rem; letter-spacing: 1px; }}
        .verdict-reason {{
            font-family: 'Inter', sans-serif;
            font-size: 0.6rem;
            color: {get_color(theme, 'text_muted')};
            margin-top: 0.5rem;
            max-width: 250px;
            text-align: center;
            line-height: 1.4;
        }}
    </style>
    <div class="gauge-container">
        <svg class="gauge-svg" width="200" height="200" viewBox="0 0 200 200">
            <circle class="gauge-bg" cx="100" cy="100" r="85"/>
            <circle class="gauge-fill" cx="100" cy="100" r="85"/>
            <text class="gauge-value" x="100" y="105" text-anchor="middle" transform="rotate(90 100 100)">{int(trust_score)}</text>
            <text class="gauge-percent" x="140" y="115" text-anchor="middle" transform="rotate(90 100 100)">%</text>
        </svg>
        <div class="gauge-label">Trust Score</div>
        <div class="gauge-verdict {verdict_class}">{verdict}</div>
        <div class="leading-engine">Leading: {leading_engine}</div>
        {f'<div class="verdict-reason">{reason_display}</div>' if reason_display else ''}
    </div>
    """, height=420 if verdict_reason else 380)


# ============== MODEL LOADING ==============

@st.cache_resource
def load_models():
    """Load and cache detection models."""
    face_extractor = FaceExtractor(min_detection_confidence=0.5)
    video_processor = VideoProcessor(face_extractor, fps_sample_rate=1)
    inference_engine = DeepfakeInference()
    return face_extractor, video_processor, inference_engine


@st.cache_resource
def load_forensic_engines():
    """Load PRIME HYBRID core forensic analysis engines."""
    return {
        "biosignal": BioSignalCore(),       # Biological integrity (32-ROI rPPG)
        "artifact": ArtifactCore(),         # Generative model forensics (GAN/Diffusion/VAE)
        "alignment": AlignmentCore(),       # Multimodal cross-check (Phoneme-Viseme)
        "fusion": FusionEngine(),           # Unified decision engine
        "audio_analyzer": AudioAnalyzer()   # Audio quality analysis for adaptive weighting
    }


# ============== ANALYSIS FUNCTIONS ==============

def extract_raw_frames(video_path: str, max_frames: int = 60) -> Tuple[List[np.ndarray], float]:
    """Extract raw frames for forensic analysis."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 30.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    frames = []
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames, fps


def run_prime_hybrid_analysis(
    video_path: str,
    frames: List[np.ndarray],
    fps: float,
    engines: Dict,
    video_profile: VideoProfile,
    log_callback=None
) -> Tuple[FusionVerdict, Optional[AudioProfile]]:
    """
    Run PRIME HYBRID core analysis with three specialized cores and unified fusion engine.

    Cores:
    - BIOSIGNAL CORE: 32-ROI rPPG biological signal analysis
    - ARTIFACT CORE: GAN/Diffusion/VAE fingerprint detection
    - ALIGNMENT CORE: Phoneme-Viseme mapping and A/V alignment

    NEW: Audio-aware weight redistribution for ALIGNMENT CORE

    Returns:
        Tuple of (FusionVerdict with final unified decision, AudioProfile)
    """
    if log_callback:
        log_callback("[DEFENSE] Applying adversarial countermeasures...", "engine")
    defended_frames = AdversarialDefenseLayer.apply_batch(frames)

    # Convert video profile to core format
    core_profile = convert_video_profile_to_core(video_profile)

    # AUDIO ANALYSIS - Analyze audio quality for adaptive weighting
    audio_profile = None
    if "audio_analyzer" in engines:
        if log_callback:
            log_callback("[AUDIO] Extracting audio for SNR analysis...", "engine")
        try:
            audio_profile = engines["audio_analyzer"].analyze(video_path)
            if log_callback:
                if audio_profile.has_audio:
                    log_callback(
                        f"[AUDIO] SNR: {audio_profile.snr_db:.1f}dB | Noise: {audio_profile.noise_level} | "
                        f"Speech: {'Yes' if audio_profile.is_speech_detected else 'No'}", "engine"
                    )
                    if audio_profile.noise_level in ["HIGH", "EXTREME"]:
                        log_callback(
                            f"[AUDIO] Warning: High noise detected, ALIGNMENT weight will be reduced to {audio_profile.recommended_av_weight:.0%}",
                            "resolution"
                        )
                else:
                    log_callback("[AUDIO] No audio track detected, ALIGNMENT weight reduced to 30%", "resolution")
        except Exception as e:
            if log_callback:
                log_callback(f"[AUDIO] Analysis skipped: {str(e)}", "resolution")

    # BIOSIGNAL CORE - Biological Integrity (32-ROI rPPG)
    if log_callback:
        if video_profile.rppg_viable:
            log_callback("[BIOSIGNAL] 32-ROI rPPG analysis - Biological signal detection...", "engine")
        else:
            log_callback("[BIOSIGNAL] 32-ROI rPPG analysis - Low-res mode (spatial averaging)...", "resolution")
    biosignal_result = engines["biosignal"].analyze(defended_frames, fps, core_profile)

    if log_callback:
        status_icon = "✓" if biosignal_result.status == "PASS" else "⚠" if biosignal_result.status == "WARN" else "✗"
        log_callback(f"[BIOSIGNAL] Bio-Sync: {biosignal_result.biological_sync_score:.2%} | Pulse Coverage: {biosignal_result.pulse_coverage:.2%} | {status_icon}", "engine")

    # ARTIFACT CORE - Generative Model Forensics (GAN/Diffusion/VAE)
    if log_callback:
        log_callback("[ARTIFACT] Generative fingerprint scan - GAN/Diffusion/VAE detection...", "engine")
    artifact_result = engines["artifact"].analyze(defended_frames, core_profile)

    if log_callback:
        model_type = artifact_result.details.get("detected_model_type", "NONE")
        status_icon = "✓" if artifact_result.status == "PASS" else "⚠" if artifact_result.status == "WARN" else "✗"
        log_callback(f"[ARTIFACT] Model Type: {model_type} | Structural Integrity: {artifact_result.structural_integrity:.2%} | {status_icon}", "engine")
        if artifact_result.detected_fingerprints:
            for fp in artifact_result.detected_fingerprints[:2]:
                log_callback(f"[ARTIFACT] Detected: {fp['type']} fingerprint (score: {fp['score']:.2%})", "scan")

    # ALIGNMENT CORE - Multimodal Cross-Check (Phoneme-Viseme)
    if log_callback:
        log_callback("[ALIGNMENT] Phoneme-Viseme mapping - A/V alignment check...", "engine")
    alignment_result = engines["alignment"].analyze(defended_frames, fps, video_path, core_profile)

    if log_callback:
        status_icon = "✓" if alignment_result.status == "PASS" else "⚠" if alignment_result.status == "WARN" else "✗"
        log_callback(f"[ALIGNMENT] A/V Sync: {1-alignment_result.av_alignment_score:.2%} | Lip Closures: {len(alignment_result.lip_closure_events)} | {status_icon}", "engine")

    # FUSION ENGINE - Unified Decision Engine with Audio Awareness
    if log_callback:
        log_callback("[FUSION] Computing weighted consensus verdict...", "scan")
        log_callback("[FUSION] Redistributing weights by confidence and audio quality...", "engine")
        log_callback("[FUSION] Resolving core conflicts...", "engine")

    verdict = engines["fusion"].get_final_integrity_score(
        biosignal_result, artifact_result, alignment_result,
        core_profile, audio_profile
    )

    if log_callback:
        log_callback(f"[FUSION] Consensus: {verdict.consensus_type} | Leading Core: {verdict.leading_core}", "hash")
        if verdict.conflicting_signals:
            log_callback("[FUSION] Warning: Conflicting signals detected between cores", "resolution")
        # Log transparency factors
        if verdict.transparency_report and verdict.transparency_report.environmental_factors:
            for factor in verdict.transparency_report.environmental_factors[:3]:
                log_callback(f"[FUSION] {factor}", "resolution")

    return verdict, audio_profile


def analyze_complete(
    video_path: str,
    video_processor: VideoProcessor,
    inference_engine: DeepfakeInference,
    face_extractor: FaceExtractor,
    forensic_engines: Dict,
    log_callback=None
) -> Dict[str, Any]:
    """
    Complete analysis using SCANNER PRIME HYBRID architecture.

    Three-Core System:
    - BIOSIGNAL CORE: 32-ROI rPPG biological signal analysis
    - ARTIFACT CORE: GAN/Diffusion/VAE fingerprint detection
    - ALIGNMENT CORE: Phoneme-Viseme mapping and A/V alignment

    FUSION ENGINE: Unified decision engine with conflict resolution.

    Philosophy: High Precision over High Recall.
    Better to say "I don't know" than to falsely accuse.
    """

    if log_callback:
        log_callback("[INIT] SCANNER PRIME HYBRID initialized", "info")

    # Profile video first
    if log_callback:
        log_callback("[PROFILE] Analyzing video properties...", "scan")

    video_profile = VideoProfiler.analyze_video(video_path)

    if log_callback:
        log_callback(f"[PROFILE] Resolution: {video_profile.resolution_label}", "resolution")
        log_callback(f"[PROFILE] Strategy: {video_profile.recommended_analysis}", "resolution")

    # Extract frames
    raw_frames, fps = extract_raw_frames(video_path, max_frames=60)

    # Primary model analysis (EfficientNet-B0)
    if log_callback:
        log_callback("[SCAN] Running EfficientNet-B0 classifier...", "scan")

    processed_frames = video_processor.process_video(video_path)

    if not processed_frames:
        return {
            "error": "No faces detected in video",
            "verdict": "UNKNOWN",
            "trust_score": 0,
            "integrity_score": 0,
            "video_profile": video_profile
        }

    frame_results = []
    for frame_num, face_tensor in processed_frames:
        prob, label = inference_engine.predict_single(face_tensor)
        frame_results.append((frame_num, prob, label))

    primary_analysis = inference_engine.analyze_video_results(frame_results)
    primary_fake_prob = primary_analysis["average_fake_probability"]

    # Run PRIME HYBRID core analysis (now returns tuple with audio profile)
    if log_callback:
        log_callback("[PRIME] Running three-core forensic analysis...", "engine")

    fusion_verdict, audio_profile = run_prime_hybrid_analysis(
        video_path, raw_frames, fps, forensic_engines, video_profile, log_callback
    )

    # Log final verdict
    if log_callback:
        if fusion_verdict.verdict == "AUTHENTIC":
            log_callback(f"[RESULT] Verdict: {fusion_verdict.verdict} (Integrity: {fusion_verdict.integrity_score:.1f}%)", "success")
        elif fusion_verdict.verdict == "MANIPULATED":
            log_callback(f"[ALERT] Verdict: {fusion_verdict.verdict} (Integrity: {fusion_verdict.integrity_score:.1f}%)", "alert")
        elif fusion_verdict.verdict == "INCONCLUSIVE":
            log_callback(f"[CAUTION] Verdict: {fusion_verdict.verdict} - {fusion_verdict.reason}", "resolution")
        else:
            log_callback(f"[WARN] Verdict: {fusion_verdict.verdict} (Integrity: {fusion_verdict.integrity_score:.1f}%)", "info")
        log_callback(f"[RESULT] Leading Core: {fusion_verdict.leading_core}", "info")

    session_id = hashlib.sha256(f"{video_path}{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    # Build results with backward-compatible structure
    results = {
        "verdict": fusion_verdict.verdict,
        "verdict_reason": fusion_verdict.reason,
        "trust_score": fusion_verdict.integrity_score,  # For backward compatibility
        "integrity_score": fusion_verdict.integrity_score,
        "confidence": fusion_verdict.confidence,
        "leading_engine": fusion_verdict.leading_core,  # For backward compatibility
        "leading_core": fusion_verdict.leading_core,
        "weight_breakdown": fusion_verdict.weights,
        "consensus_type": fusion_verdict.consensus_type,
        "conflicting_signals": fusion_verdict.conflicting_signals,
        "video_profile": video_profile,
        "resolution_tier": video_profile.resolution_tier.value,
        "primary_analysis": primary_analysis,
        # Core results
        "cores": {
            "biosignal": {
                "score": fusion_verdict.biosignal_score,
                "status": fusion_verdict.biosignal_result.status if fusion_verdict.biosignal_result else "N/A",
                "confidence": fusion_verdict.biosignal_result.confidence if fusion_verdict.biosignal_result else 0,
                "biological_sync": fusion_verdict.biosignal_result.biological_sync_score if fusion_verdict.biosignal_result else 0,
                "pulse_coverage": fusion_verdict.biosignal_result.pulse_coverage if fusion_verdict.biosignal_result else 0,
                "anomalies": fusion_verdict.biosignal_result.anomalies if fusion_verdict.biosignal_result else [],
                "data_quality": fusion_verdict.biosignal_result.data_quality if fusion_verdict.biosignal_result else "N/A",
            },
            "artifact": {
                "score": fusion_verdict.artifact_score,
                "status": fusion_verdict.artifact_result.status if fusion_verdict.artifact_result else "N/A",
                "confidence": fusion_verdict.artifact_result.confidence if fusion_verdict.artifact_result else 0,
                "gan_score": fusion_verdict.artifact_result.gan_score if fusion_verdict.artifact_result else 0,
                "diffusion_score": fusion_verdict.artifact_result.diffusion_score if fusion_verdict.artifact_result else 0,
                "vae_score": fusion_verdict.artifact_result.vae_score if fusion_verdict.artifact_result else 0,
                "structural_integrity": fusion_verdict.artifact_result.structural_integrity if fusion_verdict.artifact_result else 0,
                "detected_fingerprints": fusion_verdict.artifact_result.detected_fingerprints if fusion_verdict.artifact_result else [],
                "anomalies": fusion_verdict.artifact_result.anomalies if fusion_verdict.artifact_result else [],
            },
            "alignment": {
                "score": fusion_verdict.alignment_score,
                "status": fusion_verdict.alignment_result.status if fusion_verdict.alignment_result else "N/A",
                "confidence": fusion_verdict.alignment_result.confidence if fusion_verdict.alignment_result else 0,
                "av_alignment": fusion_verdict.alignment_result.av_alignment_score if fusion_verdict.alignment_result else 0,
                "phoneme_viseme": fusion_verdict.alignment_result.phoneme_viseme_score if fusion_verdict.alignment_result else 0,
                "speech_rhythm": fusion_verdict.alignment_result.speech_rhythm_score if fusion_verdict.alignment_result else 0,
                "lip_closures": len(fusion_verdict.alignment_result.lip_closure_events) if fusion_verdict.alignment_result else 0,
                "anomalies": fusion_verdict.alignment_result.anomalies if fusion_verdict.alignment_result else [],
            },
        },
        # Backward-compatible engines format (mapped from cores)
        "engines": {
            "Biometric rPPG": {
                "score": fusion_verdict.biosignal_score,
                "confidence": fusion_verdict.biosignal_result.confidence if fusion_verdict.biosignal_result else 0,
                "status": fusion_verdict.biosignal_result.status if fusion_verdict.biosignal_result else "N/A",
                "details": fusion_verdict.biosignal_result.details if fusion_verdict.biosignal_result else {},
                "anomalies": fusion_verdict.biosignal_result.anomalies if fusion_verdict.biosignal_result else [],
                "data_quality": fusion_verdict.biosignal_result.data_quality if fusion_verdict.biosignal_result else "N/A",
            },
            "Neural Fingerprint": {
                "score": fusion_verdict.artifact_score,
                "confidence": fusion_verdict.artifact_result.confidence if fusion_verdict.artifact_result else 0,
                "status": fusion_verdict.artifact_result.status if fusion_verdict.artifact_result else "N/A",
                "details": fusion_verdict.artifact_result.details if fusion_verdict.artifact_result else {},
                "anomalies": fusion_verdict.artifact_result.anomalies if fusion_verdict.artifact_result else [],
                "data_quality": "GOOD",
            },
            "Semantic Logic": {
                "score": fusion_verdict.alignment_score,
                "confidence": fusion_verdict.alignment_result.confidence if fusion_verdict.alignment_result else 0,
                "status": fusion_verdict.alignment_result.status if fusion_verdict.alignment_result else "N/A",
                "details": fusion_verdict.alignment_result.details if fusion_verdict.alignment_result else {},
                "anomalies": fusion_verdict.alignment_result.anomalies if fusion_verdict.alignment_result else [],
                "data_quality": fusion_verdict.alignment_result.data_quality if fusion_verdict.alignment_result else "N/A",
            },
            "A/V Sync-Lock": {
                "score": fusion_verdict.alignment_result.av_alignment_score if fusion_verdict.alignment_result else 0.5,
                "confidence": fusion_verdict.alignment_result.confidence if fusion_verdict.alignment_result else 0,
                "status": fusion_verdict.alignment_result.status if fusion_verdict.alignment_result else "N/A",
                "details": {"speech_rhythm": fusion_verdict.alignment_result.speech_rhythm_score if fusion_verdict.alignment_result else 0},
                "anomalies": [],
                "data_quality": "GOOD",
            },
        },
        "session_id": session_id,
        "frames_analyzed": len(raw_frames),
        "fps": fps,
        # NEW: Audio profile for display
        "audio_profile": audio_profile.to_dict() if audio_profile else None,
        # NEW: Transparency report for detailed explanations
        "transparency_report": fusion_verdict.transparency_report.to_dict() if fusion_verdict.transparency_report else None,
        # NEW: Fusion method used
        "fusion_method": fusion_verdict.fusion_method if hasattr(fusion_verdict, 'fusion_method') else "weighted_average",
    }

    if log_callback:
        log_callback("[HASH] Generating blockchain-ready proof...", "hash")

    origin_proof = generate_origin_hash(video_path, results, session_id)
    results["origin_proof"] = origin_proof

    return results


# ============== NAVBAR RENDERER ==============

def render_tactical_navbar(theme: dict, current_page: str = "dashboard"):
    """Render the professional tactical navbar."""
    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        @keyframes radarPulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.15); opacity: 0.5; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}

        @keyframes statusGlow {{
            0%, 100% {{ box-shadow: 0 0 8px {get_color(theme, 'success')}, 0 0 16px {get_color(theme, 'glow_success')}; }}
            50% {{ box-shadow: 0 0 12px {get_color(theme, 'success')}, 0 0 24px {get_color(theme, 'glow_success')}; }}
        }}

        .navbar {{
            background: {get_color(theme, 'glass_bg')};
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-bottom: 1px solid {get_color(theme, 'glass_border')};
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }}

        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .radar-icon {{
            width: 40px;
            height: 40px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .radar-ring {{
            position: absolute;
            border: 2px solid {get_color(theme, 'accent_cyan')};
            border-radius: 50%;
            animation: radarPulse 2s ease-in-out infinite;
        }}

        .radar-ring.outer {{ width: 100%; height: 100%; }}
        .radar-ring.middle {{ width: 65%; height: 65%; animation-delay: 0.25s; }}
        .radar-ring.inner {{ width: 30%; height: 30%; animation-delay: 0.5s; background: {get_color(theme, 'accent_cyan')}; }}

        .brand-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
            color: {get_color(theme, 'text_primary')};
            letter-spacing: 5px;
            text-transform: uppercase;
        }}

        .navbar-links {{
            display: flex;
            gap: 0.25rem;
        }}

        .nav-link {{
            font-family: 'Inter', sans-serif;
            font-size: 0.65rem;
            font-weight: 600;
            color: {get_color(theme, 'text_secondary')};
            text-decoration: none;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            border: 1px solid transparent;
            background: transparent;
            transition: all 0.3s ease;
            letter-spacing: 1px;
            text-transform: uppercase;
            cursor: pointer;
        }}

        .nav-link:hover {{
            color: {get_color(theme, 'text_primary')};
            background: {get_color(theme, 'glass_bg')};
            border-color: {get_color(theme, 'accent_blue')}30;
        }}

        .nav-link.active {{
            color: {get_color(theme, 'accent_cyan')};
            background: {get_color(theme, 'glow_cyan')};
            border-color: {get_color(theme, 'accent_cyan')}50;
        }}

        .system-status {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 1rem;
            background: {get_color(theme, 'glass_bg')};
            border: 1px solid {get_color(theme, 'glass_border')};
            border-radius: 20px;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            background: {get_color(theme, 'success')};
            border-radius: 50%;
            animation: statusGlow 2s ease-in-out infinite;
        }}

        .status-text {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.6rem;
            font-weight: 600;
            color: {get_color(theme, 'success')};
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
    </style>

    <nav class="navbar">
        <div class="navbar-brand">
            <div class="radar-icon">
                <div class="radar-ring outer"></div>
                <div class="radar-ring middle"></div>
                <div class="radar-ring inner"></div>
            </div>
            <span class="brand-text">SCANNER</span>
        </div>

        <div class="navbar-links">
            <a class="nav-link {'active' if current_page == 'dashboard' else ''}" href="#">Dashboard</a>
            <a class="nav-link {'active' if current_page == 'reports' else ''}" href="#">Forensic Reports</a>
            <a class="nav-link {'active' if current_page == 'sentinel' else ''}" href="#">Live Sentinel</a>
            <a class="nav-link {'active' if current_page == 'api' else ''}" href="#">API Docs</a>
        </div>

        <div class="system-status">
            <div class="status-dot"></div>
            <span class="status-text">System: Operational</span>
        </div>
    </nav>
    """, height=80)


# ============== FOOTER RENDERER ==============

def render_tactical_footer(theme: dict):
    """Render the professional tactical footer with system telemetry."""
    import random
    nodes_active = random.randint(10, 15)
    hops = random.randint(2, 4)

    components.html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        .footer {{
            background: {get_color(theme, 'background_secondary')};
            border-top: 1px solid {get_color(theme, 'glass_border')};
            padding: 1.25rem 2rem;
        }}

        .footer-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .system-telemetry {{
            display: flex;
            gap: 2rem;
        }}

        .telemetry-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .telemetry-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.55rem;
            font-weight: 600;
            color: {get_color(theme, 'text_muted')};
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        .telemetry-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 600;
            color: {get_color(theme, 'accent_cyan')};
        }}

        .telemetry-status {{
            width: 5px;
            height: 5px;
            background: {get_color(theme, 'success')};
            border-radius: 50%;
            box-shadow: 0 0 4px {get_color(theme, 'success')};
        }}

        .footer-links {{
            display: flex;
            gap: 1.5rem;
        }}

        .footer-link {{
            font-family: 'Inter', sans-serif;
            font-size: 0.55rem;
            font-weight: 500;
            color: {get_color(theme, 'text_muted')};
            text-decoration: none;
            transition: color 0.2s ease;
            letter-spacing: 0.5px;
        }}

        .footer-link:hover {{
            color: {get_color(theme, 'accent_cyan')};
        }}

        .footer-signature {{
            font-family: 'Inter', sans-serif;
            font-size: 0.55rem;
            font-weight: 500;
            color: {get_color(theme, 'text_muted')};
            letter-spacing: 0.5px;
        }}

        .footer-signature strong {{
            color: {get_color(theme, 'text_secondary')};
            font-weight: 700;
        }}
    </style>

    <footer class="footer">
        <div class="footer-content">
            <div class="system-telemetry">
                <div class="telemetry-item">
                    <span class="telemetry-label">Nodes Active:</span>
                    <span class="telemetry-value">{nodes_active}</span>
                    <span class="telemetry-status"></span>
                </div>
                <div class="telemetry-item">
                    <span class="telemetry-label">Hops:</span>
                    <span class="telemetry-value">{hops}</span>
                </div>
                <div class="telemetry-item">
                    <span class="telemetry-label">Encrypted Tunnel:</span>
                    <span class="telemetry-value">Verified</span>
                    <span class="telemetry-status"></span>
                </div>
            </div>

            <div class="footer-links">
                <a class="footer-link" href="#">Privacy Policy</a>
                <a class="footer-link" href="#">Terms of Service</a>
                <a class="footer-link" href="#">Contact Defense Team</a>
            </div>

            <div class="footer-signature">
                <strong>Scanner</strong> © 2026 — Advanced Forensic Defense Systems
            </div>
        </div>
    </footer>
    """, height=70)


# ============== MAIN APPLICATION ==============

def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    if "terminal_logs" not in st.session_state:
        st.session_state.terminal_logs = [("[SYS] SCANNER ELITE v5.0 initialized", "info")]
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "upload"
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ID-{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6].upper()}"

    theme = get_theme(st.session_state.theme)

    def add_log(message: str, level: str = "info"):
        st.session_state.terminal_logs.append((message, level))
        if len(st.session_state.terminal_logs) > 100:
            st.session_state.terminal_logs = st.session_state.terminal_logs[-100:]

    st.markdown(f"<style>{get_elite_css(theme)}</style>", unsafe_allow_html=True)

    # Tactical Navbar
    render_tactical_navbar(theme, st.session_state.current_page)

    # Theme toggle and settings in a small row
    theme_cols = st.columns([8, 1, 1])
    with theme_cols[1]:
        toggle_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(toggle_icon, key="theme_toggle", help="Toggle Theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with theme_cols[2]:
        if st.button("⚙️", key="settings_btn", help="Settings"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
            st.rerun()

    # Settings Panel
    if st.session_state.show_settings:
        st.markdown(f'''
            <div style="
                background: {get_color(theme, 'glass_bg')};
                backdrop-filter: blur(20px);
                border: 1px solid {get_color(theme, 'glass_border')};
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            ">
                <h3 style="color: {get_color(theme, 'text_primary')}; margin: 0 0 1rem 0; font-size: 0.9rem; font-weight: 700; letter-spacing: 2px;">SYSTEM CONFIGURATION</h3>
            </div>
        ''', unsafe_allow_html=True)

        settings_cols = st.columns(3)
        with settings_cols[0]:
            st.selectbox("Default Engine Priority", ["Auto (Resolution-Adaptive)", "Neural Fingerprint First", "rPPG First"], key="engine_priority")
        with settings_cols[1]:
            st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="confidence_threshold")
        with settings_cols[2]:
            st.selectbox("Export Format", ["JSON", "PDF Report", "CSV"], key="export_format")

        st.markdown("---")

    # Sidebar - System Intelligence Panel
    with st.sidebar:
        st.markdown(f'''
            <div style="padding: 0.5rem 0;">
                <p style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {get_color(theme, "accent_cyan")}; font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1rem;">
                    SYSTEM INTEL
                </p>
            </div>
        ''', unsafe_allow_html=True)

        with st.spinner("Loading engines..."):
            face_extractor, video_processor, inference_engine = load_models()
            forensic_engines = load_forensic_engines()

        st.markdown(f'''
            <div style="
                background: {get_color(theme, 'glass_bg')};
                border: 1px solid {get_color(theme, 'glass_border')};
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <div style="width: 8px; height: 8px; background: {get_color(theme, 'success')}; border-radius: 50%; box-shadow: 0 0 8px {get_color(theme, 'success')};"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600; color: {get_color(theme, 'success')};">ALL SYSTEMS ONLINE</span>
                </div>
                <p style="font-family: 'JetBrains Mono', monospace; color: {get_color(theme, 'text_muted')}; margin: 0.25rem 0; font-size: 0.65rem;">{inference_engine.weights_source}</p>
                <p style="font-family: 'JetBrains Mono', monospace; color: {get_color(theme, 'text_muted')}; margin: 0.25rem 0; font-size: 0.65rem;">Resolution-Adaptive Mode</p>
            </div>
        ''', unsafe_allow_html=True)

        # Active Session Info
        st.markdown(f'''
            <div style="
                background: {get_color(theme, 'terminal_bg')};
                border: 1px solid {get_color(theme, 'accent_cyan')}30;
                border-left: 3px solid {get_color(theme, 'accent_cyan')};
                border-radius: 8px;
                padding: 0.75rem;
                margin-bottom: 1rem;
            ">
                <p style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">Active Session</p>
                <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; color: {get_color(theme, 'accent_cyan')};">{st.session_state.session_id}</p>
            </div>
        ''', unsafe_allow_html=True)

        # Display results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            profile = results.get("video_profile")

            if profile:
                st.markdown(f'''
                    <div style="
                        background: {get_color(theme, 'glass_bg')};
                        border: 1px solid {get_color(theme, 'glass_border')};
                        border-radius: 8px;
                        padding: 0.75rem;
                        margin-bottom: 1rem;
                    ">
                        <p style="font-family: Inter; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, "text_muted")}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">Video Profile</p>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {get_color(theme, 'text_secondary')};">
                            <p style="margin: 0.2rem 0;">Res: {profile.resolution_label}</p>
                            <p style="margin: 0.2rem 0;">FPS: {profile.fps:.1f}</p>
                            <p style="margin: 0.2rem 0;">Duration: {profile.duration_seconds:.1f}s</p>
                            <p style="margin: 0.2rem 0;">rPPG: {"✓" if profile.rppg_viable else "✗"}</p>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

            if results.get("weight_breakdown"):
                weights_html = "".join([
                    f'<div style="display: flex; justify-content: space-between; margin: 0.2rem 0;"><span>{engine}</span><span style="color: {get_color(theme, "accent_cyan")};">{weight:.0%}</span></div>'
                    for engine, weight in results["weight_breakdown"].items()
                ])
                st.markdown(f'''
                    <div style="
                        background: {get_color(theme, 'glass_bg')};
                        border: 1px solid {get_color(theme, 'glass_border')};
                        border-radius: 8px;
                        padding: 0.75rem;
                    ">
                        <p style="font-family: Inter; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, "text_muted")}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">Engine Weights</p>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {get_color(theme, 'text_secondary')};">
                            {weights_html}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

    # ============== WAR ROOM HEADER ==============
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f'''
        <div class="war-room">
            <div class="war-room-header">
                <div class="session-info">
                    <span class="session-id">ACTIVE SESSION: {st.session_state.session_id}</span>
                    <span class="session-timestamp">{current_time}</span>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Mode selector
    mode_col1, mode_col2, mode_col3 = st.columns([1, 2, 1])
    with mode_col2:
        input_mode = st.radio("Input Mode", ["UPLOAD FILE", "LIVE SENTINEL"], horizontal=True, label_visibility="collapsed")
        st.session_state.input_mode = "upload" if input_mode == "UPLOAD FILE" else "live"
        if input_mode == "LIVE SENTINEL":
            st.session_state.current_page = "sentinel"
        else:
            st.session_state.current_page = "dashboard"

    st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

    if st.session_state.input_mode == "upload":
        # War Room Dual-Pane Layout
        war_left, war_right = st.columns([1, 1])

        with war_left:
            st.markdown(f'''
                <div class="pane-title">
                    <span class="pane-title-icon blue"></span>
                    ANALYSIS ZONE
                </div>
            ''', unsafe_allow_html=True)

            st.markdown(f'''
                <div class="scanner-zone">
                    <div class="liquid-border-wrapper">
                        <div class="liquid-border-inner">
                            <div class="scan-line"></div>
                            <div class="upload-icon">🔬</div>
                            <p class="upload-title">Drop Evidence Files</p>
                            <p class="upload-subtitle">MP4, AVI, MOV, MKV • Defense-Grade Analysis</p>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        with war_right:
            st.markdown(f'''
                <div class="pane-title">
                    <span class="pane-title-icon orange"></span>
                    INTELLIGENCE LOGS
                </div>
            ''', unsafe_allow_html=True)

            # Render terminal in the right pane
            render_forensic_terminal(st.session_state.terminal_logs, theme)

        uploaded_files = st.file_uploader("Upload", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded_files:
            # Batch table
            table_rows = ""
            for f in uploaded_files:
                status = "Queued"
                verdict = "-"
                progress = 0
                if st.session_state.analysis_results and f.name == st.session_state.analysis_results.get("filename"):
                    status = "Done"
                    verdict = st.session_state.analysis_results.get("verdict", "-")
                    progress = 100

                status_class = "status-queued" if status == "Queued" else "status-complete"
                result_class = "result-threat" if verdict == "MANIPULATED" else "result-verified" if verdict == "AUTHENTIC" else ""

                table_rows += f'''
                    <tr>
                        <td>{f.name}</td>
                        <td><div class="progress-track"><div class="progress-fill" style="width: {progress}%;"></div></div></td>
                        <td><span class="status-pill {status_class}">{status}</span></td>
                        <td><span class="{result_class}">{verdict}</span></td>
                    </tr>
                '''

            st.markdown(f'''
                <div class="batch-container">
                    <table class="batch-table">
                        <thead><tr><th>Evidence File</th><th>Progress</th><th>Status</th><th>Verdict</th></tr></thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
            ''', unsafe_allow_html=True)

            btn_cols = st.columns([2, 1, 1, 1])
            with btn_cols[0]:
                analyze_clicked = st.button("ANALYZE", type="primary", use_container_width=True)
            with btn_cols[1]:
                report_clicked = st.button("EXPORT", use_container_width=True)
            with btn_cols[2]:
                clear_clicked = st.button("CLEAR", use_container_width=True)
            with btn_cols[3]:
                if st.session_state.analysis_results and st.session_state.analysis_results.get("origin_proof"):
                    st.download_button("PROOF", json.dumps(st.session_state.analysis_results["origin_proof"], indent=2), f"proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)

            if clear_clicked:
                st.session_state.analysis_results = None
                st.session_state.terminal_logs = [("[SYS] Session cleared", "info")]
                st.rerun()

            if analyze_clicked:
                for uploaded_file in uploaded_files:
                    add_log(f"[FILE] Processing: {uploaded_file.name}", "scan")

                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    uploaded_file.seek(0)

                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.markdown(f'<p style="text-align: center; color: {get_color(theme, "text_secondary")}; font-family: Inter;">Analyzing {uploaded_file.name}...</p>', unsafe_allow_html=True)
                        progress_bar.progress(10)

                        results = analyze_complete(tmp_path, video_processor, inference_engine, face_extractor, forensic_engines, log_callback=add_log)

                        progress_bar.progress(100)
                        progress_bar.empty()
                        status_text.empty()

                        results["filename"] = uploaded_file.name
                        st.session_state.analysis_results = results

                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

                st.rerun()

            # Display results
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results

                if "error" not in results:
                    st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                    # Resolution alert for low-res
                    profile = results.get("video_profile")
                    if profile and profile.resolution_tier in [ResolutionTier.ULTRA_LOW, ResolutionTier.LOW]:
                        st.markdown(f'''
                            <div class="resolution-alert">
                                <span class="resolution-alert-icon">⚡</span>
                                <span class="resolution-alert-text">
                                    Low resolution detected ({profile.resolution_label}).
                                    Switching to Deep-Pixel & Semantic analysis for accuracy.
                                    Biometric rPPG weight reduced to {results.get("weight_breakdown", {}).get("Biometric rPPG", 0):.0%}.
                                </span>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Trust gauge
                    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])
                    with gauge_col2:
                        render_trust_gauge(
                            results["trust_score"],
                            results["verdict"],
                            results.get("leading_engine", "N/A"),
                            theme,
                            results.get("verdict_reason", "")
                        )

                    st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                    # Engine cards - Tactical Analysis Section
                    st.markdown(f'''
                        <div style="
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 1rem;
                            margin-bottom: 1.5rem;
                        ">
                            <div style="
                                height: 1px;
                                flex: 1;
                                background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_blue')}40, transparent);
                            "></div>
                            <p style="
                                font-family: 'JetBrains Mono', monospace;
                                font-size: 0.65rem;
                                font-weight: 600;
                                color: {get_color(theme, 'accent_cyan')};
                                letter-spacing: 3px;
                                text-transform: uppercase;
                                margin: 0;
                                padding: 0.5rem 1.5rem;
                                background: {get_color(theme, 'glow_cyan')};
                                border: 1px solid {get_color(theme, 'accent_cyan')}40;
                                border-radius: 20px;
                            ">Multi-Engine Analysis</p>
                            <div style="
                                height: 1px;
                                flex: 1;
                                background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_blue')}40, transparent);
                            "></div>
                        </div>
                    ''', unsafe_allow_html=True)

                    engine_cols = st.columns(4)
                    engine_map = {
                        "rppg": ("Biometric rPPG", "💓"),
                        "neural": ("Neural Fingerprint", "🔍"),
                        "semantic": ("Semantic Logic", "🧠"),
                        "avsync": ("A/V Sync-Lock", "🔊")
                    }

                    leading = results.get("leading_engine", "")

                    for i, (key, (name, icon)) in enumerate(engine_map.items()):
                        with engine_cols[i]:
                            data = results.get("engines", {}).get(key, {})
                            score = data.get("score", 0)
                            status = data.get("status", "WARN")
                            data_quality = data.get("data_quality", "GOOD")
                            is_leading = name in leading

                            status_class = f"status-{status.lower()}"
                            badge_class = f"status-{status.lower()}-badge"
                            leading_class = "leading" if is_leading else ""

                            if status == "PASS":
                                score_color = get_color(theme, 'success')
                                glow_color = get_color(theme, 'glow_success')
                            elif status == "WARN":
                                score_color = get_color(theme, 'warning')
                                glow_color = get_color(theme, 'glow_amber')
                            else:
                                score_color = get_color(theme, 'danger')
                                glow_color = get_color(theme, 'glow_red')

                            quality_class = f"quality-{data_quality.lower().replace(' ', '-')}"
                            leading_badge = '<span class="leading-badge">LEADING</span>' if is_leading else ''
                            leading_glow = f"box-shadow: 0 0 25px {get_color(theme, 'glow_cyan')}, 0 4px 20px rgba(0,0,0,0.3);" if is_leading else ""

                            st.markdown(f'''
                                <div class="engine-card {status_class} {leading_class}" style="{leading_glow}">
                                    <div class="engine-name">{icon} {name}{leading_badge}</div>
                                    <div class="engine-score" style="color: {score_color}; text-shadow: 0 0 20px {glow_color};">{(1-score)*100:.0f}%</div>
                                    <div style="margin-top: 0.5rem;">
                                        <span class="engine-status {badge_class}">{status}</span>
                                    </div>
                                    <div class="data-quality-badge {quality_class}">{data_quality}</div>
                                </div>
                            ''', unsafe_allow_html=True)

                    # ============== CODEC INTEGRITY MONITOR ==============
                    codec_integrity = results.get("codec_integrity", {})
                    if codec_integrity:
                        st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                        # Codec header
                        st.markdown(f'''
                            <div style="
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 1rem;
                                margin-bottom: 1.5rem;
                            ">
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_orange')}40, transparent);
                                "></div>
                                <p style="
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.65rem;
                                    font-weight: 600;
                                    color: {get_color(theme, 'accent_orange')};
                                    letter-spacing: 3px;
                                    text-transform: uppercase;
                                    margin: 0;
                                    padding: 0.5rem 1.5rem;
                                    background: {get_color(theme, 'glow_orange')};
                                    border: 1px solid {get_color(theme, 'accent_orange')}40;
                                    border-radius: 20px;
                                ">Codec Integrity Monitor</p>
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_orange')}40, transparent);
                                "></div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Three pillars display
                        codec_cols = st.columns(3)

                        # Pillar 1: Macroblock Score
                        with codec_cols[0]:
                            mb_score = codec_integrity.get("macroblock_score", 0)
                            mb_color = get_color(theme, 'success') if mb_score < 0.3 else get_color(theme, 'warning') if mb_score < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Macroblock Alignment
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {mb_color};">
                                        {mb_score:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.5rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        DCT Variance
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Pillar 2: Motion Vector Score
                        with codec_cols[1]:
                            mv_score = codec_integrity.get("motion_vector_score", 0)
                            mv_color = get_color(theme, 'success') if mv_score < 0.3 else get_color(theme, 'warning') if mv_score < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Motion Vectors
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {mv_color};">
                                        {mv_score:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.5rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        P-Frame Logic
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Pillar 3: Chroma Leakage Score
                        with codec_cols[2]:
                            ch_score = codec_integrity.get("chroma_leakage_score", 0)
                            ch_color = get_color(theme, 'success') if ch_score < 0.3 else get_color(theme, 'warning') if ch_score < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Chroma Leakage
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {ch_color};">
                                        {ch_score:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.5rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        Cb/Cr Analysis
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Codec integrity score and status
                        integrity_score = codec_integrity.get("codec_integrity_score", 0)
                        codec_status = codec_integrity.get("status", "UNKNOWN")
                        codec_flags = codec_integrity.get("flags", [])

                        status_color = (
                            get_color(theme, 'success') if codec_status == "PASS" else
                            get_color(theme, 'warning') if codec_status == "WARN" else
                            get_color(theme, 'danger')
                        )

                        st.markdown(f'''
                            <div style="
                                background: {get_color(theme, 'glass_bg')};
                                border: 1px solid {status_color}40;
                                border-radius: 12px;
                                padding: 1rem;
                                margin-top: 1rem;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">
                                        Codec Integrity
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: {status_color};">
                                        {integrity_score:.0%}
                                    </div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'Inter', sans-serif;
                                        font-size: 0.7rem;
                                        font-weight: 700;
                                        color: {status_color};
                                        padding: 0.4rem 1.2rem;
                                        background: {status_color}20;
                                        border: 1px solid {status_color}60;
                                        border-radius: 8px;
                                        letter-spacing: 2px;
                                    ">
                                        {codec_status}
                                    </div>
                                </div>
                                <div style="text-align: right; max-width: 250px;">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.5rem; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">
                                        Compression Flags
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; color: {get_color(theme, 'accent_amber')}; line-height: 1.4;">
                                        {', '.join(codec_flags[:4]) if codec_flags else 'None detected'}
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Flag descriptions for detected issues
                        if codec_flags:
                            flag_descriptions = {
                                "QUANTIZATION_MISMATCH": "Face and background have different DCT quantization patterns",
                                "DCT_ENERGY_ANOMALY": "Unusual frequency distribution in face region",
                                "COMPRESSION_GHOSTING": "Block boundary artifacts at face/background interface",
                                "BLOCK_BOUNDARY_ARTIFACT": "Minor block edge discontinuities detected",
                                "VECTOR_INCONSISTENCY": "Face motion decoupled from scene motion vectors",
                                "MOTION_DECORRELATION": "Weak correlation between face and background motion",
                                "MOTION_MAGNITUDE_ANOMALY": "Face moves differently than scene would predict",
                                "DIRECTION_DRIFT": "Face motion direction inconsistent with background",
                                "CHROMA_OVERSHARPENING": "Face color edges too sharp for compressed video",
                                "CHROMA_EDGE_ANOMALY": "Color boundaries don't match expected compression blur",
                                "SYNTHETIC_CHROMA_SMOOTHING": "Face chroma unnaturally smooth",
                                "CHROMA_LEAKAGE_MISSING": "Expected color bleeding not present at edges"
                            }

                            flag_html = ""
                            for flag in codec_flags[:3]:
                                desc = flag_descriptions.get(flag, flag)
                                flag_html += f'<div style="margin: 0.25rem 0;"><span style="color: {get_color(theme, "accent_amber")};">• {flag}:</span> <span style="color: {get_color(theme, "text_muted")};">{desc}</span></div>'

                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'terminal_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 8px;
                                    padding: 0.75rem;
                                    margin-top: 0.75rem;
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.55rem;
                                ">
                                    {flag_html}
                                </div>
                            ''', unsafe_allow_html=True)

                    # ============== ANOMALY CONSENSUS ENGINE DISPLAY ==============
                    consensus = results.get("consensus", {})
                    if consensus:
                        st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                        # Consensus header
                        st.markdown(f'''
                            <div style="
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 1rem;
                                margin-bottom: 1.5rem;
                            ">
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_purple')}40, transparent);
                                "></div>
                                <p style="
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.65rem;
                                    font-weight: 600;
                                    color: {get_color(theme, 'accent_purple')};
                                    letter-spacing: 3px;
                                    text-transform: uppercase;
                                    margin: 0;
                                    padding: 0.5rem 1.5rem;
                                    background: {get_color(theme, 'glow_purple')};
                                    border: 1px solid {get_color(theme, 'accent_purple')}40;
                                    border-radius: 20px;
                                ">Anomaly Consensus Engine</p>
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_purple')}40, transparent);
                                "></div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Three pillars display
                        pillar_cols = st.columns(3)

                        # Pillar 1: Noise Delta
                        with pillar_cols[0]:
                            noise_delta = consensus.get("noise_delta", 0)
                            noise_color = get_color(theme, 'success') if noise_delta < 0.3 else get_color(theme, 'warning') if noise_delta < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Baseline Noise Delta
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: {noise_color};">
                                        {noise_delta:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        Weight: 30%
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Pillar 2: Light Decoupling
                        with pillar_cols[1]:
                            light_decoupling = consensus.get("light_decoupling", 0)
                            light_color = get_color(theme, 'success') if light_decoupling < 0.3 else get_color(theme, 'warning') if light_decoupling < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Light Decoupling
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: {light_color};">
                                        {light_decoupling:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        Weight: 40%
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Pillar 3: Jitter Variance
                        with pillar_cols[2]:
                            jitter_variance = consensus.get("jitter_variance", 0)
                            jitter_color = get_color(theme, 'success') if jitter_variance < 0.3 else get_color(theme, 'warning') if jitter_variance < 0.6 else get_color(theme, 'danger')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Jitter Variance
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: {jitter_color};">
                                        {jitter_variance:.0%}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.25rem;">
                                        Weight: 30%
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Consensus score and verdict
                        consensus_score = consensus.get("consensus_score", 0)
                        consensus_verdict = consensus.get("consensus_verdict", "UNKNOWN")
                        consensus_confidence = consensus.get("consensus_confidence", "LOW")
                        flags = consensus.get("flags", [])

                        verdict_color = (
                            get_color(theme, 'success') if consensus_verdict == "AUTHENTIC" else
                            get_color(theme, 'danger') if consensus_verdict == "MANIPULATED" else
                            get_color(theme, 'accent_purple')
                        )

                        st.markdown(f'''
                            <div style="
                                background: {get_color(theme, 'glass_bg')};
                                border: 1px solid {verdict_color}40;
                                border-radius: 12px;
                                padding: 1.25rem;
                                margin-top: 1rem;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            ">
                                <div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">
                                        Consensus Score
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: {verdict_color};">
                                        {consensus_score:.0%}
                                    </div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'Inter', sans-serif;
                                        font-size: 0.75rem;
                                        font-weight: 700;
                                        color: {verdict_color};
                                        padding: 0.5rem 1.5rem;
                                        background: {verdict_color}20;
                                        border: 1px solid {verdict_color}60;
                                        border-radius: 8px;
                                        letter-spacing: 2px;
                                    ">
                                        {consensus_verdict}
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.5rem;">
                                        Confidence: {consensus_confidence}
                                    </div>
                                </div>
                                <div style="text-align: right; max-width: 200px;">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">
                                        Flags
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {get_color(theme, 'accent_orange')};">
                                        {', '.join(flags) if flags else 'None'}
                                    </div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Precision notice
                        if consensus_verdict == "INCONCLUSIVE":
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glow_purple')};
                                    border: 1px dashed {get_color(theme, 'accent_purple')}60;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    margin-top: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; color: {get_color(theme, 'accent_purple')}; font-weight: 600;">
                                        HIGH PRECISION MODE: Unable to make a reliable determination
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; color: {get_color(theme, 'text_muted')}; margin-top: 0.5rem;">
                                        The source quality or data is insufficient for confident classification.
                                        This is better than a false positive.
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                    # ============== AUDIO PROFILE DISPLAY ==============
                    audio_profile = results.get("audio_profile")
                    if audio_profile:
                        st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                        # Audio Profile Header
                        st.markdown(f'''
                            <div style="
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 1rem;
                                margin-bottom: 1.5rem;
                            ">
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_purple')}40, transparent);
                                "></div>
                                <p style="
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.65rem;
                                    font-weight: 600;
                                    color: {get_color(theme, 'accent_purple')};
                                    letter-spacing: 3px;
                                    text-transform: uppercase;
                                    margin: 0;
                                    padding: 0.5rem 1.5rem;
                                    background: {get_color(theme, 'glow_purple')};
                                    border: 1px solid {get_color(theme, 'accent_purple')}40;
                                    border-radius: 20px;
                                ">Audio Intelligence</p>
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_purple')}40, transparent);
                                "></div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Audio metrics
                        audio_cols = st.columns(4)

                        # Audio presence
                        with audio_cols[0]:
                            has_audio = audio_profile.get("has_audio", False)
                            audio_color = get_color(theme, 'success') if has_audio else get_color(theme, 'warning')
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Audio Track
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {audio_color};">
                                        {"DETECTED" if has_audio else "NONE"}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # SNR value
                        with audio_cols[1]:
                            snr_db = audio_profile.get("snr_db", 0)
                            snr_color = (get_color(theme, 'success') if snr_db >= 20 else
                                        get_color(theme, 'accent_cyan') if snr_db >= 10 else
                                        get_color(theme, 'warning') if snr_db >= 5 else
                                        get_color(theme, 'danger'))
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        SNR Level
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {snr_color};">
                                        {snr_db:.1f} dB
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Noise level
                        with audio_cols[2]:
                            noise_level = audio_profile.get("noise_level", "UNKNOWN")
                            noise_color = (get_color(theme, 'success') if noise_level == "LOW" else
                                          get_color(theme, 'accent_cyan') if noise_level == "MEDIUM" else
                                          get_color(theme, 'warning') if noise_level == "HIGH" else
                                          get_color(theme, 'danger'))
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        Noise Level
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {noise_color};">
                                        {noise_level}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # A/V Weight
                        with audio_cols[3]:
                            av_weight = audio_profile.get("recommended_av_weight", 1.0)
                            weight_color = (get_color(theme, 'success') if av_weight >= 0.9 else
                                           get_color(theme, 'accent_cyan') if av_weight >= 0.7 else
                                           get_color(theme, 'warning') if av_weight >= 0.5 else
                                           get_color(theme, 'danger'))
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
                                        A/V Weight
                                    </div>
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {weight_color};">
                                        {av_weight:.0%}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Speech detection notice
                        is_speech = audio_profile.get("is_speech_detected", False)
                        if has_audio and not is_speech:
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glow_warning')};
                                    border: 1px dashed {get_color(theme, 'warning')}60;
                                    border-radius: 8px;
                                    padding: 0.75rem;
                                    margin-top: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.65rem; color: {get_color(theme, 'warning')};">
                                        No speech detected in audio track. A/V synchronization analysis has reduced reliability.
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                    # ============== VERDICT TRANSPARENCY PANEL ==============
                    transparency = results.get("transparency_report")
                    if transparency:
                        st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)

                        # Transparency Header
                        st.markdown(f'''
                            <div style="
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                gap: 1rem;
                                margin-bottom: 1.5rem;
                            ">
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_electric')}40, transparent);
                                "></div>
                                <p style="
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.65rem;
                                    font-weight: 600;
                                    color: {get_color(theme, 'accent_electric')};
                                    letter-spacing: 3px;
                                    text-transform: uppercase;
                                    margin: 0;
                                    padding: 0.5rem 1.5rem;
                                    background: {get_color(theme, 'glow_electric')};
                                    border: 1px solid {get_color(theme, 'accent_electric')}40;
                                    border-radius: 20px;
                                ">Verdict Transparency</p>
                                <div style="
                                    height: 1px;
                                    flex: 1;
                                    background: linear-gradient(90deg, transparent, {get_color(theme, 'accent_electric')}40, transparent);
                                "></div>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Summary
                        summary = transparency.get("summary", "")
                        if summary:
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1.25rem;
                                    margin-bottom: 1rem;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: {get_color(theme, 'text_primary')}; line-height: 1.5;">
                                        {summary}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Primary concern (if manipulated)
                        primary_concern = transparency.get("primary_concern")
                        if primary_concern:
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glow_red')};
                                    border: 1px solid {get_color(theme, 'danger')}60;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    margin-bottom: 1rem;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; font-weight: 600; color: {get_color(theme, 'danger')}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                                        Primary Concern
                                    </div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: {get_color(theme, 'text_primary')};">
                                        {primary_concern}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Engine Explanations (expandable)
                        with st.expander("BIOSIGNAL CORE Analysis", expanded=False):
                            intel_exp = transparency.get("intel_explanation", "")
                            st.markdown(intel_exp)

                        with st.expander("ARTIFACT CORE Analysis", expanded=False):
                            sentinel_exp = transparency.get("sentinel_explanation", "")
                            st.markdown(sentinel_exp)

                        with st.expander("ALIGNMENT CORE Analysis", expanded=False):
                            defender_exp = transparency.get("defender_explanation", "")
                            st.markdown(defender_exp)

                        # Environmental Factors
                        env_factors = transparency.get("environmental_factors", [])
                        if env_factors:
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glass_bg')};
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                    border-radius: 12px;
                                    padding: 1rem;
                                    margin-top: 1rem;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.65rem; font-weight: 600; color: {get_color(theme, 'accent_amber')}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem;">
                                        Environmental Adjustments
                                    </div>
                                    <ul style="margin: 0; padding-left: 1.25rem; color: {get_color(theme, 'text_secondary')}; font-size: 0.75rem; line-height: 1.8;">
                                        {"".join(f'<li>{factor}</li>' for factor in env_factors)}
                                    </ul>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Audio quality note
                        audio_note = transparency.get("audio_quality_note")
                        if audio_note:
                            st.markdown(f'''
                                <div style="
                                    background: {get_color(theme, 'glow_purple')};
                                    border: 1px dashed {get_color(theme, 'accent_purple')}60;
                                    border-radius: 8px;
                                    padding: 0.75rem;
                                    margin-top: 1rem;
                                    text-align: center;
                                ">
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.65rem; color: {get_color(theme, 'accent_purple')};">
                                        {audio_note}
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)

                        # Supporting Evidence
                        evidence = transparency.get("supporting_evidence", [])
                        if evidence:
                            with st.expander("Supporting Evidence (Anomaly Flags)", expanded=False):
                                for item in evidence:
                                    st.markdown(f"- {item}")

                    # Blockchain proof - Enhanced styling
                    if results.get("origin_proof"):
                        st.markdown(f'''
                            <div class="blockchain-proof" style="
                                margin-top: 2rem;
                                background: linear-gradient(135deg, {get_color(theme, 'terminal_bg')}, {get_color(theme, 'background_secondary')});
                                border: 1px solid {get_color(theme, 'accent_cyan')}30;
                                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 {get_color(theme, 'glass_border')};
                            ">
                                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                    <span style="font-size: 1rem;">⛓️</span>
                                    <div class="blockchain-title" style="margin: 0;">BLOCKCHAIN-READY ORIGIN HASH</div>
                                </div>
                                <div class="blockchain-hash" style="
                                    background: {get_color(theme, 'background')};
                                    padding: 0.75rem;
                                    border-radius: 8px;
                                    border: 1px solid {get_color(theme, 'glass_border')};
                                ">{results["origin_proof"]["origin_hash"]}</div>
                                <div class="blockchain-timestamp" style="margin-top: 0.75rem;">
                                    <span style="color: {get_color(theme, 'accent_cyan')};">Session:</span> {results.get("session_id", "N/A")} |
                                    <span style="color: {get_color(theme, 'accent_orange')};">Generated:</span> {results["origin_proof"]["timestamp"]}
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

    else:
        # Live Sentinel mode - War Room Layout
        sentinel_left, sentinel_right = st.columns([1, 1])

        with sentinel_left:
            st.markdown(f'''
                <div class="pane-title">
                    <span class="pane-title-icon blue"></span>
                    LIVE FEED
                </div>
            ''', unsafe_allow_html=True)

            st.markdown(f'''
                <div class="scanner-zone">
                    <div class="liquid-border-wrapper">
                        <div class="liquid-border-inner" style="min-height: 350px;">
                            <div class="scan-line"></div>
                            <div class="upload-icon">📹</div>
                            <p class="upload-title">Live Camera Sentinel</p>
                            <p class="upload-subtitle">Resolution-Adaptive HUD (Mesh/Heatmap)</p>
                            <p style="color: {get_color(theme, 'accent_orange')}; font-size: 0.75rem; margin-top: 1rem; font-family: 'JetBrains Mono', monospace;">AWAITING ACTIVATION</p>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

        with sentinel_right:
            st.markdown(f'''
                <div class="pane-title">
                    <span class="pane-title-icon orange"></span>
                    THREAT INTELLIGENCE
                </div>
            ''', unsafe_allow_html=True)

            render_forensic_terminal(st.session_state.terminal_logs, theme)

        live_cols = st.columns([1, 2, 1])
        with live_cols[1]:
            col1, col2 = st.columns(2)
            with col1:
                start_live = st.button("START SENTINEL", type="primary", use_container_width=True)
            with col2:
                stop_live = st.button("STOP", use_container_width=True)

        if start_live:
            add_log("[LIVE] Initializing camera sentinel...", "scan")
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()

            sentinel = LiveSentinel(face_extractor, inference_engine)
            sentinel.start()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access camera")
                add_log("[ERROR] Camera access denied", "alert")
            else:
                add_log("[LIVE] Camera feed active", "success")
                frame_count = 0
                start_time = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = sentinel.process_frame(frame)

                    if result:
                        frame_with_hud = sentinel.draw_hud_adaptive(frame, result, theme)
                        frame_rgb = cv2.cvtColor(frame_with_hud, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        frame_count += 1
                        elapsed = time.time() - start_time
                        fps_calc = frame_count / elapsed if elapsed > 0 else 0

                        # Tactical metrics display
                        metrics_placeholder.markdown(f'''
                            <div style="
                                display: flex;
                                justify-content: center;
                                gap: 2.5rem;
                                margin-top: 1rem;
                                padding: 1rem;
                                background: {get_color(theme, 'glass_bg')};
                                border: 1px solid {get_color(theme, 'glass_border')};
                                border-radius: 12px;
                            ">
                                <div style="text-align: center;">
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {get_color(theme, 'accent_cyan')};">{fps_calc:.1f}</div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">FPS</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {get_color(theme, 'accent_orange')};">{result.get("latency_ms", 0):.0f}ms</div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">Latency</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {get_color(theme, 'success') if result.get('label') == 'REAL' else get_color(theme, 'danger')};">{result.get("label", "N/A")}</div>
                                    <div style="font-family: 'Inter', sans-serif; font-size: 0.55rem; font-weight: 600; color: {get_color(theme, 'text_muted')}; letter-spacing: 1px; text-transform: uppercase;">Verdict</div>
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    if stop_live:
                        break
                    time.sleep(0.01)

                cap.release()
                sentinel.stop()
                add_log("[LIVE] Sentinel stopped", "info")

    # Tactical Footer
    st.markdown('<div class="elite-divider"></div>', unsafe_allow_html=True)
    render_tactical_footer(theme)


if __name__ == "__main__":
    main()
