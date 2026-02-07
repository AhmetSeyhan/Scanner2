"""
Scanner - Celery Async Worker
=============================

Provides asynchronous video analysis via Celery task queue.
Enables horizontal scaling by running multiple workers.

Usage:
    # Start worker
    celery -A worker worker --loglevel=info --concurrency=2

    # Start worker with GPU (single concurrency to avoid GPU memory issues)
    celery -A worker worker --loglevel=info --concurrency=1

    # Monitor tasks
    celery -A worker flower

Requirements:
    pip install celery[redis]

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import os
import hashlib
import tempfile
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Celery configuration
try:
    from celery import Celery

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    app = Celery(
        "scanner",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=600,       # 10 minute hard limit
        task_soft_time_limit=300,  # 5 minute soft limit
        worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory cleanup)
        worker_prefetch_multiplier=1,   # One task at a time per worker
    )

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    app = None
    logger.warning("Celery not installed. Async processing unavailable.")


def _init_cores():
    """Lazily initialize PRIME HYBRID cores (called once per worker process)."""
    import cv2
    from core.biosignal_core import BioSignalCore
    from core.artifact_core import ArtifactCore
    from core.alignment_core import AlignmentCore
    from core.fusion_engine import FusionEngine
    from core.audio_analyzer import AudioAnalyzer
    from core.input_sanity_guard import InputSanityGuard

    return {
        "cv2": cv2,
        "biosignal": BioSignalCore(),
        "artifact": ArtifactCore(),
        "alignment": AlignmentCore(),
        "fusion": FusionEngine(),
        "audio": AudioAnalyzer(),
        "sanity": InputSanityGuard(),
    }


# Module-level cache for cores (initialized once per worker process)
_cores = None


def get_cores():
    """Get or initialize PRIME HYBRID cores."""
    global _cores
    if _cores is None:
        _cores = _init_cores()
    return _cores


if CELERY_AVAILABLE:

    @app.task(bind=True, name="scanner.analyze_video")
    def analyze_video_task(
        self,
        video_path: str,
        filename: str,
        session_id: str,
        user: str = "async_worker",
        max_frames: int = 60,
    ) -> Dict[str, Any]:
        """
        Async video analysis task.

        Args:
            video_path: Path to the video file on disk
            filename: Original filename
            session_id: Unique session identifier
            user: Username who initiated the scan
            max_frames: Maximum frames to extract

        Returns:
            Analysis result dictionary
        """
        from core.forensic_types import VideoProfile, ResolutionTier

        cores = get_cores()
        cv2 = cores["cv2"]

        self.update_state(state="PROCESSING", meta={"stage": "extracting_frames"})

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file", "session_id": session_id}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Resolution tier
        if height <= 360:
            tier = ResolutionTier.ULTRA_LOW
        elif height <= 480:
            tier = ResolutionTier.LOW
        elif height <= 720:
            tier = ResolutionTier.MEDIUM
        elif height <= 1080:
            tier = ResolutionTier.HIGH
        else:
            tier = ResolutionTier.ULTRA_HIGH

        video_profile = VideoProfile(
            width=width, height=height, fps=fps,
            frame_count=frame_count, duration_seconds=duration,
            resolution_tier=tier, pixel_count=width * height,
            aspect_ratio=width / height if height > 0 else 1.0,
            rppg_viable=height >= 480 and fps >= 24,
            mesh_viable=height >= 720,
            recommended_analysis="PRIME HYBRID"
        )

        # Extract frames
        interval = max(1, frame_count // max_frames)
        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        if not frames:
            return {"error": "No frames extracted", "session_id": session_id}

        # Sanity check
        self.update_state(state="PROCESSING", meta={"stage": "sanity_check"})
        sanity = cores["sanity"].validate(frames)
        if not sanity.is_valid:
            return {
                "error": "Input validation failed",
                "reason": sanity.rejection_reason,
                "session_id": session_id
            }

        # Run PRIME HYBRID
        self.update_state(state="PROCESSING", meta={"stage": "biosignal_analysis"})
        biosignal_result = cores["biosignal"].analyze(frames, fps, video_profile)

        self.update_state(state="PROCESSING", meta={"stage": "artifact_analysis"})
        artifact_result = cores["artifact"].analyze(frames, video_profile)

        self.update_state(state="PROCESSING", meta={"stage": "alignment_analysis"})
        alignment_result = cores["alignment"].analyze(frames, fps, video_path, video_profile)

        # Audio analysis
        self.update_state(state="PROCESSING", meta={"stage": "audio_analysis"})
        audio_profile = None
        try:
            audio_profile = cores["audio"].analyze(video_path)
        except Exception:
            pass

        # Fusion
        self.update_state(state="PROCESSING", meta={"stage": "fusion"})
        verdict = cores["fusion"].get_final_integrity_score(
            biosignal_result, artifact_result, alignment_result,
            video_profile, audio_profile
        )

        # Build result
        result = {
            "session_id": session_id,
            "filename": filename,
            "analyzed_by": user,
            "verdict": verdict.verdict,
            "integrity_score": verdict.integrity_score,
            "confidence": verdict.confidence,
            "core_scores": {
                "biosignal": verdict.biosignal_score,
                "artifact": verdict.artifact_score,
                "alignment": verdict.alignment_score,
            },
            "consensus_type": verdict.consensus_type,
            "leading_core": verdict.leading_core,
            "frames_analyzed": len(frames),
            "video_profile": {
                "resolution": f"{height}p",
                "fps": fps,
                "duration": duration,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result
