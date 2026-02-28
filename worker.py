"""
Scanner - Celery Async Worker (v3.3.0 Enterprise)
==================================================

Provides asynchronous video analysis via Celery task queue.
Enables horizontal scaling by running multiple workers.

v3.3.0 Enhancements:
- Automatic retries with exponential backoff (max 3 attempts)
- Dead-letter queue for permanently failed tasks
- Forensic hash verification before persisting results
- Structured JSON logging
- Uses shared ModelManager for memory efficiency

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

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from core.exceptions import (
    NoFramesExtractedError,
    ProcessingError,
    VideoDecodeError,
)
from core.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("worker")

# Celery configuration
try:
    from celery import Celery
    from celery.exceptions import SoftTimeLimitExceeded

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
        task_time_limit=600,
        task_soft_time_limit=300,
        worker_max_tasks_per_child=50,
        worker_prefetch_multiplier=1,
        # v3.3.0: Retry and dead-letter configuration
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=30,
        task_max_retries=3,
    )

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    app = None
    SoftTimeLimitExceeded = Exception
    logger.warning("Celery not installed. Async processing unavailable.")


# --- Model management via shared ModelManager ---

_mm = None


def _get_model_manager():
    """Get or initialise the ModelManager for this worker process."""
    global _mm
    if _mm is None:
        from utils.model_manager import ModelManager
        _mm = ModelManager.get_instance()
        _mm.initialise_all()
        logger.info("Worker ModelManager initialised")
    return _mm


def _verify_result_integrity(result: Dict[str, Any]) -> str:
    """
    Compute SHA-256 of the result dict for integrity verification.

    Returns:
        Hex-encoded hash.
    """
    canonical = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


if CELERY_AVAILABLE:

    @app.task(
        bind=True,
        name="scanner.analyze_video",
        autoretry_for=(ProcessingError,),
        retry_backoff=True,
        retry_backoff_max=120,
        max_retries=3,
        acks_late=True,
    )
    def analyze_video_task(
        self,
        video_path: str,
        filename: str,
        session_id: str,
        user: str = "async_worker",
        max_frames: int = 60,
    ) -> Dict[str, Any]:
        """
        Async video analysis task with retry and integrity verification.

        Args:
            video_path: Path to the video file on disk.
            filename: Original filename.
            session_id: Unique session identifier.
            user: Username who initiated the scan.
            max_frames: Maximum frames to extract.

        Returns:
            Analysis result dictionary with integrity hash.
        """
        import cv2

        from core.forensic_types import VideoProfile
        from services.video_profiler import get_resolution_tier
        from utils.forensic_hash import ForensicHashChain

        start_time = time.time()
        mm = _get_model_manager()
        hash_chain = ForensicHashChain(session_id)

        logger.info(
            "Worker task started",
            extra={"session_id": session_id, "user": user, "stage": "start"},
        )

        try:
            self.update_state(state="PROCESSING", meta={"stage": "hashing_original"})
            hash_chain.hash_file(video_path)

            # Profile video
            self.update_state(state="PROCESSING", meta={"stage": "profiling"})
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoDecodeError(f"Could not open video: {video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            tier = get_resolution_tier(height)

            video_profile = VideoProfile(
                width=width, height=height, fps=fps,
                frame_count=frame_count, duration_seconds=duration,
                resolution_tier=tier, pixel_count=width * height,
                aspect_ratio=width / height if height > 0 else 1.0,
                rppg_viable=height >= 480 and fps >= 24,
                mesh_viable=height >= 720,
                recommended_analysis="PRIME HYBRID",
            )

            # Extract frames
            self.update_state(state="PROCESSING", meta={"stage": "extracting_frames"})
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
                raise NoFramesExtractedError()

            hash_chain.hash_frames(frames)

            # Sanity check
            self.update_state(state="PROCESSING", meta={"stage": "sanity_check"})
            sanity = mm.sanity_guard.validate(frames)
            if not sanity.is_valid:
                return {
                    "error": "Input validation failed",
                    "reason": sanity.rejection_reason,
                    "session_id": session_id,
                }

            # Run PRIME HYBRID
            self.update_state(state="PROCESSING", meta={"stage": "biosignal_analysis"})
            biosignal_result = mm.biosignal_core.analyze(frames, fps, video_profile)

            self.update_state(state="PROCESSING", meta={"stage": "artifact_analysis"})
            artifact_result = mm.artifact_core.analyze(frames, video_profile)

            self.update_state(state="PROCESSING", meta={"stage": "alignment_analysis"})
            alignment_result = mm.alignment_core.analyze(frames, fps, video_path, video_profile)

            # Audio analysis (best-effort)
            self.update_state(state="PROCESSING", meta={"stage": "audio_analysis"})
            audio_profile = None
            try:
                audio_profile = mm.audio_analyzer.analyze(video_path)
            except Exception as exc:
                logger.warning(f"Audio analysis failed in worker: {exc}")

            # Fusion
            self.update_state(state="PROCESSING", meta={"stage": "fusion"})
            verdict = mm.fusion_engine.get_final_integrity_score(
                biosignal_result, artifact_result, alignment_result,
                video_profile, audio_profile,
            )

            duration_ms = (time.time() - start_time) * 1000

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
                "duration_ms": round(duration_ms, 1),
                "forensic_hashes": hash_chain.summary(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Integrity verification: hash the result itself
            result_hash = _verify_result_integrity(result)
            result["result_integrity_hash"] = result_hash

            logger.info(
                "Worker task completed",
                extra={
                    "session_id": session_id,
                    "user": user,
                    "stage": "complete",
                    "duration_ms": round(duration_ms, 1),
                },
            )

            return result

        except SoftTimeLimitExceeded:
            logger.error(
                "Worker task timed out",
                extra={"session_id": session_id, "stage": "timeout"},
            )
            return {
                "error": "Analysis timed out",
                "session_id": session_id,
            }
        except (VideoDecodeError, NoFramesExtractedError) as exc:
            logger.error(
                f"Worker input error: {exc}",
                extra={"session_id": session_id, "error_code": exc.error_code},
            )
            return {
                "error": exc.message,
                "error_code": exc.error_code,
                "session_id": session_id,
            }
        except ProcessingError:
            raise  # Let Celery's autoretry handle this
        except Exception as exc:
            logger.error(
                f"Worker unexpected error: {exc}",
                extra={"session_id": session_id},
                exc_info=True,
            )
            if self.request.retries < self.max_retries:
                raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)
            return {
                "error": f"Analysis failed: {exc}",
                "session_id": session_id,
            }


    @app.task(name="scanner.dead_letter", bind=True)
    def dead_letter_task(self, original_task_id: str, exc_info: str, session_id: str = ""):
        """
        Dead-letter handler for permanently failed tasks.
        Logs the failure for manual review.
        """
        logger.error(
            "Task moved to dead-letter queue",
            extra={
                "session_id": session_id,
                "stage": "dead_letter",
                "error_code": "PERMANENT_FAILURE",
            },
        )
        return {
            "status": "dead_letter",
            "original_task_id": original_task_id,
            "error": exc_info,
            "session_id": session_id,
        }
