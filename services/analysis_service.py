"""
Scanner Prime - Analysis Service
Core business logic for deepfake analysis.

Orchestrates the full PRIME HYBRID pipeline:
  1. Video profiling
  2. Frame extraction
  3. Input sanity check
  4. Audio analysis
  5. BIOSIGNAL / ARTIFACT / ALIGNMENT core execution
  6. Fusion engine verdict
  7. Forensic hash chain
  8. History recording and audit logging

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import time
import uuid
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.exceptions import (
    NoFaceDetectedError,
    ProcessingError,
)
from core.logging_config import get_logger
from services.video_profiler import VideoProfiler
from utils.forensic_hash import ForensicHashChain
from utils.model_manager import ModelManager

logger = get_logger("analysis_service")


class AnalysisService:
    """
    Stateless service that runs PRIME HYBRID analysis.

    All state (models, history, etc.) is obtained via ModelManager
    and injected dependencies.
    """

    def __init__(self) -> None:
        self.profiler = VideoProfiler()

    def analyze_video_v2(
        self,
        video_path: str,
        filename: str,
        user: str,
        session_id: Optional[str] = None,
        max_frames: int = 60,
    ) -> Dict[str, Any]:
        """
        Run full PRIME HYBRID analysis on a video file.

        Args:
            video_path: Path to the video on disk.
            filename: Original upload filename.
            user: Username performing the analysis.
            session_id: Optional session ID (generated if not provided).
            max_frames: Maximum frames to sample.

        Returns:
            Complete analysis result dict.

        Raises:
            ScannerBaseError subtypes for each failure mode.
        """
        session_id = session_id or str(uuid.uuid4())[:8].upper()
        start_time = time.time()
        mm = ModelManager.get_instance()
        hash_chain = ForensicHashChain(session_id)

        logger.info(
            "Analysis started",
            extra={"session_id": session_id, "user": user, "stage": "start"},
        )

        # 1. Forensic hash of original file
        hash_chain.hash_file(video_path)

        # 2. Video profiling
        video_profile = self.profiler.profile_video(video_path)

        # 3. Frame extraction
        frames = self.profiler.extract_frames(video_path, max_frames)
        hash_chain.hash_frames(frames)

        # 4. Audio analysis (best-effort)
        audio_profile = None
        try:
            audio_profile = mm.audio_analyzer.analyze(video_path)
        except Exception as exc:
            logger.warning(
                f"Audio analysis failed: {exc}",
                extra={"session_id": session_id, "stage": "audio_analysis"},
            )

        # 5. Run PRIME HYBRID cores
        try:
            biosignal_result = mm.biosignal_core.analyze(
                frames, video_profile.fps, video_profile
            )
        except Exception as exc:
            raise ProcessingError("biosignal_analysis", str(exc)) from exc

        try:
            artifact_result = mm.artifact_core.analyze(frames, video_profile)
        except Exception as exc:
            raise ProcessingError("artifact_analysis", str(exc)) from exc

        try:
            alignment_result = mm.alignment_core.analyze(
                frames, video_profile.fps, video_path, video_profile
            )
        except Exception as exc:
            raise ProcessingError("alignment_analysis", str(exc)) from exc

        # 6. Fusion verdict
        try:
            verdict = mm.fusion_engine.get_final_integrity_score(
                biosignal_result, artifact_result, alignment_result,
                video_profile, audio_profile,
            )
        except Exception as exc:
            raise ProcessingError("fusion", str(exc)) from exc

        # 6b. Generate explainability artifacts (PPG map + heatmap)
        explainability = {}
        try:
            ppg_data = mm.biosignal_core.generate_ppg_map(frames, video_profile.fps)
            explainability["ppg_map"] = {
                "mean_ppg_strength": ppg_data["mean_ppg_strength"],
                "ppg_coverage": ppg_data["ppg_coverage"],
                "mean_quality": ppg_data["mean_quality"],
                "grid_size": ppg_data["grid_size"],
                "status": ppg_data["status"],
            }
        except Exception as exc:
            logger.warning(
                f"PPG map generation failed: {exc}",
                extra={"session_id": session_id, "stage": "ppg_map"},
            )

        try:
            if artifact_result.heatmap:
                explainability["artifact_heatmap"] = artifact_result.heatmap.to_dict()
            elif len(frames) > 0:
                heatmap = mm.artifact_core.generate_spatial_heatmap(
                    frames[len(frames) // 2]
                )
                explainability["artifact_heatmap"] = heatmap.to_dict()
        except Exception as exc:
            logger.warning(
                f"Heatmap generation failed: {exc}",
                extra={"session_id": session_id, "stage": "heatmap"},
            )

        duration_ms = (time.time() - start_time) * 1000

        # 7. Build response
        height = video_profile.height
        tier = video_profile.resolution_tier

        response = {
            "session_id": session_id,
            "filename": filename,
            "analyzed_by": user,
            "api_version": "v3",
            "architecture": "PRIME HYBRID",
            # Verdict
            "verdict": verdict.verdict,
            "integrity_score": verdict.integrity_score,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "consensus_type": verdict.consensus_type,
            "leading_core": verdict.leading_core,
            "conflicting_signals": verdict.conflicting_signals,
            "fusion_method": verdict.fusion_method,
            # Core scores
            "core_scores": {
                "biosignal": verdict.biosignal_score,
                "artifact": verdict.artifact_score,
                "alignment": verdict.alignment_score,
            },
            "weights": verdict.weights,
            # Video profile
            "video_profile": {
                "resolution": f"{height}p",
                "resolution_tier": tier.value,
                "fps": video_profile.fps,
                "duration_seconds": video_profile.duration_seconds,
                "rppg_viable": video_profile.rppg_viable,
            },
            # Core details
            "cores": {
                "biosignal": {
                    "biological_sync": biosignal_result.biological_sync_score,
                    "pulse_coverage": biosignal_result.pulse_coverage,
                    "hr_consistency": biosignal_result.hr_consistency,
                    "status": biosignal_result.status,
                    "anomalies": biosignal_result.anomalies,
                },
                "artifact": {
                    "gan_score": artifact_result.gan_score,
                    "diffusion_score": artifact_result.diffusion_score,
                    "vae_score": artifact_result.vae_score,
                    "structural_integrity": artifact_result.structural_integrity,
                    "detected_model": artifact_result.details.get("detected_model_type", "NONE"),
                    "status": artifact_result.status,
                    "anomalies": artifact_result.anomalies,
                },
                "alignment": {
                    "av_alignment": alignment_result.av_alignment_score,
                    "phoneme_viseme": alignment_result.phoneme_viseme_score,
                    "speech_rhythm": alignment_result.speech_rhythm_score,
                    "lip_closures": len(alignment_result.lip_closure_events),
                    "status": alignment_result.status,
                    "anomalies": alignment_result.anomalies,
                },
            },
            "frames_analyzed": len(frames),
            "duration_ms": round(duration_ms, 1),
            # Explainability (v4.0.0)
            "explainability": explainability,
            # Forensic integrity
            "forensic_hashes": hash_chain.summary(),
        }

        # 8. Hash the result itself for tamper evidence
        hash_chain.hash_result(response)
        response["forensic_hashes"] = hash_chain.summary()

        # Audio profile
        if audio_profile:
            response["audio_profile"] = {
                "has_audio": audio_profile.has_audio,
                "snr_db": round(audio_profile.snr_db, 2),
                "noise_level": audio_profile.noise_level,
                "recommended_av_weight": round(audio_profile.recommended_av_weight, 3),
                "is_speech_detected": audio_profile.is_speech_detected,
                "duration_seconds": round(audio_profile.duration_seconds, 2),
            }

        # Transparency report
        if verdict.transparency_report:
            tr = verdict.transparency_report
            response["transparency"] = {
                "summary": tr.summary,
                "primary_concern": tr.primary_concern,
                "biosignal_explanation": tr.biosignal_explanation,
                "artifact_explanation": tr.artifact_explanation,
                "alignment_explanation": tr.alignment_explanation,
                "environmental_factors": tr.environmental_factors,
                "supporting_evidence": tr.supporting_evidence,
                "weight_justification": tr.weight_justification,
                "fusion_method": tr.fusion_method,
                "audio_quality_note": tr.audio_quality_note,
            }

        logger.info(
            "Analysis complete",
            extra={
                "session_id": session_id,
                "user": user,
                "stage": "complete",
                "duration_ms": round(duration_ms, 1),
            },
        )

        return response

    def analyze_video_v1(
        self,
        video_path: str,
        filename: str,
        user: str,
    ) -> Dict[str, Any]:
        """
        Run legacy v1 analysis (EfficientNet-B0 per-frame classification).

        Args:
            video_path: Path to the video on disk.
            filename: Original upload filename.
            user: Username.

        Returns:
            Analysis result dict.
        """
        mm = ModelManager.get_instance()
        processed_frames = mm.video_processor.process_video(video_path)

        if not processed_frames:
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "message": "No faces detected in the video",
                "frames_analyzed": 0,
                "analyzed_by": user,
            }

        frame_results = []
        for frame_num, face_tensor in processed_frames:
            prob, label = mm.inference_engine.predict_single(face_tensor)
            frame_results.append((frame_num, prob, label))

        analysis = mm.inference_engine.analyze_video_results(frame_results)
        return {"filename": filename, "analyzed_by": user, **analysis}

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        user: str,
    ) -> Dict[str, Any]:
        """
        Run single-image analysis.

        Args:
            image_bytes: Raw image bytes.
            filename: Original filename.
            user: Username.

        Returns:
            Analysis result dict.
        """
        mm = ModelManager.get_instance()

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ProcessingError("image_decode", "Could not decode image")

        face = mm.face_extractor.extract_primary_face(image)
        if face is None:
            raise NoFaceDetectedError()

        preprocessed = mm.face_extractor.preprocess_for_model(face)
        prob, label = mm.inference_engine.predict_single(preprocessed)

        return {
            "filename": filename,
            "verdict": label,
            "fake_probability": prob,
            "confidence": prob if label == "FAKE" else 1 - prob,
            "analyzed_by": user,
        }
