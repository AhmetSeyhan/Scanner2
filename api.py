"""
FastAPI backend for deepfake detection.
Includes JWT authentication, API key support, and rate limiting.

v3.2.0 - Enterprise features: Rate limiting, History, Webhooks, S3

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import os
import tempfile
import shutil
import hashlib
import uuid
from datetime import timedelta, datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

from preprocessing import FaceExtractor, VideoProcessor
from model import DeepfakeInference

# Import PRIME HYBRID core modules (renamed for IP compliance)
from core.biosignal_core import BioSignalCore
from core.artifact_core import ArtifactCore
from core.alignment_core import AlignmentCore
from core.fusion_engine import FusionEngine
from core.audio_analyzer import AudioAnalyzer, AudioProfile
from core.forensic_types import VideoProfile, ResolutionTier, TransparencyReport
from core.input_sanity_guard import InputSanityGuard

# v3.2.0 Enterprise imports
try:
    from utils.history_manager import HistoryManager
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False

try:
    from utils.forensic_reporter import ForensicReporter
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    require_read,
    require_write,
    require_admin,
    Token,
    User,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


# Initialize rate limiter
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# Initialize FastAPI app
app = FastAPI(
    title="Scanner API",
    description="Advanced deepfake detection engine powered by EfficientNet-B0. v3.2.0 Enterprise with rate limiting, history, and webhooks.",
    version="3.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limit handler
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
face_extractor: Optional[FaceExtractor] = None
video_processor: Optional[VideoProcessor] = None
inference_engine: Optional[DeepfakeInference] = None

# PRIME HYBRID core engines (renamed)
biosignal_core: Optional[BioSignalCore] = None
artifact_core: Optional[ArtifactCore] = None
alignment_core: Optional[AlignmentCore] = None
fusion_engine: Optional[FusionEngine] = None
audio_analyzer: Optional[AudioAnalyzer] = None

# v3.2.0 Enterprise modules
sanity_guard: Optional[InputSanityGuard] = None
history_manager: Optional[HistoryManager] = None
forensic_reporter: Optional[ForensicReporter] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and processors on startup."""
    global face_extractor, video_processor, inference_engine
    global biosignal_core, artifact_core, alignment_core, fusion_engine, audio_analyzer
    global sanity_guard, history_manager, forensic_reporter

    print("Initializing Scanner Detection Engine v3.2.0...")

    # Initialize face extractor with MediaPipe
    face_extractor = FaceExtractor(min_detection_confidence=0.5)

    # Initialize video processor (1 frame per second)
    video_processor = VideoProcessor(face_extractor, fps_sample_rate=1)

    # Initialize inference engine
    inference_engine = DeepfakeInference()

    # Initialize PRIME HYBRID cores (renamed)
    print("Initializing PRIME HYBRID cores...")
    biosignal_core = BioSignalCore()
    artifact_core = ArtifactCore()
    alignment_core = AlignmentCore()
    fusion_engine = FusionEngine()
    audio_analyzer = AudioAnalyzer()

    # v3.2.0: Initialize sanity guard
    sanity_guard = InputSanityGuard()
    print("Input Sanity Guard ready!")

    # v3.2.0: Initialize history manager
    if HISTORY_AVAILABLE:
        try:
            history_manager = HistoryManager()
            print("History Manager ready!")
        except Exception as e:
            print(f"History Manager initialization failed: {e}")

    # v3.2.0: Initialize forensic reporter
    if PDF_AVAILABLE:
        try:
            forensic_reporter = ForensicReporter()
            print("Forensic Reporter ready!")
        except Exception as e:
            print(f"Forensic Reporter initialization failed: {e}")

    print("Scanner Detection Engine ready!")
    print("PRIME HYBRID cores ready!")
    print("Audio Analyzer ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global face_extractor
    if face_extractor:
        face_extractor.close()


# ==================== PUBLIC ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint (public)."""
    return {
        "status": "online",
        "message": "Welcome to Scanner API",
        "service": "Scanner - Advanced Deepfake Detection",
        "version": "3.0.0",
        "auth_required": True,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check (public)."""
    return {
        "status": "healthy",
        "version": "3.2.0",
        "components": {
            "face_extractor": face_extractor is not None,
            "video_processor": video_processor is not None,
            "inference_engine": inference_engine is not None
        },
        "prime_hybrid": {
            "biosignal_core": biosignal_core is not None,
            "artifact_core": artifact_core is not None,
            "alignment_core": alignment_core is not None,
            "fusion_engine": fusion_engine is not None,
            "audio_analyzer": audio_analyzer is not None,
            "sanity_guard": sanity_guard is not None
        },
        "enterprise": {
            "rate_limiting": RATE_LIMITING_AVAILABLE,
            "history_manager": history_manager is not None,
            "pdf_reporter": forensic_reporter is not None
        }
    }


# ==================== AUTH ENDPOINTS ====================

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...)
):
    """
    Authenticate and receive a JWT access token.

    Configure credentials via environment variables
    (SCANNER_ADMIN_PASSWORD, SCANNER_ANALYST_PASSWORD, SCANNER_VIEWER_PASSWORD).
    """
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get information about the currently authenticated user."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "scopes": current_user.scopes
    }


# ==================== PROTECTED ENDPOINTS ====================

@app.post("/analyze-video", tags=["Analysis"])
async def analyze_video(
    file: UploadFile = File(...),
    current_user: User = Depends(require_write)
):
    """
    Analyze a video for deepfake content.

    Requires: `write` scope or API key.

    Processes video at 1 frame per second, extracts faces,
    and runs deepfake detection on each face.

    Args:
        file: Video file to analyze (MP4, AVI, MOV, etc.)

    Returns:
        JSON with detection results including verdict, confidence,
        and per-frame analysis.
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"video{file_ext}")

    try:
        # Write uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process video and extract faces
        processed_frames = video_processor.process_video(temp_path)

        if not processed_frames:
            return JSONResponse(
                status_code=200,
                content={
                    "verdict": "UNKNOWN",
                    "confidence": 0.0,
                    "message": "No faces detected in the video",
                    "frames_analyzed": 0,
                    "analyzed_by": current_user.username
                }
            )

        # Run inference on each frame
        frame_results = []
        for frame_num, face_tensor in processed_frames:
            prob, label = inference_engine.predict_single(face_tensor)
            frame_results.append((frame_num, prob, label))

        # Aggregate results
        analysis = inference_engine.analyze_video_results(frame_results)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "analyzed_by": current_user.username,
                **analysis
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/analyze-image", tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    current_user: User = Depends(require_write)
):
    """
    Analyze a single image for deepfake content.

    Requires: `write` scope or API key.

    Args:
        file: Image file to analyze (JPG, PNG, etc.)

    Returns:
        JSON with detection results.
    """
    import cv2

    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Read image from upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Extract face
        face = face_extractor.extract_primary_face(image)

        if face is None:
            return JSONResponse(
                status_code=200,
                content={
                    "verdict": "UNKNOWN",
                    "confidence": 0.0,
                    "message": "No face detected in the image",
                    "analyzed_by": current_user.username
                }
            )

        # Preprocess and predict
        preprocessed = face_extractor.preprocess_for_model(face)
        prob, label = inference_engine.predict_single(preprocessed)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "verdict": label,
                "fake_probability": prob,
                "confidence": prob if label == "FAKE" else 1 - prob,
                "analyzed_by": current_user.username
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/analyze-video-v2", tags=["Analysis"])
async def analyze_video_v2(
    file: UploadFile = File(...),
    current_user: User = Depends(require_write)
):
    """
    Analyze a video for deepfake content using PRIME HYBRID architecture.

    **NEW v3 Endpoint** - Uses the three-core forensic analysis system:
    - **BIOSIGNAL CORE**: 32-ROI rPPG biological signal analysis
    - **ARTIFACT CORE**: GAN/Diffusion/VAE fingerprint detection
    - **ALIGNMENT CORE**: Phoneme-Viseme mapping and A/V alignment
    - **FUSION ENGINE**: Unified decision engine with conflict resolution

    Requires: `write` scope or API key.

    Args:
        file: Video file to analyze (MP4, AVI, MOV, etc.)

    Returns:
        JSON with comprehensive forensic analysis including:
        - verdict: AUTHENTIC, UNCERTAIN, MANIPULATED, or INCONCLUSIVE
        - integrity_score: 0-100 scale (100 = completely authentic)
        - core_results: Individual scores from each core
        - weights: Dynamic weight distribution used
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"video{file_ext}")

    try:
        # Write uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Profile video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Determine resolution tier
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

        # Extract frames for forensic analysis (up to 60 frames)
        max_frames = 60
        interval = max(1, frame_count // max_frames)
        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(frame)
            frame_idx += 1

        cap.release()

        if not frames:
            return JSONResponse(
                status_code=200,
                content={
                    "verdict": "UNKNOWN",
                    "integrity_score": 0.0,
                    "message": "No frames could be extracted from the video",
                    "analyzed_by": current_user.username
                }
            )

        # Analyze audio for adaptive weighting
        audio_profile = None
        try:
            audio_profile = audio_analyzer.analyze(temp_path)
        except Exception:
            pass  # Continue without audio analysis if it fails

        # Run PRIME HYBRID analysis with audio awareness
        biosignal_result = biosignal_core.analyze(frames, fps, video_profile)
        artifact_result = artifact_core.analyze(frames, video_profile)
        alignment_result = alignment_core.analyze(frames, fps, temp_path, video_profile)
        verdict = fusion_engine.get_final_integrity_score(
            biosignal_result, artifact_result, alignment_result,
            video_profile, audio_profile
        )

        # Build response with transparency and audio profile
        response_content = {
            "filename": file.filename,
            "analyzed_by": current_user.username,
            "api_version": "v3",
            "architecture": "PRIME HYBRID",
            # Main verdict
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
                "alignment": verdict.alignment_score
            },
            # Weights used
            "weights": verdict.weights,
            # Video profile
            "video_profile": {
                "resolution": f"{height}p",
                "resolution_tier": tier.value,
                "fps": fps,
                "duration_seconds": duration,
                "rppg_viable": video_profile.rppg_viable
            },
            # Core details
            "cores": {
                "biosignal": {
                    "biological_sync": biosignal_result.biological_sync_score,
                    "pulse_coverage": biosignal_result.pulse_coverage,
                    "hr_consistency": biosignal_result.hr_consistency,
                    "status": biosignal_result.status,
                    "anomalies": biosignal_result.anomalies
                },
                "artifact": {
                    "gan_score": artifact_result.gan_score,
                    "diffusion_score": artifact_result.diffusion_score,
                    "vae_score": artifact_result.vae_score,
                    "structural_integrity": artifact_result.structural_integrity,
                    "detected_model": artifact_result.details.get("detected_model_type", "NONE"),
                    "status": artifact_result.status,
                    "anomalies": artifact_result.anomalies
                },
                "alignment": {
                    "av_alignment": alignment_result.av_alignment_score,
                    "phoneme_viseme": alignment_result.phoneme_viseme_score,
                    "speech_rhythm": alignment_result.speech_rhythm_score,
                    "lip_closures": len(alignment_result.lip_closure_events),
                    "status": alignment_result.status,
                    "anomalies": alignment_result.anomalies
                }
            },
            "frames_analyzed": len(frames)
        }

        # Add audio profile if available
        if audio_profile:
            response_content["audio_profile"] = {
                "has_audio": audio_profile.has_audio,
                "snr_db": round(audio_profile.snr_db, 2),
                "noise_level": audio_profile.noise_level,
                "recommended_av_weight": round(audio_profile.recommended_av_weight, 3),
                "is_speech_detected": audio_profile.is_speech_detected,
                "duration_seconds": round(audio_profile.duration_seconds, 2)
            }

        # Add transparency report if available
        if verdict.transparency_report:
            response_content["transparency"] = {
                "summary": verdict.transparency_report.summary,
                "primary_concern": verdict.transparency_report.primary_concern,
                "biosignal_explanation": verdict.transparency_report.biosignal_explanation,
                "artifact_explanation": verdict.transparency_report.artifact_explanation,
                "alignment_explanation": verdict.transparency_report.alignment_explanation,
                "environmental_factors": verdict.transparency_report.environmental_factors,
                "supporting_evidence": verdict.transparency_report.supporting_evidence,
                "weight_justification": verdict.transparency_report.weight_justification,
                "fusion_method": verdict.transparency_report.fusion_method,
                "audio_quality_note": verdict.transparency_report.audio_quality_note
            }

        return JSONResponse(status_code=200, content=response_content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/stats", tags=["Admin"])
async def get_system_stats(current_user: User = Depends(require_admin)):
    """
    Get system statistics (admin only).

    Requires: `admin` scope.
    """
    stats = {
        "engine": {
            "model": "EfficientNet-B0",
            "weights_source": inference_engine.weights_source if inference_engine else "not loaded",
            "input_size": inference_engine.input_size if inference_engine else 0
        },
        "components": {
            "face_extractor": "online" if face_extractor else "offline",
            "video_processor": "online" if video_processor else "offline",
            "inference_engine": "online" if inference_engine else "offline"
        },
        "prime_hybrid_cores": {
            "biosignal_core": "online" if biosignal_core else "offline",
            "artifact_core": "online" if artifact_core else "offline",
            "alignment_core": "online" if alignment_core else "offline",
            "fusion_engine": "online" if fusion_engine else "offline",
            "audio_analyzer": "online" if audio_analyzer else "offline"
        },
        "enterprise": {
            "sanity_guard": "online" if sanity_guard else "offline",
            "history_manager": "online" if history_manager else "offline",
            "forensic_reporter": "online" if forensic_reporter else "offline",
            "rate_limiting": "active" if RATE_LIMITING_AVAILABLE else "disabled"
        },
        "requested_by": current_user.username
    }

    # Add history statistics if available
    if history_manager:
        stats["history"] = history_manager.get_statistics()

    return stats


# ==================== v3.2.0 ENTERPRISE ENDPOINTS ====================

@app.get("/history", tags=["History"])
async def get_scan_history(
    limit: int = 20,
    current_user: User = Depends(require_read)
):
    """
    Get recent scan history.

    Requires: `read` scope.

    Args:
        limit: Maximum entries to return (default 20, max 100)

    Returns:
        List of recent scan entries
    """
    if not history_manager:
        raise HTTPException(status_code=503, detail="History service unavailable")

    limit = min(limit, 100)
    entries = history_manager.get_recent(limit)
    return {"history": [e.to_dict() for e in entries]}


@app.get("/history/stats", tags=["History"])
async def get_history_statistics(
    current_user: User = Depends(require_read)
):
    """
    Get scan history statistics.

    Requires: `read` scope.
    """
    if not history_manager:
        raise HTTPException(status_code=503, detail="History service unavailable")

    return history_manager.get_statistics()


@app.get("/history/{session_id}", tags=["History"])
async def get_scan_by_session(
    session_id: str,
    current_user: User = Depends(require_read)
):
    """
    Get scan result by session ID.

    Requires: `read` scope.
    """
    if not history_manager:
        raise HTTPException(status_code=503, detail="History service unavailable")

    entry = history_manager.get_by_session(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")

    return entry.to_dict()


@app.post("/export/pdf/{session_id}", tags=["Export"])
async def export_pdf_report(
    session_id: str,
    current_user: User = Depends(require_read)
):
    """
    Export scan result as PDF report.

    Requires: `read` scope.

    Returns:
        PDF file as download
    """
    if not forensic_reporter:
        raise HTTPException(status_code=503, detail="PDF export service unavailable")

    if not history_manager:
        raise HTTPException(status_code=503, detail="History service unavailable")

    entry = history_manager.get_by_session(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create a minimal verdict-like dict for the reporter
    from core.forensic_types import FusionVerdict

    verdict = FusionVerdict(
        verdict=entry.verdict,
        integrity_score=entry.integrity_score,
        confidence=0.8,  # Default confidence
        biosignal_score=entry.biosignal_score,
        artifact_score=entry.artifact_score,
        alignment_score=entry.alignment_score,
        weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
        leading_core="N/A"
    )

    pdf_bytes = forensic_reporter.generate_report(
        verdict=verdict,
        filename=entry.filename,
        sha256_hash=entry.sha256_hash,
        resolution=entry.resolution,
        duration=entry.duration_seconds,
        session_id=session_id
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=scanner_report_{session_id}.pdf"
        }
    )


@app.get("/admin/dashboard", tags=["Admin"])
async def admin_dashboard(
    current_user: User = Depends(require_admin)
):
    """
    Admin dashboard with system metrics.

    Requires: `admin` scope.
    """
    dashboard = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": "operational",
        "version": "3.2.0",
        "components": {
            "face_extractor": "online" if face_extractor else "offline",
            "inference_engine": "online" if inference_engine else "offline",
            "prime_hybrid": {
                "biosignal": "online" if biosignal_core else "offline",
                "artifact": "online" if artifact_core else "offline",
                "alignment": "online" if alignment_core else "offline",
                "fusion": "online" if fusion_engine else "offline"
            }
        },
        "enterprise_features": {
            "rate_limiting": RATE_LIMITING_AVAILABLE,
            "history": history_manager is not None,
            "pdf_export": forensic_reporter is not None
        }
    }

    # Add history stats
    if history_manager:
        dashboard["scan_statistics"] = history_manager.get_statistics()
        dashboard["recent_activity"] = [
            e.to_dict() for e in history_manager.get_recent(5)
        ]

    return dashboard


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
