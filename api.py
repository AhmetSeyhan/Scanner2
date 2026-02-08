"""
Scanner API - FastAPI Router Layer (v3.3.0 Enterprise)
Thin routing layer that validates input, delegates to services, and returns responses.

All business logic lives in the services/ directory.
Model lifecycle is managed by utils/model_manager.py.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import os
import time
import shutil
import tempfile
from datetime import timedelta, datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# --- Structured logging (replaces all print statements) ---
from core.logging_config import setup_logging, get_logger
from core.exceptions import ScannerBaseError, VideoUploadError

setup_logging()
logger = get_logger("api")

# --- Rate limiting (graceful degradation) ---
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

# --- Auth ---
from auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    require_read,
    require_write,
    require_admin,
    Token,
    User,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# --- Services ---
from services.analysis_service import AnalysisService
from services.video_profiler import VideoProfiler, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from services.report_service import ReportService
from services.history_service import HistoryService

# --- Monitoring ---
from utils.metrics import get_metrics_response, record_request, record_analysis, record_error
from utils.audit_logger import AuditLogger
from utils.model_manager import ModelManager


# ==================== APPLICATION LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    # --- Startup ---
    logger.info("Initialising Scanner Detection Engine v3.3.0...")

    mm = ModelManager.get_instance()
    try:
        mm.initialise_all()
    except Exception as exc:
        logger.error(f"Core initialisation failed: {exc}", exc_info=True)

    logger.info("Scanner Detection Engine ready")
    yield
    # --- Shutdown ---
    mm.shutdown()
    logger.info("Scanner shutdown complete")


# ==================== APP FACTORY ====================

def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    _app = FastAPI(
        title="Scanner API",
        description="Advanced deepfake detection engine powered by PRIME HYBRID. v3.3.0 Enterprise.",
        version="3.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiter
    if RATE_LIMITING_AVAILABLE:
        limiter = Limiter(key_func=get_remote_address)
        _app.state.limiter = limiter
        _app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("SCANNER_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app


app = create_app()

# --- Singletons ---
_analysis_service = AnalysisService()
_audit_logger = AuditLogger(redis_url=os.getenv("REDIS_URL"))


# ==================== GLOBAL EXCEPTION HANDLER ====================

@app.exception_handler(ScannerBaseError)
async def scanner_error_handler(request: Request, exc: ScannerBaseError):
    """Convert ScannerBaseError subtypes into structured JSON responses."""
    record_error(exc.error_code)
    logger.error(
        exc.message,
        extra={"error_code": exc.error_code},
        exc_info=False,
    )
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


# ==================== PUBLIC ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint (public)."""
    return {
        "status": "online",
        "service": "Scanner - Advanced Deepfake Detection",
        "version": "3.3.0",
        "auth_required": True,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check (public)."""
    mm = ModelManager.get_instance()
    return {
        "status": "healthy" if mm.is_ready else "degraded",
        "version": "3.3.0",
        "components": mm.health_status(),
        "enterprise": {
            "rate_limiting": RATE_LIMITING_AVAILABLE,
            "history": HistoryService.is_available(),
            "pdf_reporter": ReportService.is_available(),
        },
    }


@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    result = get_metrics_response()
    if result is None:
        raise HTTPException(status_code=501, detail="Prometheus client not installed")
    content, content_type = result
    return Response(content=content, media_type=content_type)


# ==================== AUTH ENDPOINTS ====================

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Authenticate and receive a JWT access token."""
    user = authenticate_user(username, password)
    if not user:
        _audit_logger.log_auth_event(
            user=username,
            ip_address=request.client.host if request.client else "unknown",
            event="login",
            success=False,
        )
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _audit_logger.log_auth_event(
        user=username,
        ip_address=request.client.host if request.client else "unknown",
        event="login",
        success=True,
    )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get information about the currently authenticated user."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "scopes": current_user.scopes,
    }


# ==================== ANALYSIS ENDPOINTS ====================

@app.post("/analyze-video", tags=["Analysis"])
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(require_write),
):
    """
    Analyze a video for deepfake content (legacy v1 endpoint).

    Requires: `write` scope or API key.
    """
    profiler = VideoProfiler()
    ext = profiler.validate_extension(file.filename or "", "video")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"video{ext}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = _analysis_service.analyze_video_v1(temp_path, file.filename or "", current_user.username)
        return JSONResponse(status_code=200, content=result)

    except ScannerBaseError:
        raise
    except Exception as exc:
        logger.error(f"Video analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {exc}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/analyze-image", tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    current_user: User = Depends(require_write),
):
    """
    Analyze a single image for deepfake content.

    Requires: `write` scope or API key.
    """
    profiler = VideoProfiler()
    profiler.validate_extension(file.filename or "", "image")

    try:
        contents = await file.read()
        result = _analysis_service.analyze_image(contents, file.filename or "", current_user.username)
        return JSONResponse(status_code=200, content=result)
    except ScannerBaseError:
        raise
    except Exception as exc:
        logger.error(f"Image analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


@app.post("/analyze-video-v2", tags=["Analysis"])
async def analyze_video_v2(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(require_write),
):
    """
    Analyze a video using PRIME HYBRID architecture.

    Returns comprehensive forensic analysis including:
    - verdict: AUTHENTIC, UNCERTAIN, MANIPULATED, or INCONCLUSIVE
    - integrity_score: 0-100 scale
    - core_results: Individual scores from each core
    - forensic_hashes: Chain-of-custody hash chain

    Requires: `write` scope or API key.
    """
    profiler = VideoProfiler()
    ext = profiler.validate_extension(file.filename or "", "video")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"video{ext}")
    start = time.time()

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Audit log
        _audit_logger.log_analysis_request(
            session_id="pending",
            user=current_user.username,
            ip_address=request.client.host if request.client else "unknown",
            filename=file.filename or "",
            video_hash="",
        )

        result = _analysis_service.analyze_video_v2(
            video_path=temp_path,
            filename=file.filename or "",
            user=current_user.username,
        )

        # Record in history
        HistoryService.add_entry_from_result(
            session_id=result.get("session_id", ""),
            filename=file.filename or "",
            result=result,
            sha256_hash=result.get("forensic_hashes", {}).get("original_file", ""),
            user=current_user.username,
        )

        # Audit result
        _audit_logger.log_analysis_result(
            session_id=result.get("session_id", ""),
            user=current_user.username,
            verdict=result.get("verdict", "UNKNOWN"),
            integrity_score=result.get("integrity_score", 0.0),
            confidence=result.get("confidence", 0.0),
            leading_core=result.get("leading_core", ""),
            duration_ms=result.get("duration_ms", 0.0),
            filename=file.filename or "",
            video_hash=result.get("forensic_hashes", {}).get("original_file", ""),
        )

        # Prometheus metrics
        record_analysis(
            "prime_hybrid_v2",
            result.get("verdict", "UNKNOWN"),
            (time.time() - start),
        )

        return JSONResponse(status_code=200, content=result)

    except ScannerBaseError:
        raise
    except Exception as exc:
        logger.error(f"PRIME HYBRID analysis failed: {exc}", exc_info=True)
        record_error("PROCESSING_ERROR")
        raise HTTPException(status_code=500, detail=f"Error processing video: {exc}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== STATS & ADMIN ENDPOINTS ====================

@app.get("/stats", tags=["Admin"])
async def get_system_stats(current_user: User = Depends(require_admin)):
    """System statistics (admin only)."""
    mm = ModelManager.get_instance()
    stats = {
        "engine": {
            "model": "EfficientNet-B0",
            "device": mm.device,
        },
        "components": mm.health_status(),
        "enterprise": {
            "rate_limiting": "active" if RATE_LIMITING_AVAILABLE else "disabled",
            "history": "online" if HistoryService.is_available() else "offline",
            "pdf_reporter": "online" if ReportService.is_available() else "offline",
        },
        "requested_by": current_user.username,
    }

    if HistoryService.is_available():
        stats["history"] = HistoryService.get_statistics()

    return stats


# ==================== HISTORY ENDPOINTS ====================

@app.get("/history", tags=["History"])
async def get_scan_history(
    limit: int = 20,
    current_user: User = Depends(require_read),
):
    """Get recent scan history. Requires: `read` scope."""
    if not HistoryService.is_available():
        raise HTTPException(status_code=503, detail="History service unavailable")
    entries = HistoryService.get_recent(limit)
    return {"history": entries}


@app.get("/history/stats", tags=["History"])
async def get_history_statistics(current_user: User = Depends(require_read)):
    """Scan history statistics. Requires: `read` scope."""
    if not HistoryService.is_available():
        raise HTTPException(status_code=503, detail="History service unavailable")
    return HistoryService.get_statistics()


@app.get("/history/{session_id}", tags=["History"])
async def get_scan_by_session(
    session_id: str,
    current_user: User = Depends(require_read),
):
    """Get scan result by session ID. Requires: `read` scope."""
    if not HistoryService.is_available():
        raise HTTPException(status_code=503, detail="History service unavailable")
    entry = HistoryService.get_by_session(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")
    return entry.to_dict()


# ==================== EXPORT ENDPOINTS ====================

@app.post("/export/pdf/{session_id}", tags=["Export"])
async def export_pdf_report(
    session_id: str,
    current_user: User = Depends(require_read),
):
    """Export scan result as PDF report. Requires: `read` scope."""
    if not ReportService.is_available():
        raise HTTPException(status_code=503, detail="PDF export service unavailable")
    if not HistoryService.is_available():
        raise HTTPException(status_code=503, detail="History service unavailable")

    entry = HistoryService.get_by_session(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")

    pdf_bytes = ReportService.generate_pdf_from_history_entry(entry, session_id)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=scanner_report_{session_id}.pdf"
        },
    )


# ==================== ADMIN DASHBOARD ====================

@app.get("/admin/dashboard", tags=["Admin"])
async def admin_dashboard(current_user: User = Depends(require_admin)):
    """Admin dashboard with system metrics. Requires: `admin` scope."""
    mm = ModelManager.get_instance()
    dashboard = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_status": "operational" if mm.is_ready else "degraded",
        "version": "3.3.0",
        "components": mm.health_status(),
        "enterprise_features": {
            "rate_limiting": RATE_LIMITING_AVAILABLE,
            "history": HistoryService.is_available(),
            "pdf_export": ReportService.is_available(),
        },
    }

    if HistoryService.is_available():
        dashboard["scan_statistics"] = HistoryService.get_statistics()
        recent = HistoryService.get_recent(5)
        dashboard["recent_activity"] = recent

    return dashboard


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
