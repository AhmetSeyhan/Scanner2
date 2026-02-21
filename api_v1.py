"""
Scanner API v1 - Versioned Router (v5.0.0)

Unified, detector-framework-based API surface.
Mounted at /v1/ alongside the legacy routes.

Endpoints:
  POST /v1/scan         - Unified scan (auto-detects input type)
  GET  /v1/results/{id} - Retrieve scan result
  GET  /v1/detectors    - List registered detectors
  GET  /v1/health       - Versioned health check

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from core.logging_config import get_logger
from auth import get_current_active_user, require_write, require_read, User
from detectors.base import DetectorInput, DetectorStatus
from detectors.registry import DetectorRegistry
from services.history_service import HistoryService

logger = get_logger("api_v1")

router = APIRouter(prefix="/v1", tags=["v1"])


def _get_registry() -> DetectorRegistry:
    """Return the global DetectorRegistry, registering defaults on first call."""
    reg = DetectorRegistry.get_instance()
    if len(reg) == 0:
        reg.register_defaults()
    return reg


# ==================== HEALTH & DISCOVERY ============================

@router.get("/health")
async def v1_health():
    """Versioned health check with detector status."""
    reg = _get_registry()
    return {
        "status": "online",
        "api_version": "v1",
        "scanner_version": "5.0.0",
        "detectors": reg.health_check_all(),
        "detector_count": len(reg),
    }


@router.get("/detectors")
async def list_detectors(current_user: User = Depends(require_read)):
    """List all registered detectors and their capabilities."""
    reg = _get_registry()
    detectors = []
    for d in reg.list_all():
        detectors.append({
            "name": d.name,
            "type": d.detector_type.value,
            "capabilities": [c.value for c in d.capabilities],
            "version": d.version,
            "enabled": d.enabled,
        })
    return {"detectors": detectors, "total": len(detectors)}


# ==================== UNIFIED SCAN ==================================

@router.post("/scan")
async def unified_scan(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    current_user: User = Depends(require_write),
):
    """Unified scan endpoint.

    Accepts either:
    - ``file``: a video or image upload
    - ``text``: raw text for AI-text detection

    The router auto-detects the content type and dispatches to the
    appropriate set of detectors.
    """
    session_id = uuid.uuid4().hex[:12]
    start = time.time()
    reg = _get_registry()
    results = []

    # --- Text scan ---
    if text:
        from detectors.base import DetectorCapability
        inp = DetectorInput(text=text)
        for det in reg.get_by_capability(DetectorCapability.TEXT_CONTENT):
            if det.enabled:
                results.append(det.detect(inp))

        return _build_response(session_id, results, start, "text", current_user)

    # --- File scan ---
    if file is None:
        raise HTTPException(status_code=422, detail="Provide either 'file' or 'text'")

    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

    if ext in image_exts:
        import cv2
        import numpy as np
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        inp = DetectorInput(image=image, metadata={"filename": filename})
        from detectors.base import DetectorCapability
        for det in reg.get_by_capability(DetectorCapability.SINGLE_IMAGE):
            if det.enabled:
                results.append(det.detect(inp))
        return _build_response(session_id, results, start, "image", current_user)

    if ext in video_exts:
        import cv2
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"video{ext}")
        try:
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)

            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = []
            max_frames = 120
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                raise HTTPException(status_code=400, detail="Could not read video frames")

            inp = DetectorInput(
                frames=frames,
                fps=fps,
                video_path=temp_path,
                metadata={"filename": filename},
            )
            from detectors.base import DetectorCapability
            for det in reg.get_by_capability(DetectorCapability.VIDEO_FRAMES):
                if det.enabled:
                    results.append(det.detect(inp))
            return _build_response(session_id, results, start, "video", current_user)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    raise HTTPException(status_code=415, detail=f"Unsupported file extension: {ext}")


# ==================== RESULTS =======================================

@router.get("/results/{session_id}")
async def get_result(session_id: str, current_user: User = Depends(require_read)):
    """Retrieve a previous scan result by session ID."""
    if not HistoryService.is_available():
        raise HTTPException(status_code=503, detail="History service unavailable")
    entry = HistoryService.get_by_session(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")
    return entry.to_dict()


# ==================== HELPERS =======================================

def _build_response(
    session_id: str,
    results: list,
    start: float,
    input_type: str,
    user: User,
) -> JSONResponse:
    """Aggregate detector results into a unified response."""
    duration_ms = (time.time() - start) * 1000

    if not results:
        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_id,
                "input_type": input_type,
                "detectors_run": 0,
                "aggregate_score": None,
                "verdict": "NO_DETECTORS",
                "duration_ms": round(duration_ms, 2),
            },
        )

    scores = [r.score for r in results if r.status not in (DetectorStatus.SKIPPED, DetectorStatus.ERROR)]
    confidences = [r.confidence for r in results if r.status not in (DetectorStatus.SKIPPED, DetectorStatus.ERROR)]

    if scores:
        # Confidence-weighted average
        total_w = sum(confidences) or 1.0
        agg = sum(s * c for s, c in zip(scores, confidences)) / total_w
    else:
        agg = 0.5

    if agg <= 0.30:
        verdict = "AUTHENTIC"
    elif agg <= 0.50:
        verdict = "UNCERTAIN"
    elif agg <= 0.65:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "MANIPULATED"

    return JSONResponse(
        status_code=200,
        content={
            "session_id": session_id,
            "api_version": "v1",
            "input_type": input_type,
            "verdict": verdict,
            "aggregate_score": round(agg, 4),
            "detectors_run": len(results),
            "results": [r.to_dict() for r in results],
            "duration_ms": round(duration_ms, 2),
            "analyzed_by": user.username,
        },
    )
