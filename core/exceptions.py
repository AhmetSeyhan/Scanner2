"""
Scanner Prime - Custom Exception Hierarchy
Enterprise-grade error handling for all subsystems.

Provides structured, typed exceptions for:
- Video upload and validation errors
- Model loading and inference errors
- GPU/resource errors
- Processing pipeline errors
- Authentication and authorization errors

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from typing import Optional, Dict, Any


class ScannerBaseError(Exception):
    """Base exception for all Scanner errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "SCANNER_ERROR",
        details: Optional[Dict[str, Any]] = None,
        http_status: int = 500,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.http_status = http_status
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict for API responses."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# --- Upload & Validation Errors (4xx) ---

class VideoUploadError(ScannerBaseError):
    """Raised when video upload fails validation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VIDEO_UPLOAD_ERROR",
            details=details,
            http_status=400,
        )


class UnsupportedFileTypeError(VideoUploadError):
    """Raised when uploaded file type is not supported."""

    def __init__(self, extension: str, allowed: list[str]):
        super().__init__(
            message=f"Unsupported file type '{extension}'. Allowed: {', '.join(allowed)}",
            details={"extension": extension, "allowed": allowed},
        )


class InputValidationError(ScannerBaseError):
    """Raised when input sanity checks fail."""

    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="INPUT_VALIDATION_ERROR",
            details={"reason": reason} if reason else {},
            http_status=422,
        )


# --- Model & Resource Errors (5xx) ---

class ModelLoadError(ScannerBaseError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_ERROR",
            details={"model_name": model_name, "reason": reason},
            http_status=503,
        )


class GPUMemoryError(ScannerBaseError):
    """Raised when GPU memory is exhausted."""

    def __init__(self, message: str = "GPU memory exhausted"):
        super().__init__(
            message=message,
            error_code="GPU_MEMORY_ERROR",
            details={},
            http_status=503,
        )


# --- Processing Errors (5xx) ---

class ProcessingError(ScannerBaseError):
    """Raised when analysis processing fails."""

    def __init__(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Processing error at '{stage}': {message}",
            error_code="PROCESSING_ERROR",
            details={"stage": stage, **(details or {})},
            http_status=500,
        )


class AnalysisTimeoutError(ScannerBaseError):
    """Raised when analysis exceeds time limit."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"Analysis timed out after {timeout_seconds:.0f}s",
            error_code="ANALYSIS_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
            http_status=504,
        )


class VideoDecodeError(ProcessingError):
    """Raised when a video file cannot be decoded."""

    def __init__(self, message: str = "Could not open or decode video file"):
        super().__init__(stage="video_decode", message=message)
        self.http_status = 400


class NoFramesExtractedError(ProcessingError):
    """Raised when no frames could be extracted from video."""

    def __init__(self):
        super().__init__(stage="frame_extraction", message="No frames could be extracted")
        self.http_status = 422


class NoFaceDetectedError(ProcessingError):
    """Raised when no face is detected in input."""

    def __init__(self):
        super().__init__(stage="face_detection", message="No face detected in input")
        self.http_status = 422


# --- Worker / Queue Errors ---

class WorkerError(ScannerBaseError):
    """Raised for Celery worker-level failures."""

    def __init__(self, task_id: str, message: str):
        super().__init__(
            message=message,
            error_code="WORKER_ERROR",
            details={"task_id": task_id},
            http_status=500,
        )


class IntegrityVerificationError(ScannerBaseError):
    """Raised when forensic hash verification fails."""

    def __init__(self, expected: str, actual: str):
        super().__init__(
            message="Integrity verification failed: hash mismatch",
            error_code="INTEGRITY_ERROR",
            details={"expected_hash": expected[:16] + "...", "actual_hash": actual[:16] + "..."},
            http_status=500,
        )
