"""
Scanner Test Suite - Custom Exceptions Tests
Unit tests for core/exceptions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exceptions import (
    AnalysisTimeoutError,
    GPUMemoryError,
    IntegrityVerificationError,
    ModelLoadError,
    NoFaceDetectedError,
    NoFramesExtractedError,
    ProcessingError,
    ScannerBaseError,
    UnsupportedFileTypeError,
    VideoDecodeError,
    VideoUploadError,
    WorkerError,
)


class TestScannerBaseError:
    """Tests for the base exception class."""

    def test_base_error_creation(self):
        exc = ScannerBaseError("test error")
        assert exc.message == "test error"
        assert exc.error_code == "SCANNER_ERROR"
        assert exc.http_status == 500
        assert exc.details == {}

    def test_to_dict(self):
        exc = ScannerBaseError("test", error_code="TEST", details={"key": "val"})
        d = exc.to_dict()
        assert d["error"] is True
        assert d["error_code"] == "TEST"
        assert d["message"] == "test"
        assert d["details"]["key"] == "val"


class TestVideoErrors:
    """Tests for video-related exceptions."""

    def test_video_upload_error(self):
        exc = VideoUploadError("bad upload")
        assert exc.http_status == 400
        assert exc.error_code == "VIDEO_UPLOAD_ERROR"

    def test_unsupported_file_type(self):
        exc = UnsupportedFileTypeError(".exe", [".mp4", ".avi"])
        assert ".exe" in exc.message
        assert exc.http_status == 400
        assert exc.details["extension"] == ".exe"

    def test_video_decode_error(self):
        exc = VideoDecodeError()
        assert exc.http_status == 400
        assert "decode" in exc.message.lower()

    def test_no_frames_extracted(self):
        exc = NoFramesExtractedError()
        assert exc.http_status == 422

    def test_no_face_detected(self):
        exc = NoFaceDetectedError()
        assert exc.http_status == 422


class TestModelErrors:
    """Tests for model-related exceptions."""

    def test_model_load_error(self):
        exc = ModelLoadError("BioSignalCore", "file not found")
        assert "BioSignalCore" in exc.message
        assert exc.http_status == 503
        assert exc.details["model_name"] == "BioSignalCore"

    def test_gpu_memory_error(self):
        exc = GPUMemoryError()
        assert exc.http_status == 503


class TestProcessingErrors:
    """Tests for processing pipeline exceptions."""

    def test_processing_error(self):
        exc = ProcessingError("biosignal", "signal too weak")
        assert "biosignal" in exc.message
        assert exc.details["stage"] == "biosignal"

    def test_analysis_timeout(self):
        exc = AnalysisTimeoutError(300.0)
        assert "300" in exc.message
        assert exc.http_status == 504

    def test_worker_error(self):
        exc = WorkerError("task-123", "worker died")
        assert exc.details["task_id"] == "task-123"

    def test_integrity_verification_error(self):
        exc = IntegrityVerificationError("aabbcc", "ddeeff")
        assert exc.error_code == "INTEGRITY_ERROR"
        assert exc.http_status == 500
