"""
Scanner Prime - Thread-Safe Singleton ModelManager
Centralised lifecycle management for all heavy ML models and forensic cores.

Features:
- Thread-safe singleton (double-checked locking)
- Lazy initialisation of each core on first access
- Automatic GPU/CPU detection with graceful fallback
- Clear error reporting if model loading fails
- Single point of truth for all model instances

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import threading
from typing import Optional

from core.logging_config import get_logger
from core.exceptions import ModelLoadError

logger = get_logger("model_manager")


class ModelManager:
    """
    Thread-safe singleton that owns every heavy model instance.

    Usage:
        mm = ModelManager.get_instance()
        result = mm.biosignal_core.analyze(frames, fps, profile)
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    # --- Singleton access ---

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Return the global ModelManager, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing). Not thread-safe by design."""
        cls._instance = None

    # --- Initialisation ---

    def __init__(self) -> None:
        # Guard against direct construction
        self._initialised = False
        self._init_lock = threading.Lock()

        # Core instances (lazily loaded)
        self._biosignal_core = None
        self._artifact_core = None
        self._alignment_core = None
        self._fusion_engine = None
        self._audio_analyzer = None
        self._sanity_guard = None
        self._face_extractor = None
        self._video_processor = None
        self._inference_engine = None

        # Device detection
        self._device: Optional[str] = None

    # --- Device detection ---

    @property
    def device(self) -> str:
        """Detect and cache available compute device."""
        if self._device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("GPU detected, using CUDA", extra={"stage": "device_detection"})
                else:
                    self._device = "cpu"
                    logger.info("No GPU available, using CPU", extra={"stage": "device_detection"})
            except ImportError:
                self._device = "cpu"
                logger.warning("PyTorch not available, defaulting to CPU")
        return self._device

    # --- Lazy-loaded core accessors ---

    @property
    def biosignal_core(self):
        if self._biosignal_core is None:
            with self._init_lock:
                if self._biosignal_core is None:
                    try:
                        from core.biosignal_core import BioSignalCore
                        self._biosignal_core = BioSignalCore()
                        logger.info("BioSignalCore initialised")
                    except Exception as exc:
                        raise ModelLoadError("BioSignalCore", str(exc)) from exc
        return self._biosignal_core

    @property
    def artifact_core(self):
        if self._artifact_core is None:
            with self._init_lock:
                if self._artifact_core is None:
                    try:
                        from core.artifact_core import ArtifactCore
                        self._artifact_core = ArtifactCore()
                        logger.info("ArtifactCore initialised")
                    except Exception as exc:
                        raise ModelLoadError("ArtifactCore", str(exc)) from exc
        return self._artifact_core

    @property
    def alignment_core(self):
        if self._alignment_core is None:
            with self._init_lock:
                if self._alignment_core is None:
                    try:
                        from core.alignment_core import AlignmentCore
                        self._alignment_core = AlignmentCore()
                        logger.info("AlignmentCore initialised")
                    except Exception as exc:
                        raise ModelLoadError("AlignmentCore", str(exc)) from exc
        return self._alignment_core

    @property
    def fusion_engine(self):
        if self._fusion_engine is None:
            with self._init_lock:
                if self._fusion_engine is None:
                    try:
                        from core.fusion_engine import FusionEngine
                        self._fusion_engine = FusionEngine()
                        logger.info("FusionEngine initialised")
                    except Exception as exc:
                        raise ModelLoadError("FusionEngine", str(exc)) from exc
        return self._fusion_engine

    @property
    def audio_analyzer(self):
        if self._audio_analyzer is None:
            with self._init_lock:
                if self._audio_analyzer is None:
                    try:
                        from core.audio_analyzer import AudioAnalyzer
                        self._audio_analyzer = AudioAnalyzer()
                        logger.info("AudioAnalyzer initialised")
                    except Exception as exc:
                        raise ModelLoadError("AudioAnalyzer", str(exc)) from exc
        return self._audio_analyzer

    @property
    def sanity_guard(self):
        if self._sanity_guard is None:
            with self._init_lock:
                if self._sanity_guard is None:
                    try:
                        from core.input_sanity_guard import InputSanityGuard
                        self._sanity_guard = InputSanityGuard()
                        logger.info("InputSanityGuard initialised")
                    except Exception as exc:
                        raise ModelLoadError("InputSanityGuard", str(exc)) from exc
        return self._sanity_guard

    @property
    def face_extractor(self):
        if self._face_extractor is None:
            with self._init_lock:
                if self._face_extractor is None:
                    try:
                        from preprocessing import FaceExtractor
                        self._face_extractor = FaceExtractor(min_detection_confidence=0.5)
                        logger.info("FaceExtractor initialised")
                    except Exception as exc:
                        raise ModelLoadError("FaceExtractor", str(exc)) from exc
        return self._face_extractor

    @property
    def video_processor(self):
        if self._video_processor is None:
            with self._init_lock:
                if self._video_processor is None:
                    try:
                        from preprocessing import VideoProcessor
                        self._video_processor = VideoProcessor(
                            self.face_extractor, fps_sample_rate=1
                        )
                        logger.info("VideoProcessor initialised")
                    except Exception as exc:
                        raise ModelLoadError("VideoProcessor", str(exc)) from exc
        return self._video_processor

    @property
    def inference_engine(self):
        if self._inference_engine is None:
            with self._init_lock:
                if self._inference_engine is None:
                    try:
                        from model import DeepfakeInference
                        self._inference_engine = DeepfakeInference()
                        logger.info("DeepfakeInference initialised")
                    except Exception as exc:
                        raise ModelLoadError("DeepfakeInference", str(exc)) from exc
        return self._inference_engine

    # --- Bulk operations ---

    def initialise_all(self) -> None:
        """Eagerly load every core. Called at application startup."""
        logger.info("Initialising all PRIME HYBRID cores...")
        _ = self.device
        _ = self.face_extractor
        _ = self.video_processor
        _ = self.inference_engine
        _ = self.biosignal_core
        _ = self.artifact_core
        _ = self.alignment_core
        _ = self.fusion_engine
        _ = self.audio_analyzer
        _ = self.sanity_guard
        self._initialised = True
        logger.info("All PRIME HYBRID cores initialised successfully")

    @property
    def is_ready(self) -> bool:
        return self._initialised

    def health_status(self) -> dict:
        """Return component health for /health endpoint."""
        return {
            "device": self.device,
            "face_extractor": self._face_extractor is not None,
            "video_processor": self._video_processor is not None,
            "inference_engine": self._inference_engine is not None,
            "biosignal_core": self._biosignal_core is not None,
            "artifact_core": self._artifact_core is not None,
            "alignment_core": self._alignment_core is not None,
            "fusion_engine": self._fusion_engine is not None,
            "audio_analyzer": self._audio_analyzer is not None,
            "sanity_guard": self._sanity_guard is not None,
        }

    def shutdown(self) -> None:
        """Release resources."""
        if self._face_extractor is not None:
            try:
                self._face_extractor.close()
            except Exception:
                pass
        logger.info("ModelManager shutdown complete")
