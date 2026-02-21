"""
Scanner - Detector Registry (v5.0.0)

Thread-safe registry for runtime detector discovery, health checks,
and capability-based querying.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from detectors.base import BaseDetector, DetectorCapability, DetectorType


class DetectorRegistry:
    """Singleton registry of all active detectors."""

    _instance: Optional[DetectorRegistry] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._detectors: Dict[str, BaseDetector] = {}

    # ---- singleton ----

    @classmethod
    def get_instance(cls) -> DetectorRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for tests)."""
        with cls._lock:
            cls._instance = None

    # ---- register / unregister ----

    def register(self, detector: BaseDetector) -> None:
        self._detectors[detector.name] = detector

    def unregister(self, name: str) -> None:
        self._detectors.pop(name, None)

    # ---- queries ----

    def get(self, name: str) -> Optional[BaseDetector]:
        return self._detectors.get(name)

    def get_by_type(self, dtype: DetectorType) -> List[BaseDetector]:
        return [d for d in self._detectors.values() if d.detector_type == dtype]

    def get_by_capability(self, cap: DetectorCapability) -> List[BaseDetector]:
        return [d for d in self._detectors.values() if cap in d.capabilities]

    def list_all(self) -> List[BaseDetector]:
        return list(self._detectors.values())

    def list_names(self) -> List[str]:
        return list(self._detectors.keys())

    def __len__(self) -> int:
        return len(self._detectors)

    # ---- health ----

    def health_check_all(self) -> Dict[str, dict]:
        return {d.name: d.health_check() for d in self._detectors.values()}

    # ---- bulk helpers ----

    def register_defaults(self) -> None:
        """Register the standard set of adapters wrapping existing cores.

        Each import is lazy so missing dependencies degrade gracefully.
        """
        adapters: list = []

        try:
            from detectors.visual.biosignal_detector import BioSignalDetector
            adapters.append(BioSignalDetector())
        except Exception:
            pass

        try:
            from detectors.visual.artifact_detector import ArtifactDetector
            adapters.append(ArtifactDetector())
        except Exception:
            pass

        try:
            from detectors.visual.alignment_detector import AlignmentDetector
            adapters.append(AlignmentDetector())
        except Exception:
            pass

        try:
            from detectors.visual.efficientnet_detector import EfficientNetDetector
            adapters.append(EfficientNetDetector())
        except Exception:
            pass

        try:
            from detectors.text.text_detector import TextDetector
            adapters.append(TextDetector())
        except Exception:
            pass

        for adapter in adapters:
            self.register(adapter)
