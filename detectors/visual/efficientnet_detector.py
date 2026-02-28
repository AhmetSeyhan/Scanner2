"""
Scanner - EfficientNet Detector Adapter (v5.0.0)
Wraps the existing DeepfakeDetector (EfficientNet-B0) model as a BaseDetector.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


class EfficientNetDetector(BaseDetector):
    """Adapter that delegates to ``model.DeepfakeInference``."""

    def __init__(self) -> None:
        self._inference = None

    @property
    def name(self) -> str:
        return "EfficientNet-B0 Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.SINGLE_IMAGE}

    def _ensure_model(self) -> Any:
        if self._inference is None:
            from model import DeepfakeInference
            self._inference = DeepfakeInference()
        return self._inference

    def initialize(self) -> None:
        self._ensure_model()

    def health_check(self) -> Dict[str, Any]:
        base = super().health_check()
        if self._inference is not None:
            base["weights_source"] = getattr(
                self._inference.model, "_weights_source", "unknown"
            )
        return base

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.image is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No image provided"},
            )

        inference = self._ensure_model()

        # The model expects a preprocessed face (3, 224, 224) float32 array.
        # If we receive a raw BGR image, attempt lightweight preprocessing.
        image = inp.image
        if image.ndim == 3 and image.shape[0] != 3:
            import cv2
            resized = cv2.resize(image, (224, 224))
            image = resized.astype(np.float32).transpose(2, 0, 1) / 255.0

        prob, label = inference.predict_single(image)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=prob,
            confidence=min(abs(prob - 0.5) * 2, 1.0),
            status=DetectorStatus.FAIL if prob > 0.5 else DetectorStatus.PASS,
            details={"label": label, "probability": prob},
        )
