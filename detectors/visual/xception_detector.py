"""
Scanner - Xception Detector Adapter (v5.1.0)
Wraps an Xception model with FaceForensics++ weights as a BaseDetector.

The Xception architecture (Chollet, 2017) uses depthwise separable convolutions
and has been widely adopted for deepfake detection due to its strong performance
on FaceForensics++ benchmarks.

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


class XceptionDetector(BaseDetector):
    """Adapter for Xception model with FaceForensics++ pre-trained weights.

    Uses timm library to load the Xception architecture and applies
    custom deepfake detection weights from ``weights/xception_best.pth``.
    Input size: 299x299 (standard Xception input).
    """

    WEIGHT_PATHS = [
        "weights/xception_best.pth",
        "deepfake_detector/weights/xception_best.pth",
    ]
    INPUT_SIZE = (299, 299)

    def __init__(self) -> None:
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return "Xception Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.VISUAL

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.SINGLE_IMAGE}

    @property
    def version(self) -> str:
        return "1.0.0"

    def _find_weights(self) -> str | None:
        import os
        for p in self.WEIGHT_PATHS:
            if os.path.isfile(p):
                return p
        return None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import timm
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Xception model with 1 output (binary classification)
        self._model = timm.create_model("xception", pretrained=False, num_classes=1)

        # Try loading deepfake-specific weights
        weights_path = self._find_weights()
        self._weights_source = "random"
        if weights_path:
            try:
                checkpoint = torch.load(weights_path, map_location=self._device, weights_only=False)
                state_dict = checkpoint
                if isinstance(checkpoint, dict):
                    for key in ("state_dict", "model", "model_state_dict", "net"):
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break

                # Try strict load first
                try:
                    self._model.load_state_dict(state_dict, strict=True)
                    self._weights_source = "xception_best.pth (strict)"
                except RuntimeError:
                    # Partial load with matching keys
                    model_dict = self._model.state_dict()
                    filtered = {k: v for k, v in state_dict.items()
                                if k in model_dict and v.shape == model_dict[k].shape}
                    if filtered:
                        model_dict.update(filtered)
                        self._model.load_state_dict(model_dict)
                        self._weights_source = f"xception_best.pth (partial: {len(filtered)}/{len(model_dict)} keys)"
            except Exception:
                self._weights_source = "random (load failed)"

        self._model.to(self._device)
        self._model.eval()

    def initialize(self) -> None:
        self._ensure_model()

    def health_check(self) -> Dict[str, Any]:
        base = super().health_check()
        base["weights_source"] = getattr(self, "_weights_source", "not loaded")
        base["input_size"] = f"{self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}"
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

        import cv2
        import torch

        self._ensure_model()

        image = inp.image
        # Preprocess: resize to 299x299, normalize
        if image.ndim == 3 and image.shape[0] != 3:
            resized = cv2.resize(image, self.INPUT_SIZE)
            # BGR -> RGB, normalize to [0, 1], then ImageNet normalization
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std
            image = normalized.transpose(2, 0, 1)  # HWC -> CHW

        tensor = torch.from_numpy(image).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            logit = self._model(tensor)
            prob = torch.sigmoid(logit).item()

        confidence = min(abs(prob - 0.5) * 2, 1.0)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=prob,
            confidence=confidence,
            status=DetectorStatus.FAIL if prob > 0.5 else DetectorStatus.PASS,
            details={
                "probability": round(prob, 4),
                "label": "FAKE" if prob > 0.5 else "REAL",
                "weights_source": self._weights_source,
                "input_size": f"{self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}",
            },
        )
