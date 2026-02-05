"""
Deepfake detection model using EfficientNet-B0 backbone.

Supports loading pre-trained weights from FaceForensics++ or similar datasets,
with automatic fallback to ImageNet weights if specialized weights are unavailable.
"""

import os
import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Default path for deepfake-trained weights
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "efficientnet_b0_faceforensics.pth"


def get_deepfake_weights_path() -> Optional[Path]:
    """
    Get the path to deepfake-trained weights if available.

    Returns:
        Path to weights file if exists, None otherwise.
    """
    if DEFAULT_WEIGHTS_PATH.exists():
        return DEFAULT_WEIGHTS_PATH

    # Also check for weights in common locations
    alt_paths = [
        Path(__file__).parent / "efficientnet_b0_faceforensics.pth",
        Path(__file__).parent / "weights.pth",
        Path(__file__).parent / "model_weights.pth",
    ]

    for path in alt_paths:
        if path.exists():
            return path

    return None


class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model using EfficientNet-B0 as backbone.
    Binary classification: Real (0) vs Fake (1).

    Supports loading weights trained on deepfake datasets (FaceForensics++, DFDC, etc.)
    with automatic fallback to ImageNet initialization.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        weights_path: Optional[str] = None,
        auto_load_deepfake_weights: bool = True
    ):
        """
        Initialize the deepfake detector.

        Args:
            pretrained: Whether to use ImageNet pretrained weights for backbone.
            dropout_rate: Dropout rate before final classification layer.
            weights_path: Path to custom weights file. If None and auto_load_deepfake_weights
                          is True, will attempt to load from default location.
            auto_load_deepfake_weights: If True, automatically load deepfake-trained weights
                                        from default location if available.
        """
        super().__init__()

        self._weights_source = "imagenet"  # Track which weights are loaded
        self._input_size = 224  # Default input size for EfficientNet-B0

        # Load EfficientNet-B0 backbone from timm
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get the feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)  # Binary output
        )

        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()

        # Attempt to load deepfake-trained weights
        weights_loaded = False

        if weights_path is not None:
            # Explicit weights path provided
            weights_loaded = self._load_weights(weights_path)
        elif auto_load_deepfake_weights:
            # Try to auto-load from default location
            default_path = get_deepfake_weights_path()
            if default_path is not None:
                weights_loaded = self._load_weights(str(default_path))

        if not weights_loaded:
            print(f"[DeepfakeDetector] Using ImageNet initialization (no deepfake weights found)")
            print(f"[DeepfakeDetector] Run 'python download_weights.py' to get specialized weights")

    def _load_weights(self, weights_path: str) -> bool:
        """
        Load weights from a file.

        Args:
            weights_path: Path to the weights file.

        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(weights_path):
                print(f"[DeepfakeDetector] Warning: Weights file not found: {weights_path}")
                return False

            print(f"[DeepfakeDetector] Loading deepfake weights from: {weights_path}")

            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # Try to load the full state dict first
            try:
                self.load_state_dict(state_dict, strict=True)
                self._weights_source = "faceforensics++"
                print(f"[DeepfakeDetector] Successfully loaded deepfake-trained weights (strict)")
                return True
            except RuntimeError as e:
                # If strict loading fails, try partial loading
                print(f"[DeepfakeDetector] Strict loading failed, attempting partial load...")

                # Filter to only matching keys
                model_dict = self.state_dict()
                filtered_dict = {}

                for k, v in state_dict.items():
                    # Handle potential prefix differences
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    elif k.replace('backbone.', '') in model_dict:
                        new_k = k.replace('backbone.', '')
                        if v.shape == model_dict[new_k].shape:
                            filtered_dict[new_k] = v
                    elif f'backbone.{k}' in model_dict:
                        new_k = f'backbone.{k}'
                        if v.shape == model_dict[new_k].shape:
                            filtered_dict[new_k] = v

                if filtered_dict:
                    model_dict.update(filtered_dict)
                    self.load_state_dict(model_dict)
                    self._weights_source = "faceforensics++ (partial)"
                    print(f"[DeepfakeDetector] Loaded {len(filtered_dict)}/{len(model_dict)} weight tensors")
                    return True
                else:
                    print(f"[DeepfakeDetector] Warning: No compatible weights found in checkpoint")
                    return False

        except Exception as e:
            print(f"[DeepfakeDetector] Warning: Failed to load weights: {e}")
            return False

    @property
    def weights_source(self) -> str:
        """Return the source of the currently loaded weights."""
        return self._weights_source

    @property
    def input_size(self) -> int:
        """Return the expected input image size."""
        return self._input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Logits tensor of shape (batch, 1).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Probability tensor of shape (batch, 1) with values in [0, 1].
            Higher values indicate higher likelihood of being fake.
        """
        logits = self.forward(x)
        return self.sigmoid(logits)


class DeepfakeInference:
    """Inference wrapper for the deepfake detection model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        auto_load_deepfake_weights: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to saved model weights. If provided, overrides auto-loading.
            device: Device to run inference on. If None, auto-detects.
            auto_load_deepfake_weights: If True, automatically load deepfake-trained weights
                                        from default location if available.
        """
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model with deepfake weights auto-loading
        self.model = DeepfakeDetector(
            pretrained=True,
            weights_path=model_path,
            auto_load_deepfake_weights=auto_load_deepfake_weights if model_path is None else False
        )

        self.model.to(self.device)
        self.model.eval()

        # Log initialization
        print(f"[DeepfakeInference] Initialized on device: {self.device}")
        print(f"[DeepfakeInference] Weights source: {self.model.weights_source}")

    @property
    def weights_source(self) -> str:
        """Return the source of the currently loaded weights."""
        return self.model.weights_source

    @property
    def input_size(self) -> int:
        """Return the expected input image size."""
        return self.model.input_size

    @torch.no_grad()
    def predict_single(self, face: np.ndarray) -> Tuple[float, str]:
        """
        Predict on a single preprocessed face.

        Args:
            face: Preprocessed face array of shape (3, 224, 224).

        Returns:
            Tuple of (fake_probability, label).
        """
        # Add batch dimension and convert to tensor
        tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)

        # Get prediction
        prob = self.model.predict_proba(tensor).item()
        label = "FAKE" if prob > 0.5 else "REAL"

        return prob, label

    @torch.no_grad()
    def predict_batch(self, faces: List[np.ndarray]) -> List[Tuple[float, str]]:
        """
        Predict on a batch of preprocessed faces.

        Args:
            faces: List of preprocessed face arrays.

        Returns:
            List of tuples (fake_probability, label).
        """
        if not faces:
            return []

        # Stack and convert to tensor
        batch = np.stack(faces, axis=0)
        tensor = torch.from_numpy(batch).to(self.device)

        # Get predictions
        probs = self.model.predict_proba(tensor).cpu().numpy().flatten()

        results = []
        for prob in probs:
            label = "FAKE" if prob > 0.5 else "REAL"
            results.append((float(prob), label))

        return results

    def analyze_video_results(self, frame_results: List[Tuple[int, float, str]]) -> dict:
        """
        Aggregate results from video analysis.

        Args:
            frame_results: List of (frame_number, fake_probability, label).

        Returns:
            Aggregated analysis results.
        """
        if not frame_results:
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "fake_frame_ratio": 0.0,
                "average_fake_probability": 0.0,
                "frame_details": []
            }

        fake_probs = [r[1] for r in frame_results]
        fake_count = sum(1 for r in frame_results if r[2] == "FAKE")

        avg_prob = np.mean(fake_probs)
        fake_ratio = fake_count / len(frame_results)

        # Determine verdict based on average probability
        if avg_prob > 0.7:
            verdict = "FAKE"
            confidence = avg_prob
        elif avg_prob < 0.3:
            verdict = "REAL"
            confidence = 1 - avg_prob
        else:
            verdict = "UNCERTAIN"
            confidence = 1 - abs(0.5 - avg_prob) * 2

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "frames_analyzed": len(frame_results),
            "fake_frame_ratio": float(fake_ratio),
            "average_fake_probability": float(avg_prob),
            "frame_details": [
                {"frame": r[0], "fake_probability": r[1], "label": r[2]}
                for r in frame_results
            ]
        }
