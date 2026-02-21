"""
Scanner - Cross-Modal Attention Fusion (v6.0.0)

Attention-based fusion mechanism that learns to weight different
detection modalities based on their relevance and reliability for
the current input. Replaces simple weighted average with dynamic
attention-based combination.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModalityScore:
    """Score from a single detection modality."""
    name: str
    score: float           # 0.0 -> 1.0
    confidence: float      # 0.0 -> 1.0
    modality_type: str     # visual, audio, text, biological
    features: Optional[np.ndarray] = None


@dataclass
class AttentionFusionResult:
    """Result from cross-modal attention fusion."""
    fused_score: float
    fused_confidence: float
    attention_weights: Dict[str, float]
    agreement_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class CrossModalAttention:
    """Attention-based cross-modal fusion engine.

    Computes attention weights dynamically based on:
    1. Individual confidence scores
    2. Cross-modal agreement/disagreement
    3. Input quality indicators
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_attention(self, modalities: List[ModalityScore]) -> Dict[str, float]:
        """Compute attention weights for each modality.

        Uses confidence-weighted softmax with agreement bonus.
        """
        if not modalities:
            return {}

        if len(modalities) == 1:
            return {modalities[0].name: 1.0}

        # Base weights from confidence
        confidences = np.array([m.confidence for m in modalities])

        # Agreement bonus: modalities that agree get higher weight
        scores = np.array([m.score for m in modalities])
        median_score = np.median(scores)
        agreement = 1.0 - np.abs(scores - median_score)

        # Combined weight: confidence * agreement
        raw_weights = confidences * (0.7 + 0.3 * agreement)

        # Softmax with temperature
        exp_weights = np.exp(raw_weights / self.temperature)
        attention = exp_weights / (np.sum(exp_weights) + 1e-10)

        return {m.name: float(w) for m, w in zip(modalities, attention)}

    def fuse(self, modalities: List[ModalityScore]) -> AttentionFusionResult:
        """Fuse multiple modality scores using attention mechanism."""
        if not modalities:
            return AttentionFusionResult(
                fused_score=0.5,
                fused_confidence=0.0,
                attention_weights={},
                agreement_score=0.0,
            )

        attention_weights = self.compute_attention(modalities)

        # Weighted fusion
        fused_score = sum(
            m.score * attention_weights[m.name]
            for m in modalities
        )

        # Fused confidence
        fused_confidence = sum(
            m.confidence * attention_weights[m.name]
            for m in modalities
        )

        # Agreement score: how much do modalities agree?
        scores = [m.score for m in modalities]
        agreement = 1.0 - float(np.std(scores)) * 2 if len(scores) > 1 else 1.0
        agreement = float(np.clip(agreement, 0.0, 1.0))

        # Boost confidence when there's strong agreement
        if agreement > 0.8:
            fused_confidence = min(fused_confidence * 1.2, 1.0)
        elif agreement < 0.3:
            fused_confidence *= 0.7

        return AttentionFusionResult(
            fused_score=float(np.clip(fused_score, 0.0, 1.0)),
            fused_confidence=float(np.clip(fused_confidence, 0.0, 1.0)),
            attention_weights=attention_weights,
            agreement_score=agreement,
            details={
                "num_modalities": len(modalities),
                "score_range": [round(min(scores), 4), round(max(scores), 4)],
                "temperature": self.temperature,
            },
        )
