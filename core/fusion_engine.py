"""
Scanner Prime - FUSION ENGINE
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements the unified decision engine with dynamic weight redistribution
and conflict resolution using standard open-source libraries (numpy). All algorithms
are based on publicly available academic research on multi-modal fusion and
decision theory.

Key Features:
- Dynamic weight redistribution when module confidence < 0.4
- Audio-aware weight adjustment for ALIGNMENT CORE
- Dual fusion modes: weighted_average and confidence_product
- Conflict resolution rules for contradictory signals
- Conservative thresholds to prevent false positives
- INCONCLUSIVE verdict for ambiguous cases
- Detailed transparency reporting for verdict explanations

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.forensic_types import (
    AlignmentCoreResult,
    ArtifactCoreResult,
    BioSignalCoreResult,
    FusionVerdict,
    TransparencyReport,
    VideoProfile,
)

# Import AudioProfile conditionally to avoid circular imports
try:
    from core.audio_analyzer import AudioProfile
except ImportError:
    AudioProfile = None


class FusionMode(Enum):
    """Fusion mode for combining core scores."""
    WEIGHTED_AVERAGE = "weighted_average"      # Current default method
    CONFIDENCE_PRODUCT = "confidence_product"  # User's formula: (Score*Conf) / TotalConf


class FusionEngine:
    """
    FUSION ENGINE - Unified Decision Engine

    Combines results from BIOSIGNAL, ARTIFACT, and ALIGNMENT cores
    with dynamic weight redistribution and conflict resolution.

    Philosophy: High Precision over High Recall.
    Better to say "I don't know" than to falsely accuse.

    Features:
    - Dual fusion modes: weighted_average and confidence_product
    - Audio-aware weight adjustment for ALIGNMENT CORE
    - Resolution-aware weight adjustment for BIOSIGNAL CORE
    - Detailed transparency reporting
    """

    # Base weights (sum to 1.0)
    BASE_WEIGHTS = {
        "biosignal": 0.33,
        "artifact": 0.33,
        "alignment": 0.34,
    }

    # Verdict thresholds (applied to 0-1 score where 1 = manipulated)
    THRESHOLD_AUTHENTIC = 0.30
    THRESHOLD_UNCERTAIN = 0.50
    THRESHOLD_MANIPULATED = 0.65

    # Minimum confidence for a core to be trusted
    MIN_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, fusion_mode: FusionMode = FusionMode.WEIGHTED_AVERAGE):
        """
        Initialize the Fusion Engine.

        Args:
            fusion_mode: Method for combining core scores
        """
        self.name = "FUSION ENGINE"
        self.fusion_mode = fusion_mode

    def _redistribute_weights(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        video_profile: Optional[VideoProfile] = None,
        audio_profile: Optional["AudioProfile"] = None
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Dynamically redistribute weights based on module confidence, video profile,
        and audio quality.

        When a module has low confidence (< 0.4), its weight is redistributed
        to higher-confidence modules.

        For low-resolution videos, BIOSIGNAL (rPPG) is inherently less reliable,
        so weight shifts to ARTIFACT and ALIGNMENT.

        For low-quality audio or no audio, ALIGNMENT (A/V sync) weight is reduced.

        Returns:
            Tuple of (Dictionary of adjusted weights, List of environmental factors)
        """
        weights = self.BASE_WEIGHTS.copy()
        environmental_factors = []

        # Get confidence levels
        confidences = {
            "biosignal": biosignal_result.confidence,
            "artifact": artifact_result.confidence,
            "alignment": alignment_result.confidence,
        }

        # ===== Resolution-based adjustment =====
        if video_profile and video_profile.is_low_res:
            # Low-res: BIOSIGNAL less reliable, ARTIFACT more reliable
            weights["biosignal"] = 0.20
            weights["artifact"] = 0.45
            weights["alignment"] = 0.35
            environmental_factors.append(
                f"Low resolution ({video_profile.height}p): BIOSIGNAL weight reduced to 20%, ARTIFACT boosted to 45%"
            )

        # ===== Audio-based adjustment =====
        if audio_profile is not None:
            if not audio_profile.has_audio:
                # No audio: significantly reduce ALIGNMENT weight
                weights["alignment"] *= 0.3
                environmental_factors.append(
                    "No audio track detected: ALIGNMENT (A/V Sync) weight reduced to 30%"
                )
            elif audio_profile.noise_level in ["HIGH", "EXTREME"]:
                # High noise: reduce ALIGNMENT weight based on recommended value
                weights["alignment"] *= audio_profile.recommended_av_weight
                environmental_factors.append(
                    f"High audio noise (SNR: {audio_profile.snr_db:.1f}dB, {audio_profile.noise_level}): "
                    f"ALIGNMENT weight reduced to {audio_profile.recommended_av_weight * 100:.0f}%"
                )
            elif audio_profile.noise_level == "MEDIUM":
                weights["alignment"] *= audio_profile.recommended_av_weight
                environmental_factors.append(
                    f"Moderate audio noise (SNR: {audio_profile.snr_db:.1f}dB): "
                    f"ALIGNMENT weight at {audio_profile.recommended_av_weight * 100:.0f}%"
                )
            elif not audio_profile.is_speech_detected:
                # No speech detected: reduce ALIGNMENT weight
                weights["alignment"] *= 0.5
                environmental_factors.append(
                    "No speech detected in audio: ALIGNMENT weight reduced to 50%"
                )

        # ===== Confidence-based redistribution =====
        low_confidence_modules = [
            k for k, v in confidences.items() if v < self.MIN_CONFIDENCE_THRESHOLD
        ]
        high_confidence_modules = [
            k for k, v in confidences.items() if v >= self.MIN_CONFIDENCE_THRESHOLD
        ]

        if low_confidence_modules and high_confidence_modules:
            # Calculate weight to redistribute
            weight_to_redistribute = sum(weights[k] * 0.5 for k in low_confidence_modules)

            # Reduce low-confidence weights
            for k in low_confidence_modules:
                weights[k] *= 0.5

            # Distribute to high-confidence modules proportionally
            high_conf_sum = sum(confidences[k] for k in high_confidence_modules)
            for k in high_confidence_modules:
                proportion = confidences[k] / high_conf_sum
                weights[k] += weight_to_redistribute * proportion

            for k in low_confidence_modules:
                environmental_factors.append(
                    f"{k.upper()} CORE low confidence ({confidences[k]:.2f}): weight reduced by 50%"
                )

        # Normalize to ensure sum is 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights, environmental_factors

    def _resolve_conflicts(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        weights: Dict[str, float]
    ) -> Tuple[str, str]:
        """
        Resolve conflicts when cores disagree.

        Returns:
            Tuple of (consensus_type, leading_core)

        Consensus types:
        - CONSENSUS_FAIL: 2+ modules indicate manipulation
        - CONSENSUS_PASS: 2+ modules indicate authentic
        - DEFER_TO_X: One module has high confidence signal
        - INCONCLUSIVE: Conflicting signals, no clear answer
        """
        # Classify each core's verdict
        def classify_score(score: float) -> str:
            if score > 0.6:
                return "FAIL"
            elif score > 0.3:
                return "WARN"
            else:
                return "PASS"

        verdicts = {
            "biosignal": classify_score(biosignal_result.score),
            "artifact": classify_score(artifact_result.score),
            "alignment": classify_score(alignment_result.score),
        }

        confidences = {
            "biosignal": biosignal_result.confidence,
            "artifact": artifact_result.confidence,
            "alignment": alignment_result.confidence,
        }

        # Count verdict types
        fail_count = sum(1 for v in verdicts.values() if v == "FAIL")
        pass_count = sum(1 for v in verdicts.values() if v == "PASS")

        # Find highest confidence core
        leading_core = max(confidences, key=confidences.get)

        # Consensus rules
        if fail_count >= 2:
            return "CONSENSUS_FAIL", leading_core

        if pass_count >= 2:
            return "CONSENSUS_PASS", leading_core

        # Check for high-confidence outlier
        for core, conf in confidences.items():
            if conf > 0.8 and verdicts[core] == "FAIL":
                return f"DEFER_TO_{core.upper()}", core

        # No clear consensus
        return "INCONCLUSIVE", leading_core

    def _compute_weighted_average_score(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        weights: Dict[str, float]
    ) -> float:
        """
        Compute score using weighted average method (default).

        Returns:
            Weighted score (0-1 where 1 = manipulated)
        """
        return (
            biosignal_result.score * weights["biosignal"] +
            artifact_result.score * weights["artifact"] +
            alignment_result.score * weights["alignment"]
        )

    def _compute_confidence_product_score(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult
    ) -> float:
        """
        Compute score using confidence-product method.

        Formula: (Bio*Conf + Pattern*Conf + Sync*Conf) / Total_Confidence

        This method weighs each core's contribution by its own confidence,
        naturally down-weighting uncertain signals.

        Returns:
            Weighted score (0-1 where 1 = manipulated)
        """
        total_conf = (
            biosignal_result.confidence +
            artifact_result.confidence +
            alignment_result.confidence
        )

        # Avoid division by zero
        if total_conf < 0.1:
            total_conf = 0.1

        weighted_sum = (
            biosignal_result.score * biosignal_result.confidence +
            artifact_result.score * artifact_result.confidence +
            alignment_result.score * alignment_result.confidence
        )

        return weighted_sum / total_conf

    def _generate_transparency_report(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        weights: Dict[str, float],
        environmental_factors: List[str],
        verdict: str,
        audio_profile: Optional["AudioProfile"] = None
    ) -> TransparencyReport:
        """
        Generate detailed transparency report explaining the verdict.

        Args:
            biosignal_result: BIOSIGNAL CORE result
            artifact_result: ARTIFACT CORE result
            alignment_result: ALIGNMENT CORE result
            weights: Final weights used
            environmental_factors: Factors that affected weights
            verdict: Final verdict
            audio_profile: Optional audio profile

        Returns:
            TransparencyReport with detailed explanations
        """
        # Generate per-engine explanations
        biosignal_explanation = self._explain_biosignal(biosignal_result, weights["biosignal"])
        artifact_explanation = self._explain_artifact(artifact_result, weights["artifact"])
        alignment_explanation = self._explain_alignment(alignment_result, weights["alignment"])

        # Identify primary concern
        primary_concern = None
        if verdict == "MANIPULATED":
            if biosignal_result.score > 0.6 and biosignal_result.confidence > 0.5:
                primary_concern = "Missing or inconsistent biological signals detected"
            elif artifact_result.score > 0.6 and artifact_result.confidence > 0.5:
                primary_concern = "Generative model fingerprints detected"
            elif alignment_result.score > 0.6 and alignment_result.confidence > 0.5:
                primary_concern = "Audio-visual synchronization anomalies detected"
            else:
                primary_concern = "Multiple weak signals indicate potential manipulation"

        # Collect supporting evidence
        supporting_evidence = []
        for anomaly in biosignal_result.anomalies:
            supporting_evidence.append(f"[BIOSIGNAL] {anomaly}")
        for anomaly in artifact_result.anomalies:
            supporting_evidence.append(f"[ARTIFACT] {anomaly}")
        for anomaly in alignment_result.anomalies:
            supporting_evidence.append(f"[ALIGNMENT] {anomaly}")

        # Build weight justifications
        weight_justification = {
            "biosignal": f"Weight {weights['biosignal']:.1%}: Biological signal analysis (rPPG)",
            "artifact": f"Weight {weights['artifact']:.1%}: Generative fingerprint detection",
            "alignment": f"Weight {weights['alignment']:.1%}: Audio-visual alignment check"
        }

        # Generate summary
        if verdict == "AUTHENTIC":
            summary = "All forensic checks passed. No manipulation indicators detected."
        elif verdict == "MANIPULATED":
            summary = f"Manipulation detected. {primary_concern}"
        elif verdict == "INCONCLUSIVE":
            summary = "Analysis inconclusive due to conflicting or insufficient signals."
        else:  # UNCERTAIN
            summary = "Some anomalies detected but not conclusive. Manual review recommended."

        # Audio quality note
        audio_note = None
        if audio_profile:
            if not audio_profile.has_audio:
                audio_note = "No audio track present. A/V sync analysis limited."
            elif audio_profile.noise_level in ["HIGH", "EXTREME"]:
                audio_note = f"High audio noise (SNR: {audio_profile.snr_db:.1f}dB). A/V sync confidence reduced."
            elif not audio_profile.is_speech_detected:
                audio_note = "No speech detected in audio. A/V sync analysis limited."

        return TransparencyReport(
            summary=summary,
            biosignal_explanation=biosignal_explanation,
            artifact_explanation=artifact_explanation,
            alignment_explanation=alignment_explanation,
            environmental_factors=environmental_factors,
            primary_concern=primary_concern,
            supporting_evidence=supporting_evidence,
            weight_justification=weight_justification,
            fusion_method=self.fusion_mode.value,
            audio_quality_note=audio_note
        )

    def _explain_biosignal(self, result: BioSignalCoreResult, weight: float) -> str:
        """Generate explanation for BIOSIGNAL CORE result."""
        status_map = {"PASS": "Normal", "WARN": "Anomalous", "FAIL": "Critical"}
        status_desc = status_map.get(result.status, "Unknown")

        explanation = (
            f"**BIOSIGNAL CORE** (Weight: {weight:.1%})\n\n"
            f"- **Status:** {status_desc}\n"
            f"- **Score:** {result.score:.2%} manipulation likelihood\n"
            f"- **Confidence:** {result.confidence:.2%}\n"
            f"- **Biological Sync:** {result.biological_sync_score:.2%}\n"
            f"- **Pulse Coverage:** {result.pulse_coverage:.2%}\n"
            f"- **HR Consistency:** {result.hr_consistency:.2%}\n"
        )

        if result.score > 0.6:
            explanation += "\n*Biological signals are inconsistent or missing.*"
        elif result.score > 0.3:
            explanation += "\n*Some biological irregularities detected.*"
        else:
            explanation += "\n*Biological signals appear natural.*"

        return explanation

    def _explain_artifact(self, result: ArtifactCoreResult, weight: float) -> str:
        """Generate explanation for ARTIFACT CORE result."""
        status_map = {"PASS": "Clean", "WARN": "Suspicious", "FAIL": "Detected"}
        status_desc = status_map.get(result.status, "Unknown")

        model_type = result.details.get("detected_model_type", "NONE")

        explanation = (
            f"**ARTIFACT CORE** (Weight: {weight:.1%})\n\n"
            f"- **Status:** {status_desc}\n"
            f"- **Score:** {result.score:.2%} manipulation likelihood\n"
            f"- **Confidence:** {result.confidence:.2%}\n"
            f"- **GAN Score:** {result.gan_score:.2%}\n"
            f"- **Diffusion Score:** {result.diffusion_score:.2%}\n"
            f"- **VAE Score:** {result.vae_score:.2%}\n"
            f"- **Detected Model:** {model_type}\n"
        )

        if result.detected_fingerprints:
            explanation += f"- **Fingerprints:** {len(result.detected_fingerprints)} detected\n"

        if result.score > 0.6:
            explanation += "\n*Generative model artifacts strongly present.*"
        elif result.score > 0.3:
            explanation += "\n*Some generative patterns detected.*"
        else:
            explanation += "\n*No generative model fingerprints found.*"

        return explanation

    def _explain_alignment(self, result: AlignmentCoreResult, weight: float) -> str:
        """Generate explanation for ALIGNMENT CORE result."""
        status_map = {"PASS": "Aligned", "WARN": "Minor Issues", "FAIL": "Misaligned"}
        status_desc = status_map.get(result.status, "Unknown")

        explanation = (
            f"**ALIGNMENT CORE** (Weight: {weight:.1%})\n\n"
            f"- **Status:** {status_desc}\n"
            f"- **Score:** {result.score:.2%} manipulation likelihood\n"
            f"- **Confidence:** {result.confidence:.2%}\n"
            f"- **A/V Alignment:** {1 - result.av_alignment_score:.2%}\n"
            f"- **Phoneme-Viseme Match:** {1 - result.phoneme_viseme_score:.2%}\n"
            f"- **Speech Rhythm:** {result.speech_rhythm_score:.2%}\n"
            f"- **Lip Closures Detected:** {len(result.lip_closure_events)}\n"
        )

        if result.score > 0.6:
            explanation += "\n*Significant audio-visual sync issues detected.*"
        elif result.score > 0.3:
            explanation += "\n*Minor synchronization anomalies present.*"
        else:
            explanation += "\n*Audio and video are well synchronized.*"

        return explanation

    def get_final_integrity_score(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        video_profile: Optional[VideoProfile] = None,
        audio_profile: Optional["AudioProfile"] = None
    ) -> FusionVerdict:
        """
        Compute final integrity score and verdict.

        The integrity score is on a 0-100 scale where:
        - 100 = completely authentic
        - 0 = definitely manipulated

        Supports two fusion modes:
        - WEIGHTED_AVERAGE: Score = sum(core_score * weight)
        - CONFIDENCE_PRODUCT: Score = sum(core_score * confidence) / sum(confidence)

        Args:
            biosignal_result: Result from BIOSIGNAL CORE
            artifact_result: Result from ARTIFACT CORE
            alignment_result: Result from ALIGNMENT CORE
            video_profile: Optional video profile for adaptive processing
            audio_profile: Optional audio profile for A/V weight adjustment

        Returns:
            FusionVerdict with final decision and transparency report
        """
        # Redistribute weights based on confidence, resolution, and audio quality
        weights, environmental_factors = self._redistribute_weights(
            biosignal_result, artifact_result, alignment_result,
            video_profile, audio_profile
        )

        # Calculate weighted score based on fusion mode
        if self.fusion_mode == FusionMode.CONFIDENCE_PRODUCT:
            weighted_score = self._compute_confidence_product_score(
                biosignal_result, artifact_result, alignment_result
            )
        else:  # WEIGHTED_AVERAGE (default)
            weighted_score = self._compute_weighted_average_score(
                biosignal_result, artifact_result, alignment_result, weights
            )

        # Convert to integrity score (0-100 where 100 = authentic)
        integrity_score = 100 * (1 - weighted_score)
        integrity_score = max(0, min(100, integrity_score))

        # Resolve conflicts
        consensus_type, leading_core = self._resolve_conflicts(
            biosignal_result, artifact_result, alignment_result, weights
        )

        # Determine verdict
        verdict, reason, conflicting = self._determine_verdict(
            weighted_score, consensus_type, leading_core,
            biosignal_result, artifact_result, alignment_result
        )

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            biosignal_result, artifact_result, alignment_result,
            consensus_type, weights
        )

        # Generate transparency report
        transparency = self._generate_transparency_report(
            biosignal_result, artifact_result, alignment_result,
            weights, environmental_factors, verdict, audio_profile
        )

        return FusionVerdict(
            verdict=verdict,
            integrity_score=round(integrity_score, 2),
            confidence=round(confidence, 4),
            biosignal_score=round(biosignal_result.score, 4),
            artifact_score=round(artifact_result.score, 4),
            alignment_score=round(alignment_result.score, 4),
            weights=weights,
            consensus_type=consensus_type,
            reason=reason,
            biosignal_result=biosignal_result,
            artifact_result=artifact_result,
            alignment_result=alignment_result,
            leading_core=leading_core,
            conflicting_signals=conflicting,
            transparency_report=transparency,
            audio_profile=audio_profile,
            fusion_method=self.fusion_mode.value
        )

    def _determine_verdict(
        self,
        weighted_score: float,
        consensus_type: str,
        leading_core: str,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult
    ) -> Tuple[str, str, bool]:
        """
        Determine final verdict based on score and consensus.

        Returns:
            Tuple of (verdict, reason, conflicting_signals)
        """
        conflicting = False

        # Check for conflicting signals
        scores = [biosignal_result.score, artifact_result.score, alignment_result.score]
        if max(scores) > 0.6 and min(scores) < 0.3:
            conflicting = True

        # Score-based verdict with consensus override
        if consensus_type == "CONSENSUS_FAIL":
            if weighted_score >= self.THRESHOLD_MANIPULATED:
                return "MANIPULATED", "Multiple cores detected manipulation patterns", conflicting
            elif weighted_score >= self.THRESHOLD_UNCERTAIN:
                return "MANIPULATED", "Consensus indicates manipulation with moderate confidence", conflicting
            else:
                # Unusual: consensus fail but low score
                return "UNCERTAIN", "Conflicting signals despite consensus", True

        elif consensus_type == "CONSENSUS_PASS":
            if weighted_score < self.THRESHOLD_AUTHENTIC:
                return "AUTHENTIC", "Multiple cores verified authenticity", conflicting
            elif weighted_score < self.THRESHOLD_UNCERTAIN:
                return "AUTHENTIC", "Consensus indicates authentic with moderate confidence", conflicting
            else:
                # Unusual: consensus pass but high score
                return "UNCERTAIN", "Score elevated despite positive consensus", True

        elif consensus_type.startswith("DEFER_TO_"):
            core_name = consensus_type.replace("DEFER_TO_", "")
            if weighted_score >= self.THRESHOLD_MANIPULATED:
                return "MANIPULATED", f"High confidence signal from {core_name} core", conflicting
            else:
                return "UNCERTAIN", f"Deferring to {core_name} core with elevated concern", conflicting

        else:  # INCONCLUSIVE
            if conflicting:
                return "INCONCLUSIVE", "Conflicting signals between cores", True
            elif weighted_score >= self.THRESHOLD_MANIPULATED:
                return "MANIPULATED", "Score indicates manipulation despite uncertain consensus", conflicting
            elif weighted_score < self.THRESHOLD_AUTHENTIC:
                return "AUTHENTIC", "Score indicates authenticity despite uncertain consensus", conflicting
            else:
                return "UNCERTAIN", "Ambiguous signals require manual review", conflicting

    def _calculate_overall_confidence(
        self,
        biosignal_result: BioSignalCoreResult,
        artifact_result: ArtifactCoreResult,
        alignment_result: AlignmentCoreResult,
        consensus_type: str,
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate overall confidence in the verdict.

        Confidence is higher when:
        - Individual cores have high confidence
        - Cores agree (strong consensus)
        - Weights are not heavily skewed
        """
        # Weighted average of individual confidences
        weighted_confidence = (
            biosignal_result.confidence * weights["biosignal"] +
            artifact_result.confidence * weights["artifact"] +
            alignment_result.confidence * weights["alignment"]
        )

        # Consensus bonus
        if consensus_type in ["CONSENSUS_FAIL", "CONSENSUS_PASS"]:
            consensus_bonus = 0.15
        elif consensus_type.startswith("DEFER_TO_"):
            consensus_bonus = 0.05
        else:  # INCONCLUSIVE
            consensus_bonus = -0.1

        # Weight balance factor (heavily skewed weights reduce confidence)
        max_weight = max(weights.values())
        balance_factor = 1.0 - (max_weight - 0.33) * 0.5

        confidence = weighted_confidence * balance_factor + consensus_bonus
        return max(0.1, min(1.0, confidence))


# Utility function for easy integration
def create_fusion_verdict(
    biosignal_result: BioSignalCoreResult,
    artifact_result: ArtifactCoreResult,
    alignment_result: AlignmentCoreResult,
    video_profile: Optional[VideoProfile] = None,
    audio_profile: Optional["AudioProfile"] = None,
    fusion_mode: FusionMode = FusionMode.WEIGHTED_AVERAGE
) -> FusionVerdict:
    """
    Convenience function to create a fusion verdict.

    Args:
        biosignal_result: Result from BIOSIGNAL CORE
        artifact_result: Result from ARTIFACT CORE
        alignment_result: Result from ALIGNMENT CORE
        video_profile: Optional video profile
        audio_profile: Optional audio profile for A/V weight adjustment
        fusion_mode: Method for combining core scores

    Returns:
        FusionVerdict with final decision and transparency report
    """
    engine = FusionEngine(fusion_mode=fusion_mode)
    return engine.get_final_integrity_score(
        biosignal_result, artifact_result, alignment_result,
        video_profile, audio_profile
    )
