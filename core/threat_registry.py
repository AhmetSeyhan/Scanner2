"""
Scanner Prime - Threat Registry
Tracks known and emerging deepfake generation methods for model adaptation.

Each registered threat includes expected forensic signatures that help
the ARTIFACT CORE and FUSION ENGINE apply threat-specific weights.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ThreatCategory(Enum):
    FACE_SWAP = "face_swap"
    FACE_REENACT = "face_reenactment"
    FULL_SYNTHESIS = "full_synthesis"
    VOICE_CLONE = "voice_clone"
    LIP_SYNC = "lip_sync"
    TEXT_GEN = "text_generation"


class GeneratorFamily(Enum):
    GAN = "gan"
    DIFFUSION = "diffusion"
    VAE = "vae"
    AUTOREGRESSIVE = "autoregressive"
    HYBRID = "hybrid"


@dataclass
class ThreatSignature:
    """Expected forensic signature for a threat."""
    name: str
    category: ThreatCategory
    generator_family: GeneratorFamily
    first_seen: str
    expected_biosignal_impact: float  # 0-1 how much rPPG is affected
    expected_artifact_pattern: str     # GAN/DIFFUSION/VAE/NONE
    expected_alignment_impact: float   # 0-1 how much A/V sync is affected
    notes: str = ""


# Known threat registry - extend as new generators appear
THREAT_REGISTRY: Dict[str, ThreatSignature] = {
    "deepfakes_ff": ThreatSignature(
        name="DeepFakes (FF++)", category=ThreatCategory.FACE_SWAP,
        generator_family=GeneratorFamily.GAN, first_seen="2018",
        expected_biosignal_impact=0.7, expected_artifact_pattern="GAN",
        expected_alignment_impact=0.3,
    ),
    "face2face": ThreatSignature(
        name="Face2Face", category=ThreatCategory.FACE_REENACT,
        generator_family=GeneratorFamily.GAN, first_seen="2016",
        expected_biosignal_impact=0.5, expected_artifact_pattern="GAN",
        expected_alignment_impact=0.6,
    ),
    "faceswap": ThreatSignature(
        name="FaceSwap", category=ThreatCategory.FACE_SWAP,
        generator_family=GeneratorFamily.GAN, first_seen="2018",
        expected_biosignal_impact=0.8, expected_artifact_pattern="GAN",
        expected_alignment_impact=0.2,
    ),
    "neural_textures": ThreatSignature(
        name="NeuralTextures", category=ThreatCategory.FACE_REENACT,
        generator_family=GeneratorFamily.GAN, first_seen="2019",
        expected_biosignal_impact=0.4, expected_artifact_pattern="GAN",
        expected_alignment_impact=0.5,
    ),
    "wav2lip": ThreatSignature(
        name="Wav2Lip", category=ThreatCategory.LIP_SYNC,
        generator_family=GeneratorFamily.GAN, first_seen="2020",
        expected_biosignal_impact=0.2, expected_artifact_pattern="GAN",
        expected_alignment_impact=0.8,
        notes="Only modifies lip region; ALIGNMENT core is primary detector",
    ),
    "stable_diffusion_inpaint": ThreatSignature(
        name="Stable Diffusion Inpainting", category=ThreatCategory.FULL_SYNTHESIS,
        generator_family=GeneratorFamily.DIFFUSION, first_seen="2022",
        expected_biosignal_impact=0.9, expected_artifact_pattern="DIFFUSION",
        expected_alignment_impact=0.1,
    ),
    "midjourney_v6": ThreatSignature(
        name="Midjourney v6", category=ThreatCategory.FULL_SYNTHESIS,
        generator_family=GeneratorFamily.DIFFUSION, first_seen="2024",
        expected_biosignal_impact=0.95, expected_artifact_pattern="DIFFUSION",
        expected_alignment_impact=0.0,
        notes="Image-only; no video/audio analysis applicable",
    ),
    "elevenlabs_clone": ThreatSignature(
        name="ElevenLabs Voice Clone", category=ThreatCategory.VOICE_CLONE,
        generator_family=GeneratorFamily.AUTOREGRESSIVE, first_seen="2023",
        expected_biosignal_impact=0.0, expected_artifact_pattern="NONE",
        expected_alignment_impact=0.9,
        notes="Audio-only deepfake; ALIGNMENT and AUDIO cores primary detectors",
    ),
    "heygen_avatar": ThreatSignature(
        name="HeyGen AI Avatar", category=ThreatCategory.FULL_SYNTHESIS,
        generator_family=GeneratorFamily.HYBRID, first_seen="2023",
        expected_biosignal_impact=0.85, expected_artifact_pattern="DIFFUSION",
        expected_alignment_impact=0.7,
    ),
    "sora": ThreatSignature(
        name="OpenAI Sora", category=ThreatCategory.FULL_SYNTHESIS,
        generator_family=GeneratorFamily.DIFFUSION, first_seen="2024",
        expected_biosignal_impact=0.9, expected_artifact_pattern="DIFFUSION",
        expected_alignment_impact=0.4,
        notes="Video generation; temporal coherence is high but rPPG absent",
    ),
}


def get_recommended_weights(threat_name: str) -> Optional[Dict[str, float]]:
    """Get recommended core weights for a specific known threat."""
    sig = THREAT_REGISTRY.get(threat_name)
    if sig is None:
        return None

    # Weight by expected impact (higher impact = more weight)
    raw = {
        "biosignal": sig.expected_biosignal_impact,
        "artifact": 0.5 if sig.expected_artifact_pattern != "NONE" else 0.1,
        "alignment": sig.expected_alignment_impact,
    }
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def list_threats() -> List[Dict]:
    """List all registered threats with metadata."""
    return [
        {
            "key": key,
            "name": sig.name,
            "category": sig.category.value,
            "generator_family": sig.generator_family.value,
            "first_seen": sig.first_seen,
            "notes": sig.notes,
        }
        for key, sig in THREAT_REGISTRY.items()
    ]
