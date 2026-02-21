"""
Scanner - Constant-Q Transform Audio Detector (v6.0.0)

Uses Constant-Q Transform (CQT) spectrograms for audio deepfake
detection. CQT provides logarithmically-spaced frequency bins that
better capture harmonic structure of speech, making it superior to
MelSpec for detecting AI-generated voice artifacts.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)


@dataclass
class CQTAnalysisResult:
    """Result from CQT spectrogram analysis."""
    score: float
    confidence: float
    harmonic_ratio: float
    spectral_continuity: float
    formant_consistency: float
    anomalies: List[str] = field(default_factory=list)


class CQTAnalyzer:
    """Analyze audio using Constant-Q Transform spectrograms."""

    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    N_BINS = 84  # 7 octaves
    BINS_PER_OCTAVE = 12

    def extract_cqt(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute CQT spectrogram from audio signal."""
        try:
            import librosa
            cqt = np.abs(librosa.cqt(
                y=audio.astype(np.float32),
                sr=sr,
                hop_length=self.HOP_LENGTH,
                n_bins=self.N_BINS,
                bins_per_octave=self.BINS_PER_OCTAVE,
            ))
            return librosa.amplitude_to_db(cqt, ref=np.max)
        except ImportError:
            # Fallback: compute simplified frequency representation
            return self._fallback_spectrogram(audio, sr)

    def _fallback_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback spectrogram using FFT when librosa not available."""
        frame_size = 2048
        hop = self.HOP_LENGTH
        n_frames = max(1, (len(audio) - frame_size) // hop)

        spec = np.zeros((self.N_BINS, n_frames))
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            fft = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            # Resample to N_BINS using log-spaced indices
            log_indices = np.logspace(0, np.log10(len(fft) - 1), self.N_BINS, dtype=int)
            log_indices = np.clip(log_indices, 0, len(fft) - 1)
            spec[:, i] = fft[log_indices]

        # Convert to dB
        spec = 20 * np.log10(spec + 1e-10)
        return spec

    def analyze_harmonic_structure(self, cqt_db: np.ndarray) -> float:
        """Analyze harmonic consistency in CQT spectrogram.

        Real speech has clear harmonic series with integer-ratio spacing.
        AI-generated speech often has subtle harmonic irregularities.
        """
        if cqt_db.shape[1] < 2:
            return 0.5

        # Average spectrum across time
        avg_spectrum = np.mean(cqt_db, axis=1)

        # Find peaks (harmonics)
        peaks = []
        for i in range(1, len(avg_spectrum) - 1):
            if avg_spectrum[i] > avg_spectrum[i-1] and avg_spectrum[i] > avg_spectrum[i+1]:
                peaks.append(i)

        if len(peaks) < 3:
            return 0.5

        # Check harmonic spacing regularity
        spacings = np.diff(peaks)
        if len(spacings) == 0:
            return 0.5

        mean_spacing = np.mean(spacings)
        if mean_spacing < 1e-8:
            return 0.5

        regularity = 1.0 - float(np.std(spacings) / (mean_spacing + 1e-8))
        return float(np.clip(regularity, 0.0, 1.0))

    def analyze_spectral_continuity(self, cqt_db: np.ndarray) -> float:
        """Check temporal continuity of spectral features.

        Natural speech evolves smoothly; synthetic speech may have
        frame-boundary artifacts or unnatural transitions.
        """
        if cqt_db.shape[1] < 3:
            return 0.5

        # Frame-to-frame difference
        frame_diffs = np.diff(cqt_db, axis=1)
        avg_diff = np.mean(np.abs(frame_diffs))
        max_diff = np.max(np.abs(frame_diffs))

        # Ratio of max to avg indicates spike artifacts
        if avg_diff < 1e-8:
            return 0.5

        spike_ratio = max_diff / (avg_diff + 1e-8)
        # High spike ratio = discontinuity = suspicious
        return float(np.clip(1.0 - spike_ratio / 20.0, 0.0, 1.0))

    def analyze_formant_consistency(self, cqt_db: np.ndarray) -> float:
        """Analyze formant frequency consistency over time.

        Real voices have relatively stable formant tracks.
        TTS/VC may show abrupt formant jumps.
        """
        if cqt_db.shape[1] < 5:
            return 0.5

        # Track the top-3 energy bins per frame (proxy for formants)
        n_formants = 3
        formant_tracks = np.zeros((n_formants, cqt_db.shape[1]))

        for t in range(cqt_db.shape[1]):
            frame = cqt_db[:, t]
            top_indices = np.argsort(frame)[-n_formants:]
            formant_tracks[:, t] = np.sort(top_indices)

        # Check stability of formant tracks
        stabilities = []
        for f in range(n_formants):
            track = formant_tracks[f, :]
            diffs = np.abs(np.diff(track))
            stability = 1.0 - float(np.mean(diffs) / (self.N_BINS / 2))
            stabilities.append(max(stability, 0.0))

        return float(np.mean(stabilities))

    def analyze(self, audio: np.ndarray, sr: int) -> CQTAnalysisResult:
        """Full CQT analysis pipeline."""
        cqt_db = self.extract_cqt(audio, sr)

        harmonic_ratio = self.analyze_harmonic_structure(cqt_db)
        spectral_cont = self.analyze_spectral_continuity(cqt_db)
        formant_cons = self.analyze_formant_consistency(cqt_db)

        anomalies = []
        if harmonic_ratio < 0.4:
            anomalies.append("Irregular harmonic structure")
        if spectral_cont < 0.3:
            anomalies.append("Spectral discontinuities detected")
        if formant_cons < 0.5:
            anomalies.append("Unstable formant tracks")

        # Weighted score
        score = 1.0 - (harmonic_ratio * 0.35 + spectral_cont * 0.35 + formant_cons * 0.3)
        score = float(np.clip(score, 0.0, 1.0))

        confidence = min(0.4 + abs(score - 0.5) * 0.6 + len(anomalies) * 0.1, 1.0)

        return CQTAnalysisResult(
            score=score,
            confidence=confidence,
            harmonic_ratio=harmonic_ratio,
            spectral_continuity=spectral_cont,
            formant_consistency=formant_cons,
            anomalies=anomalies,
        )


class CQTDetector(BaseDetector):
    """CQT spectrogram-based audio deepfake detector."""

    def __init__(self):
        self._analyzer = CQTAnalyzer()

    @property
    def name(self) -> str:
        return "CQT Audio Detector"

    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.AUDIO

    @property
    def capabilities(self) -> Set[DetectorCapability]:
        return {DetectorCapability.AUDIO_TRACK}

    @property
    def version(self) -> str:
        return "6.0.0"

    def _run_detection(self, inp: DetectorInput) -> DetectorResult:
        if inp.audio_profile is None and inp.video_path is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "No audio data available"},
            )

        # Extract audio from video path if needed
        audio, sr = self._load_audio(inp)
        if audio is None:
            return DetectorResult(
                detector_name=self.name,
                detector_type=self.detector_type,
                score=0.5,
                confidence=0.0,
                status=DetectorStatus.SKIPPED,
                data_quality="INSUFFICIENT",
                details={"reason": "Could not extract audio"},
            )

        result = self._analyzer.analyze(audio, sr)

        return DetectorResult(
            detector_name=self.name,
            detector_type=self.detector_type,
            score=result.score,
            confidence=result.confidence,
            status=DetectorStatus.PASS,
            anomalies=result.anomalies,
            details={
                "harmonic_ratio": round(result.harmonic_ratio, 4),
                "spectral_continuity": round(result.spectral_continuity, 4),
                "formant_consistency": round(result.formant_consistency, 4),
                "sample_rate": sr,
            },
        )

    def _load_audio(self, inp: DetectorInput) -> tuple:
        """Load audio from input."""
        if inp.video_path:
            try:
                import librosa
                audio, sr = librosa.load(inp.video_path, sr=22050, mono=True)
                return audio, sr
            except Exception:
                return None, 0
        return None, 0
