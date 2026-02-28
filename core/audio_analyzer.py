"""
Scanner Prime - Audio Analyzer Module
Original Implementation by Scanner Prime Team based on Public Academic Research.

This module implements audio extraction and Signal-to-Noise Ratio (SNR) analysis
for adaptive A/V Sync weighting in the FUSION ENGINE using standard open-source
libraries (librosa, numpy, soundfile). All algorithms are based on publicly
available academic research on audio signal processing.

Features:
- Extracts audio from video files using librosa/ffmpeg
- Estimates Signal-to-Noise Ratio (SNR) in decibels
- Classifies noise levels (LOW, MEDIUM, HIGH, EXTREME)
- Provides recommended ALIGNMENT CORE weight based on audio quality

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Audio processing imports with fallbacks
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class AudioProfile:
    """
    Audio quality profile for adaptive A/V Sync weighting.

    Attributes:
        has_audio: Whether the video contains an audio track
        snr_db: Estimated Signal-to-Noise Ratio in decibels
        noise_level: Categorical noise classification
        recommended_av_weight: Recommended weight multiplier for ALIGNMENT CORE (0.3-1.0)
        duration_seconds: Audio duration in seconds
        sample_rate: Audio sample rate in Hz
        is_speech_detected: Whether speech was detected in the audio
        extraction_method: Method used to extract audio ("librosa", "ffmpeg", "none")
    """
    has_audio: bool
    snr_db: float
    noise_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME", "NO_AUDIO"
    recommended_av_weight: float
    duration_seconds: float = 0.0
    sample_rate: int = 0
    is_speech_detected: bool = False
    extraction_method: str = "none"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "has_audio": self.has_audio,
            "snr_db": round(self.snr_db, 2),
            "noise_level": self.noise_level,
            "recommended_av_weight": round(self.recommended_av_weight, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "sample_rate": self.sample_rate,
            "is_speech_detected": self.is_speech_detected,
            "extraction_method": self.extraction_method
        }


class AudioAnalyzer:
    """
    Audio extraction and SNR analysis for adaptive weighting.

    SNR Thresholds:
    - >= 20 dB: LOW noise -> weight 1.0 (full trust in A/V sync)
    - 10-20 dB: MEDIUM noise -> weight 0.7
    - 5-10 dB: HIGH noise -> weight 0.5
    - < 5 dB: EXTREME noise -> weight 0.3
    - No audio: weight 0.3
    """

    # SNR thresholds in dB
    SNR_LOW_THRESHOLD = 20.0      # Very clean audio
    SNR_MEDIUM_THRESHOLD = 10.0   # Acceptable noise
    SNR_HIGH_THRESHOLD = 5.0      # Significant noise

    # Weight multipliers for ALIGNMENT CORE
    WEIGHT_LOW_NOISE = 1.0
    WEIGHT_MEDIUM_NOISE = 0.7
    WEIGHT_HIGH_NOISE = 0.5
    WEIGHT_EXTREME_NOISE = 0.3
    WEIGHT_NO_AUDIO = 0.3

    # Speech detection thresholds
    SPEECH_ENERGY_THRESHOLD = 0.01  # Minimum RMS for speech presence

    def __init__(self):
        """Initialize audio analyzer."""
        self.name = "AUDIO ANALYZER"
        self._librosa_available = LIBROSA_AVAILABLE

    def extract_audio(self, video_path: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract audio signal from video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (audio_signal as numpy array, sample_rate)
            Returns (None, 0) if extraction fails
        """
        if not os.path.exists(video_path):
            return None, 0

        # Try librosa first (most reliable for various formats)
        if self._librosa_available:
            try:
                audio, sr = librosa.load(video_path, sr=None, mono=True)
                if len(audio) > 0:
                    return audio, sr
            except Exception:
                pass  # Fall through to ffmpeg method

        # Fallback: Use ffmpeg to extract audio to temp file
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM format
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                '-loglevel', 'error',
                tmp_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and os.path.exists(tmp_path):
                # Read the extracted audio
                if self._librosa_available:
                    audio, sr = librosa.load(tmp_path, sr=None, mono=True)
                else:
                    # Use soundfile if available
                    try:
                        audio, sr = sf.read(tmp_path)
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)  # Convert to mono
                    except Exception:
                        audio, sr = None, 0

                # Cleanup
                os.unlink(tmp_path)

                if audio is not None and len(audio) > 0:
                    return audio, sr

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # ffmpeg not available or failed
            pass

        return None, 0

    def estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate Signal-to-Noise Ratio using spectral analysis.

        Uses the method of comparing high-energy (signal) regions
        to low-energy (noise floor) regions in the spectrogram.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate

        Returns:
            Estimated SNR in decibels
        """
        if audio is None or len(audio) == 0:
            return 0.0

        # Ensure we have enough samples
        min_samples = sr // 2  # At least 0.5 seconds
        if len(audio) < min_samples:
            return 0.0

        # Compute short-time energy in frames
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop

        # Pad audio if needed
        if len(audio) < frame_length:
            audio = np.pad(audio, (0, frame_length - len(audio)))

        # Calculate frame energies
        num_frames = (len(audio) - frame_length) // hop_length + 1
        if num_frames < 2:
            return 0.0

        energies = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            energy = np.sum(frame ** 2)
            energies.append(energy)

        energies = np.array(energies)
        energies = energies[energies > 0]  # Remove zero-energy frames

        if len(energies) < 2:
            return 0.0

        # Sort energies to find signal vs noise
        sorted_energies = np.sort(energies)

        # Noise floor: bottom 10% of energy frames
        noise_percentile = int(len(sorted_energies) * 0.1)
        if noise_percentile < 1:
            noise_percentile = 1
        noise_floor = np.mean(sorted_energies[:noise_percentile])

        # Signal: top 30% of energy frames
        signal_percentile = int(len(sorted_energies) * 0.7)
        signal_level = np.mean(sorted_energies[signal_percentile:])

        # Calculate SNR
        if noise_floor > 0 and signal_level > noise_floor:
            snr_db = 10 * np.log10(signal_level / noise_floor)
        else:
            # Very clean signal or edge case
            snr_db = 30.0  # Cap at 30 dB (very good)

        # Clamp to reasonable range
        snr_db = np.clip(snr_db, 0.0, 40.0)

        return float(snr_db)

    def detect_speech_presence(self, audio: np.ndarray, sr: int) -> bool:
        """
        Detect if speech is present in the audio.

        Uses energy-based Voice Activity Detection (VAD).

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            True if speech is likely present
        """
        if audio is None or len(audio) == 0:
            return False

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Speech typically has higher energy than background noise
        if rms < self.SPEECH_ENERGY_THRESHOLD:
            return False

        # Check for speech-like spectral characteristics
        if self._librosa_available:
            try:
                # Zero crossing rate (speech has specific patterns)
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                mean_zcr = np.mean(zcr)

                # Speech typically has ZCR between 0.02 and 0.20
                if 0.02 < mean_zcr < 0.20:
                    return True

            except Exception:
                pass

        # Fallback: energy-based detection
        return rms > self.SPEECH_ENERGY_THRESHOLD * 2

    def classify_noise_level(self, snr_db: float) -> Tuple[str, float]:
        """
        Classify noise level and determine recommended weight.

        Args:
            snr_db: Signal-to-Noise Ratio in decibels

        Returns:
            Tuple of (noise_level string, recommended_weight)
        """
        if snr_db >= self.SNR_LOW_THRESHOLD:
            return "LOW", self.WEIGHT_LOW_NOISE
        elif snr_db >= self.SNR_MEDIUM_THRESHOLD:
            return "MEDIUM", self.WEIGHT_MEDIUM_NOISE
        elif snr_db >= self.SNR_HIGH_THRESHOLD:
            return "HIGH", self.WEIGHT_HIGH_NOISE
        else:
            return "EXTREME", self.WEIGHT_EXTREME_NOISE

    def analyze(self, video_path: str) -> AudioProfile:
        """
        Complete audio analysis for a video file.

        Extracts audio, estimates SNR, and provides recommendations
        for ALIGNMENT CORE weight adjustment.

        Args:
            video_path: Path to video file

        Returns:
            AudioProfile with complete analysis results
        """
        # Extract audio
        audio, sr = self.extract_audio(video_path)

        # No audio case
        if audio is None or len(audio) == 0:
            return AudioProfile(
                has_audio=False,
                snr_db=0.0,
                noise_level="NO_AUDIO",
                recommended_av_weight=self.WEIGHT_NO_AUDIO,
                duration_seconds=0.0,
                sample_rate=0,
                is_speech_detected=False,
                extraction_method="none"
            )

        # Calculate duration
        duration = len(audio) / sr

        # Estimate SNR
        snr_db = self.estimate_snr(audio, sr)

        # Classify noise level and get weight
        noise_level, weight = self.classify_noise_level(snr_db)

        # Detect speech presence
        has_speech = self.detect_speech_presence(audio, sr)

        # If no speech detected, further reduce weight
        if not has_speech:
            weight *= 0.8  # Reduce weight if no speech
            weight = max(weight, self.WEIGHT_NO_AUDIO)  # Floor at no-audio weight

        # Determine extraction method
        extraction_method = "librosa" if self._librosa_available else "ffmpeg"

        return AudioProfile(
            has_audio=True,
            snr_db=snr_db,
            noise_level=noise_level,
            recommended_av_weight=weight,
            duration_seconds=duration,
            sample_rate=sr,
            is_speech_detected=has_speech,
            extraction_method=extraction_method
        )


# Convenience function
def analyze_video_audio(video_path: str) -> AudioProfile:
    """
    Convenience function to analyze audio in a video.

    Args:
        video_path: Path to video file

    Returns:
        AudioProfile with analysis results
    """
    analyzer = AudioAnalyzer()
    return analyzer.analyze(video_path)
