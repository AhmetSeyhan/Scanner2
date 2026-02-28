"""
Scanner Prime - TEXT CORE
AI-generated text detection using statistical and linguistic analysis.

Detection Methods:
1. Perplexity Analysis: LLM-generated text has lower perplexity (more predictable)
2. Burstiness Analysis: Human text has variable sentence complexity; LLM text is uniform
3. Vocabulary Richness: Type-Token Ratio and Hapax Legomena analysis
4. Repetition Patterns: N-gram repetition frequency analysis
5. Sentence Structure: Entropy of sentence lengths and POS tag distributions

References:
- Mitchell et al. "DetectGPT" (2023) - perturbation-based detection
- Gehrmann et al. "GLTR" (2019) - statistical detection of machine text

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TextCoreResult:
    """Result from TEXT CORE analysis."""
    core_name: str = "TEXT CORE"
    score: float = 0.0           # 0 (human) to 1 (AI-generated)
    confidence: float = 0.0
    status: str = "PASS"         # PASS, WARN, FAIL
    details: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)
    data_quality: str = "GOOD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_name": self.core_name,
            "score": self.score,
            "confidence": self.confidence,
            "status": self.status,
            "details": self.details,
            "anomalies": self.anomalies,
            "data_quality": self.data_quality,
        }


class TextCore:
    """
    TEXT CORE - AI-Generated Text Detection

    Uses statistical linguistic analysis (no external LLM required)
    to detect machine-generated text patterns.

    This is a lightweight, zero-dependency implementation suitable for
    enterprise deployment without GPU requirements for text analysis.
    """

    # Minimum text length for reliable analysis
    MIN_WORDS = 50
    MIN_SENTENCES = 3

    def __init__(self):
        self.name = "TEXT CORE"

    def analyze(self, text: str) -> TextCoreResult:
        """
        Analyze text for AI-generation indicators.

        Args:
            text: Input text string.

        Returns:
            TextCoreResult with detection scores.
        """
        words = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)

        if len(words) < self.MIN_WORDS:
            return TextCoreResult(
                score=0.5, confidence=0.2, status="WARN",
                details={"reason": f"Text too short ({len(words)} words, need {self.MIN_WORDS})"},
                anomalies=["INSUFFICIENT_TEXT"],
                data_quality="INSUFFICIENT",
            )

        # Run all detectors
        perplexity_score, perp_details = self._analyze_perplexity(words)
        burstiness_score, burst_details = self._analyze_burstiness(sentences)
        vocab_score, vocab_details = self._analyze_vocabulary_richness(words)
        repetition_score, rep_details = self._analyze_repetition(words, sentences)
        structure_score, struct_details = self._analyze_sentence_structure(sentences)

        # Weighted combination
        weights = {
            "burstiness": 0.30,
            "vocabulary": 0.25,
            "repetition": 0.20,
            "structure": 0.15,
            "perplexity": 0.10,
        }
        overall = (
            burstiness_score * weights["burstiness"]
            + vocab_score * weights["vocabulary"]
            + repetition_score * weights["repetition"]
            + structure_score * weights["structure"]
            + perplexity_score * weights["perplexity"]
        )
        overall = max(0.0, min(1.0, overall))

        # Collect anomalies
        anomalies = []
        if burstiness_score > 0.6:
            anomalies.append("LOW_BURSTINESS_DETECTED")
        if vocab_score > 0.6:
            anomalies.append("UNIFORM_VOCABULARY")
        if repetition_score > 0.6:
            anomalies.append("HIGH_NGRAM_REPETITION")
        if structure_score > 0.6:
            anomalies.append("UNIFORM_SENTENCE_LENGTH")

        # Confidence based on text length
        length_factor = min(len(words) / 300, 1.0)
        confidence = 0.4 + length_factor * 0.5

        # Status
        if overall > 0.65:
            status = "FAIL"
        elif overall > 0.35:
            status = "WARN"
        else:
            status = "PASS"

        return TextCoreResult(
            score=round(overall, 4),
            confidence=round(confidence, 4),
            status=status,
            details={
                "word_count": len(words),
                "sentence_count": len(sentences),
                "perplexity_score": round(perplexity_score, 4),
                "burstiness_score": round(burstiness_score, 4),
                "vocabulary_score": round(vocab_score, 4),
                "repetition_score": round(repetition_score, 4),
                "structure_score": round(structure_score, 4),
                **perp_details, **burst_details, **vocab_details,
                **rep_details, **struct_details,
            },
            anomalies=anomalies,
        )

    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _analyze_perplexity(self, words: List[str]) -> Tuple[float, Dict]:
        """
        Estimate perplexity via character-level entropy.

        AI text tends to be more "predictable" - lower entropy per character.
        Human text has more surprising word choices.
        """
        text = " ".join(words)
        char_counts = Counter(text)
        total = len(text)
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in char_counts.values() if c > 0
        )

        # Bigram entropy
        bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        bg_counts = Counter(bigrams)
        bg_total = len(bigrams) or 1
        bigram_entropy = -sum(
            (c / bg_total) * math.log2(c / bg_total)
            for c in bg_counts.values() if c > 0
        )

        # AI text typically has entropy in [3.5, 4.2] for English
        # Human text typically [4.0, 4.8]
        if entropy < 3.8:
            score = 0.7
        elif entropy < 4.2:
            score = 0.4
        else:
            score = 0.1

        return score, {"char_entropy": round(entropy, 4), "bigram_entropy": round(bigram_entropy, 4)}

    def _analyze_burstiness(self, sentences: List[str]) -> Tuple[float, Dict]:
        """
        Measure burstiness - variation in sentence complexity.

        Human writing has "bursts" of complexity (short + long sentences mixed).
        AI text tends to be more uniform in sentence length/complexity.
        """
        if len(sentences) < self.MIN_SENTENCES:
            return 0.5, {"burstiness_cv": 0.0}

        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)

        # Coefficient of variation (CV)
        cv = std_len / mean_len if mean_len > 0 else 0

        # Human text CV typically > 0.5, AI text < 0.3
        if cv < 0.2:
            score = 0.8
        elif cv < 0.35:
            score = 0.5
        elif cv < 0.5:
            score = 0.3
        else:
            score = 0.1

        return score, {
            "burstiness_cv": round(float(cv), 4),
            "mean_sentence_length": round(float(mean_len), 2),
            "std_sentence_length": round(float(std_len), 2),
        }

    def _analyze_vocabulary_richness(self, words: List[str]) -> Tuple[float, Dict]:
        """
        Analyze vocabulary diversity.

        AI text often has a narrower vocabulary range with fewer rare words.
        Metrics: Type-Token Ratio (TTR), Hapax Legomena ratio.
        """
        total = len(words)
        unique = set(words)
        word_freq = Counter(words)

        # Type-Token Ratio
        ttr = len(unique) / total if total > 0 else 0

        # Hapax Legomena: words appearing only once
        hapax = sum(1 for c in word_freq.values() if c == 1)
        hapax_ratio = hapax / total if total > 0 else 0

        # Corrected TTR (Guiraud's index) - normalised for text length
        guiraud = len(unique) / math.sqrt(total) if total > 0 else 0

        # AI text: lower TTR, fewer hapax legomena
        score = 0.0
        if ttr < 0.35:
            score += 0.4
        elif ttr < 0.45:
            score += 0.2

        if hapax_ratio < 0.25:
            score += 0.4
        elif hapax_ratio < 0.35:
            score += 0.2

        score = min(score, 1.0)

        return score, {
            "type_token_ratio": round(ttr, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "guiraud_index": round(guiraud, 4),
            "unique_words": len(unique),
        }

    def _analyze_repetition(self, words: List[str], sentences: List[str]) -> Tuple[float, Dict]:
        """
        Detect n-gram repetition patterns.

        AI models sometimes over-use certain phrases and transition words.
        """
        # Trigram analysis
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        trigram_freq = Counter(trigrams)
        total_trigrams = len(trigrams) or 1

        # Repeated trigrams (appearing 3+ times)
        repeated = sum(c for c in trigram_freq.values() if c >= 3)
        repetition_ratio = repeated / total_trigrams

        # Common AI transition phrases
        ai_markers = [
            "it is important to note", "it's worth noting", "in conclusion",
            "furthermore", "moreover", "additionally", "in this context",
            "it is essential", "plays a crucial role", "it should be noted",
            "on the other hand", "in terms of", "when it comes to",
        ]
        text_lower = " ".join(words)
        marker_count = sum(1 for m in ai_markers if m in text_lower)
        marker_density = marker_count / (len(sentences) or 1)

        score = 0.0
        if repetition_ratio > 0.1:
            score += 0.4
        elif repetition_ratio > 0.05:
            score += 0.2

        if marker_density > 0.3:
            score += 0.5
        elif marker_density > 0.15:
            score += 0.25

        score = min(score, 1.0)

        return score, {
            "trigram_repetition_ratio": round(repetition_ratio, 4),
            "ai_marker_density": round(marker_density, 4),
            "ai_markers_found": marker_count,
        }

    def _analyze_sentence_structure(self, sentences: List[str]) -> Tuple[float, Dict]:
        """
        Analyze structural patterns in sentence construction.

        AI text tends to have very uniform sentence beginnings and
        paragraph structure.
        """
        if len(sentences) < self.MIN_SENTENCES:
            return 0.5, {}

        # Sentence-start word analysis
        starts = [s.split()[0].lower() for s in sentences if s.split()]
        start_freq = Counter(starts)
        unique_starts = len(start_freq)
        start_diversity = unique_starts / len(starts) if starts else 0

        # Sentence length entropy
        lengths = [len(s.split()) for s in sentences]
        length_counts = Counter(lengths)
        total = len(lengths)
        length_entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in length_counts.values() if c > 0
        )

        # Max possible entropy for this number of sentences
        max_entropy = math.log2(total) if total > 1 else 1
        normalised_entropy = length_entropy / max_entropy if max_entropy > 0 else 0

        score = 0.0
        # Low diversity in sentence starts
        if start_diversity < 0.4:
            score += 0.4
        elif start_diversity < 0.6:
            score += 0.2

        # Low entropy in sentence lengths
        if normalised_entropy < 0.5:
            score += 0.4
        elif normalised_entropy < 0.7:
            score += 0.2

        score = min(score, 1.0)

        return score, {
            "sentence_start_diversity": round(start_diversity, 4),
            "sentence_length_entropy": round(normalised_entropy, 4),
        }
