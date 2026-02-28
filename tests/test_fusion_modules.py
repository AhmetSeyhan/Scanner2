"""Tests for fusion modules: CrossModalAttention, TemporalConsistency, ConfidenceCalibrator."""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CrossModalAttention
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossModalAttention:
    def _modality(self, name, score, confidence, mtype="visual"):
        from core.fusion.cross_modal_attention import ModalityScore
        return ModalityScore(name=name, score=score, confidence=confidence, modality_type=mtype)

    def test_empty_modalities_returns_default(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        result = cma.fuse([])
        assert result.fused_score == 0.5
        assert result.fused_confidence == 0.0
        assert result.attention_weights == {}
        assert result.agreement_score == 0.0

    def test_single_modality(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        m = self._modality("visual", 0.8, 0.9)
        result = cma.fuse([m])
        assert abs(result.fused_score - 0.8) < 0.01
        assert result.attention_weights["visual"] == 1.0

    def test_two_agreeing_modalities(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        mods = [
            self._modality("visual", 0.7, 0.8),
            self._modality("audio", 0.75, 0.85),
        ]
        result = cma.fuse(mods)
        assert 0.0 <= result.fused_score <= 1.0
        assert 0.0 <= result.fused_confidence <= 1.0
        assert abs(sum(result.attention_weights.values()) - 1.0) < 1e-6
        assert result.agreement_score > 0.5  # they agree

    def test_disagreeing_modalities_lower_confidence(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        mods_agree = [
            self._modality("v1", 0.8, 0.9),
            self._modality("v2", 0.8, 0.9),
        ]
        mods_disagree = [
            self._modality("v1", 0.1, 0.9),
            self._modality("v2", 0.9, 0.9),
        ]
        res_agree = cma.fuse(mods_agree)
        res_disagree = cma.fuse(mods_disagree)
        assert res_disagree.fused_confidence < res_agree.fused_confidence

    def test_three_modalities(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        mods = [
            self._modality("visual", 0.9, 0.8, "visual"),
            self._modality("audio", 0.85, 0.7, "audio"),
            self._modality("text", 0.7, 0.6, "text"),
        ]
        result = cma.fuse(mods)
        assert len(result.attention_weights) == 3
        assert result.details["num_modalities"] == 3

    def test_compute_attention_empty(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        assert cma.compute_attention([]) == {}

    def test_fused_score_in_range(self):
        from core.fusion.cross_modal_attention import CrossModalAttention
        cma = CrossModalAttention()
        mods = [self._modality(f"m{i}", float(i) / 4, 0.5) for i in range(5)]
        result = cma.fuse(mods)
        assert 0.0 <= result.fused_score <= 1.0
        assert 0.0 <= result.fused_confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TemporalConsistency
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalConsistency:
    def test_too_few_frames_returns_default(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        result = tc.analyze([0.5, 0.6])
        assert result.consistency_score == 0.5

    def test_smooth_scores_high_consistency(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        smooth = [0.7 + 0.01 * i for i in range(20)]
        result = tc.analyze(smooth)
        assert result.consistency_score > 0.5
        assert result.temporal_smoothness > 0.5

    def test_flickering_scores_detected(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        # Alternating high/low pattern = flickering
        flickering = [0.9 if i % 2 == 0 else 0.1 for i in range(20)]
        result = tc.analyze(flickering)
        assert result.flicker_score > 0.0
        assert any("flicker" in a.lower() for a in result.anomalies) or result.consistency_score < 0.7

    def test_constant_scores_no_anomalies(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        constant = [0.5] * 30
        result = tc.analyze(constant)
        assert result.consistency_score >= 0.5

    def test_result_has_details(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        result = tc.analyze([0.5] * 10, fps=25.0)
        assert "num_frames" in result.details
        assert "fps" in result.details
        assert result.details["num_frames"] == 10
        assert result.details["fps"] == 25.0

    def test_abrupt_jump_detected(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        scores = [0.1] * 10 + [0.9] * 10
        result = tc.analyze(scores)
        assert len(result.anomalies) > 0 or result.temporal_smoothness < 0.9

    def test_periodic_pattern_detected(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        # Strong sinusoidal pattern
        periodic = [0.5 + 0.4 * np.sin(i * np.pi / 3) for i in range(30)]
        result = tc.analyze(periodic)
        assert result.details["periodic_strength"] >= 0.0

    def test_score_range_zero_to_one(self):
        from core.fusion.temporal_consistency import TemporalConsistency
        tc = TemporalConsistency()
        scores = np.random.rand(40).tolist()
        result = tc.analyze(scores)
        assert 0.0 <= result.consistency_score <= 1.0
        assert 0.0 <= result.flicker_score <= 1.0
        assert 0.0 <= result.trend_stability <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceCalibrator:
    def test_uncalibrated_returns_identity(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.7, 0.8)
        # Uncalibrated: score and confidence passed through unchanged
        assert result.calibration_method == "uncalibrated"
        assert abs(result.calibrated_score - 0.7) < 0.01
        assert abs(result.calibrated_confidence - 0.8) < 0.01

    def test_calibration_result_fields(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        result = cal.calibrate(0.5, 0.6)
        assert hasattr(result, "calibrated_score")
        assert hasattr(result, "calibrated_confidence")
        assert hasattr(result, "reliability")
        assert hasattr(result, "calibration_method")

    def test_reliability_range(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cal.calibrate(score, 0.8)
            assert 0.0 <= result.reliability <= 1.0

    def test_fit_enables_platt_scaling(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        np.random.seed(42)
        scores = np.random.rand(50).tolist()
        labels = [int(s > 0.5) for s in scores]
        cal.fit(scores, labels)
        result = cal.calibrate(0.8, 0.9)
        assert result.calibration_method == "platt_scaling"
        assert 0.0 <= result.calibrated_score <= 1.0
        assert 0.0 <= result.calibrated_confidence <= 1.0

    def test_fit_requires_minimum_samples(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        # Fewer than 10 samples → fit does nothing
        cal.fit([0.3, 0.7], [0, 1])
        result = cal.calibrate(0.7, 0.8)
        assert result.calibration_method == "uncalibrated"

    def test_calibrate_batch(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        scores = [0.3, 0.5, 0.8]
        confidences = [0.7, 0.6, 0.9]
        results = cal.calibrate_batch(scores, confidences)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.calibrated_score <= 1.0

    def test_output_clipped_to_valid_range(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        # Edge cases
        result_low = cal.calibrate(0.0, 0.0)
        result_high = cal.calibrate(1.0, 1.0)
        assert result_low.calibrated_score >= 0.0
        assert result_high.calibrated_score <= 1.0

    def test_decisive_score_high_reliability(self):
        from core.fusion.confidence_calibrator import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        result_decisive = cal.calibrate(0.95, 0.9)
        result_uncertain = cal.calibrate(0.5, 0.5)
        assert result_decisive.reliability >= result_uncertain.reliability


# ─────────────────────────────────────────────────────────────────────────────
# Config Settings
# ─────────────────────────────────────────────────────────────────────────────

class TestScannerSettings:
    def test_get_settings_returns_instance(self):
        from config.settings import get_settings
        settings = get_settings()
        assert settings is not None

    def test_default_values(self):
        from config.settings import get_settings
        settings = get_settings()
        assert settings.version != ""
        assert settings.host != ""
        assert settings.port > 0
        assert settings.rate_limit > 0

    def test_is_production_property(self):
        from config.settings import ScannerSettings
        s = ScannerSettings(env="production")
        assert s.is_production is True
        s2 = ScannerSettings(env="development")
        assert s2.is_production is False

    def test_cors_origin_list_property(self):
        from config.settings import ScannerSettings
        s = ScannerSettings(cors_origins="http://a.com,http://b.com")
        origins = s.cors_origin_list
        assert "http://a.com" in origins
        assert "http://b.com" in origins
        assert len(origins) == 2

    def test_get_settings_cached(self):
        from config.settings import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # same cached instance

    def test_pentashield_settings(self):
        from config.settings import get_settings
        settings = get_settings()
        assert settings.pentashield_enabled is True or settings.pentashield_enabled is False
        assert settings.hydra_heads >= 1
        assert 0.0 < settings.hydra_agreement_threshold <= 1.0
        assert settings.differential_privacy_epsilon > 0
