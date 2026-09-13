"""Unit tests for ZENOVA Personal Baseline Engine and Longitudinal Tracking."""
import pytest
import math
from datetime import datetime, timezone

from zenova.baseline.algorithms import (
    RollingWindowAlgorithm,
    EWMAAlgorithm,
    BayesianBaselineAlgorithm,
    BaselineAlgorithmFactory
)
from zenova.baseline.extractor import FeatureExtractor
from zenova.baseline.storage import BaselineStorageManager
from zenova.baseline.engine import PersonalBaselineEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    SymptomSignal,
    RiskResult,
    RiskLevel
)
from zenova.schemas.baseline import (
    MultimodalObservation,
    BaselineStatus,
    DeviationInterpretation
)


class TestBaselineAlgorithms:
    """Test mathematical correctness of baseline algorithms."""

    def test_rolling_window_algorithm(self):
        algo = RollingWindowAlgorithm(window_size=5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        stats = algo.compute_baseline("test_feature", values)

        # Last 5 values: [3.0, 4.0, 5.0, 6.0, 7.0] -> mean = 5.0
        assert stats.mean == 5.0
        assert stats.count == 7
        assert stats.min_val == 3.0
        assert stats.max_val == 7.0
        assert stats.last_value == 7.0
        assert stats.std > 0.0

    def test_rolling_window_single_observation(self):
        algo = RollingWindowAlgorithm(window_size=10)
        stats = algo.compute_baseline("single_val", [4.5])
        assert stats.mean == 4.5
        assert stats.count == 1
        assert stats.std > 0.0  # Uses variance floor

    def test_ewma_algorithm(self):
        algo = EWMAAlgorithm(alpha=0.5)
        values = [10.0, 20.0, 30.0]
        stats = algo.compute_baseline("ewma_feat", values)

        # Step 1: 10.0
        # Step 2: 0.5*20 + 0.5*10 = 15.0
        # Step 3: 0.5*30 + 0.5*15 = 22.5
        assert math.isclose(stats.mean, 22.5, abs_tol=1e-2)
        assert stats.ewma_mean is not None
        assert stats.ewma_variance is not None
        assert stats.count == 3

    def test_ewma_invalid_alpha(self):
        with pytest.raises(ValueError):
            EWMAAlgorithm(alpha=1.5)

    def test_bayesian_baseline_algorithm(self):
        algo = BayesianBaselineAlgorithm(prior_mean=0.0, prior_kappa=2.0)
        # Empty input should return prior
        prior_stats = algo.compute_baseline("bayes_feat", [])
        assert prior_stats.mean == 0.0

        # Observations with positive mean should pull posterior toward sample mean
        values = [2.0, 2.0, 2.0, 2.0]
        post_stats = algo.compute_baseline("bayes_feat", values)
        # Prior kappa=2 (weight 2), sample N=4 (weight 4): (2*0 + 4*2) / 6 = 8/6 = 1.3333
        assert math.isclose(post_stats.mean, 1.3333, abs_tol=1e-2)
        assert post_stats.count == 4

    def test_algorithm_factory(self):
        algo1 = BaselineAlgorithmFactory.create("rolling_window", window_size=15)
        assert isinstance(algo1, RollingWindowAlgorithm)
        assert algo1.window_size == 15

        algo2 = BaselineAlgorithmFactory.create("ewma", alpha=0.2)
        assert isinstance(algo2, EWMAAlgorithm)

        algo3 = BaselineAlgorithmFactory.create("bayesian")
        assert isinstance(algo3, BayesianBaselineAlgorithm)

        with pytest.raises(ValueError):
            BaselineAlgorithmFactory.create("invalid_algo_name")


class TestFeatureExtractor:
    """Test feature extraction and missing modality resilience."""

    def test_extract_full_multimodal_turn(self):
        u_input = UserInput(session_id="s1", user_id="u1", text="I am terrified and can't sleep.")
        em_res = EmotionResult(
            is_placeholder=False,
            primary_emotion="fear",
            confidence=0.85,
            probabilities={"fear": 0.85, "sadness": 0.15},
            valence=-0.70,
            arousal=0.80
        )
        sym_res = SymptomResult(
            is_placeholder=False,
            signals=[
                SymptomSignal(marker_name="sleep disturbance", confidence=0.92),
                SymptomSignal(marker_name="anxiety", confidence=0.88)
            ]
        )
        risk_res = RiskResult(
            is_placeholder=False,
            risk_level=RiskLevel.MODERATE,
            confidence=0.75
        )
        extra = {"pitch_mean": 210.5, "ema_mood": 2.0}

        features = FeatureExtractor.extract_from_turn(
            user_input=u_input,
            emotion=em_res,
            symptom=sym_res,
            risk=risk_res,
            extra_features=extra
        )

        assert "word_count" in features
        assert features["word_count"] == 6.0
        assert "valence_score" in features
        assert features["valence_score"] < 0.0  # Fear/sadness are negative valence
        assert "anxiety_score" in features
        assert features["anxiety_score"] == 0.88
        assert "sleep_disturbance_score" in features
        assert features["sleep_disturbance_score"] == 0.92
        assert "risk_severity" in features
        assert features["risk_severity"] == 1.0
        assert features["pitch_mean"] == 210.5
        assert features["ema_mood"] == 2.0

    def test_missing_modality_graceful_handling(self):
        """Verify feature extractor functions with text only and no extra modalities."""
        u_input = UserInput(session_id="s2", user_id="u2", text="Simple text input only.")
        features = FeatureExtractor.extract_from_turn(user_input=u_input)

        assert "word_count" in features
        assert "char_count" in features
        # Absent modalities should not be in dictionary and should not error
        assert "valence_score" not in features
        assert "anxiety_score" not in features
        assert "pitch_mean" not in features


class TestPersonalBaselineEngine:
    """Test baseline evaluation, cold-start staging, deviations, and disclaimers."""

    @pytest.fixture
    def engine(self):
        return PersonalBaselineEngine(min_observations=5, established_threshold=15)

    def test_cold_start_staging(self, engine):
        u_input = UserInput(session_id="s_cold", user_id="user_cold_test", text="Hello there.")

        # Observation 1
        res1 = engine.evaluate(u_input)
        assert res1.status == BaselineStatus.INSUFFICIENT_DATA.value
        assert res1.confidence < 0.40
        assert res1.total_observations == 0

        # Observations 2 to 4
        for _ in range(3):
            engine.evaluate(u_input)

        res4 = engine.evaluate(u_input)
        # Total observations is now 4 (< 5), still insufficient
        assert res4.status == BaselineStatus.INSUFFICIENT_DATA.value

        # Observation 5 reaches provisional threshold
        res5 = engine.evaluate(u_input)
        assert res5.status == BaselineStatus.PROVISIONAL_BASELINE.value
        assert res5.confidence >= 0.40

    def test_deviation_detection_and_interpretation(self, engine):
        user_id = "user_dev_test"

        # Establish baseline of 10 observations with anxiety_score ~ 0.20
        for _ in range(10):
            obs = MultimodalObservation(
                user_id=user_id,
                valence=0.10,
                symptom_scores={"anxiety": 0.20}
            )
            engine.evaluate_observation(obs)

        # High anxiety observation (anxiety = 0.85)
        high_anx_obs = MultimodalObservation(
            user_id=user_id,
            valence=-0.60,
            symptom_scores={"anxiety": 0.85}
        )
        report = engine.evaluate_observation(high_anx_obs)

        assert report.status == BaselineStatus.PROVISIONAL_BASELINE.value
        assert report.is_significant_deviation is True
        assert len(report.deviating_features) > 0

        # Find anxiety deviation
        anx_dev = next(d for d in report.deviations if "anxiety" in d.feature)
        assert anx_dev.deviation > 3.0
        assert anx_dev.interpretation == DeviationInterpretation.SUBSTANTIALLY_ABOVE.value

    def test_clinical_non_diagnostic_disclaimer(self, engine):
        u_input = UserInput(session_id="s_disc", user_id="user_disc_test", text="Testing disclaimer.")
        res = engine.evaluate(u_input)

        assert "non-diagnostic" in res.disclaimer.lower()
        assert "not constitute" in res.disclaimer.lower() or "statistical deviation" in res.disclaimer.lower()

    def test_reset_user_baseline(self, engine):
        user_id = "user_reset_test"
        obs = MultimodalObservation(user_id=user_id, valence=0.5)
        engine.evaluate_observation(obs)

        assert engine.storage.get_profile(user_id) is not None
        engine.storage.reset_user(user_id)
        assert engine.storage.get_profile(user_id) is None
        assert len(engine.storage.get_observations(user_id)) == 0
