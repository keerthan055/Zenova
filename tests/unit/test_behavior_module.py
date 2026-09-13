"""Unit tests for ZENOVA Step 7 Behavioral / Passive-Sensing module."""
import pytest
from datetime import datetime, timezone, timedelta

from zenova.schemas.behavior import (
    BehavioralObservation,
    BehavioralAnomalyReport,
    DeviceType,
    ActivityLevel,
)
from zenova.schemas.standard import UserInput, BehavioralResult
from zenova.behavior.preprocessor import BehavioralPreprocessor, get_circadian_day_key
from zenova.behavior.extractor import BehavioralFeatureExtractor
from zenova.behavior.model import UserBehavioralProfile, BehavioralFeatureDistribution
from zenova.behavior.anomaly import BehavioralAnomalyDetector
from zenova.behavior.wearable import (
    SmartphonePassiveAdapter,
    AppleHealthKitAdapter,
    FitbitGoogleFitAdapter,
    GenericWearableAdapter,
    WearableAdapterRegistry,
    BaseWearableDeviceAdapter,
)
from zenova.behavior.placeholder import BehaviorPlaceholderAnalyzer
from zenova.behavior.analyzer import BehavioralAnalyzer, infer_activity_level


# ==============================================================================
# 1. WEARABLE ADAPTERS & REGISTRY TESTS
# ==============================================================================

def test_smartphone_passive_adapter():
    adapter = SmartphonePassiveAdapter()
    assert adapter.device_type == DeviceType.SMARTPHONE_PASSIVE

    payload = {
        "user_id": "usr_test_01",
        "timestamp": "2026-09-10T12:00:00+00:00",
        "step_count": 8200,
        "sleep_duration_hours": 7.5,
        "conversation_duration_minutes": 45.0,
    }
    assert adapter.validate_payload(payload) is True
    obs = adapter.parse_payload(payload)
    assert obs.user_id == "usr_test_01"
    assert obs.step_count == 8200
    assert obs.sleep_duration_hours == 7.5
    assert obs.source_device == DeviceType.SMARTPHONE_PASSIVE


def test_apple_healthkit_adapter():
    adapter = AppleHealthKitAdapter()
    assert adapter.device_type == DeviceType.APPLE_WATCH

    payload = {
        "user_id": "usr_apple",
        "healthKitData": {
            "HKQuantityTypeIdentifierStepCount": 6500,
            "HKQuantityTypeIdentifierAppleExerciseTime": 42.0,
            "sleepDurationHours": 6.8,
        }
    }
    obs = adapter.parse_payload(payload)
    assert obs.user_id == "usr_apple"
    assert obs.step_count == 6500
    assert obs.walking_minutes == 42.0
    assert obs.sleep_duration_hours == 6.8
    assert obs.source_device == DeviceType.APPLE_WATCH


def test_fitbit_adapter():
    adapter = FitbitGoogleFitAdapter()
    assert adapter.device_type == DeviceType.FITBIT

    payload = {
        "user_id": "usr_fitbit",
        "summary": {
            "steps": 9400,
            "sedentaryMinutes": 450.0,
        },
        "sleep": {
            "totalMinutesAsleep": 480,
        }
    }
    obs = adapter.parse_payload(payload)
    assert obs.user_id == "usr_fitbit"
    assert obs.step_count == 9400
    assert obs.sedentary_minutes == 450.0
    assert obs.sleep_duration_hours == 8.0


def test_wearable_adapter_registry():
    # Default fallback
    adapter = WearableAdapterRegistry.get_adapter()
    assert isinstance(adapter, GenericWearableAdapter)

    # Resolution by key
    hk = WearableAdapterRegistry.get_adapter("apple_watch")
    assert isinstance(hk, AppleHealthKitAdapter)

    fb = WearableAdapterRegistry.get_adapter("fitbit")
    assert isinstance(fb, FitbitGoogleFitAdapter)


# ==============================================================================
# 2. PREPROCESSOR & CIRCADIAN ALIGNMENT TESTS
# ==============================================================================

def test_circadian_day_key():
    # 02:30 AM on Sept 10 belongs to Sept 09 nocturnal period
    dt_early = datetime(2026, 9, 10, 2, 30, tzinfo=timezone.utc)
    key_early = get_circadian_day_key(dt_early, cutoff_hour=4)
    assert key_early == "2026-09-09"

    # 05:30 AM on Sept 10 belongs to Sept 10
    dt_day = datetime(2026, 9, 10, 5, 30, tzinfo=timezone.utc)
    key_day = get_circadian_day_key(dt_day, cutoff_hour=4)
    assert key_day == "2026-09-10"


def test_preprocessor_bounds_clamping():
    preprocessor = BehavioralPreprocessor()
    raw_dict = {
        "user_id": "usr_bounds",
        "step_count": 200000,  # exceeds max 100,000
        "sleep_duration_hours": 28.0,  # exceeds max 24.0
        "sedentary_minutes": -10.0,  # below min 0.0
    }
    clamped_dict = preprocessor.validate_and_clamp_dict(raw_dict)
    assert clamped_dict["step_count"] == 100000
    assert clamped_dict["sleep_duration_hours"] == 24.0
    assert clamped_dict["sedentary_minutes"] == 0.0

    obs = BehavioralObservation(**clamped_dict)
    assert obs.step_count == 100000
    assert obs.sleep_duration_hours == 24.0
    assert obs.sedentary_minutes == 0.0


def test_preprocessor_intraday_aggregation():
    preprocessor = BehavioralPreprocessor()
    obs1 = BehavioralObservation(
        user_id="usr_agg",
        timestamp=datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc),
        step_count=3000,
        walking_minutes=20.0,
        conversation_duration_minutes=15.0,
    )
    obs2 = BehavioralObservation(
        user_id="usr_agg",
        timestamp=datetime(2026, 9, 10, 16, 0, tzinfo=timezone.utc),
        step_count=4500,
        walking_minutes=25.0,
        conversation_duration_minutes=20.0,
    )
    aggregated = preprocessor.aggregate_day([obs1, obs2])
    assert aggregated.step_count == 7500
    assert aggregated.walking_minutes == 45.0
    assert aggregated.conversation_duration_minutes == 35.0


# ==============================================================================
# 3. FEATURE EXTRACTOR TESTS
# ==============================================================================

def test_feature_extractor():
    extractor = BehavioralFeatureExtractor()
    obs_list = [
        BehavioralObservation(
            user_id="u1",
            step_count=6000,
            walking_minutes=30.0,
            running_minutes=10.0,
            sedentary_minutes=480.0,
            sleep_duration_hours=7.0,
            mobility_radius_km=5.0,
            conversation_duration_minutes=60.0,
            screen_unlock_count=50,
        ),
        BehavioralObservation(
            user_id="u1",
            step_count=8000,
            walking_minutes=40.0,
            running_minutes=15.0,
            sedentary_minutes=420.0,
            sleep_duration_hours=8.0,
            mobility_radius_km=7.0,
            conversation_duration_minutes=80.0,
            screen_unlock_count=60,
        ),
    ]
    feats = extractor.extract_features(obs_list)
    assert feats["mean_daily_steps"] == 7000.0
    assert feats["mean_active_minutes"] == 47.5
    assert feats["mean_sleep_hours"] == 7.5
    assert feats["mean_mobility_radius_km"] == 6.0
    assert feats["mean_conversation_duration_minutes"] == 70.0
    assert feats["mean_screen_unlocks"] == 55.0


# ==============================================================================
# 4. BASELINE PROFILE & DISTRIBUTION TESTS
# ==============================================================================

def test_feature_distribution_welford():
    dist = BehavioralFeatureDistribution(feature_name="steps")
    # Values: 10, 20, 30 -> mean=20, variance=100, std=10
    dist.update(10.0)
    dist.update(20.0)
    dist.update(30.0)
    assert dist.count == 3
    assert dist.mean == pytest.approx(20.0)
    assert dist.std == pytest.approx(10.0)

    # Test Z-score
    z = dist.compute_z_score(50.0)
    assert z == pytest.approx(3.0)


def test_user_behavioral_profile_cold_start():
    profile = UserBehavioralProfile(user_id="u_cold", min_observations_for_baseline=3)
    assert not profile.is_established
    assert profile.confidence == 0.0

    profile.update_with_features({"steps": 5000.0})
    assert not profile.is_established
    assert profile.confidence == 0.33

    profile.update_with_features({"steps": 5500.0})
    profile.update_with_features({"steps": 6000.0})
    assert profile.is_established
    assert profile.confidence == 1.0


# ==============================================================================
# 5. ANOMALY DETECTOR & CLINICAL CONSTRAINT TESTS
# ==============================================================================

def test_single_feature_non_inference_constraint():
    """CRITICAL CLINICAL BOUNDARY TEST:
    An isolated metric deviation (e.g., low steps on a rainy day) MUST NOT
    trigger an anomaly without multi-feature convergence across >= 2 domains.
    """
    detector = BehavioralAnomalyDetector(z_threshold=2.0, min_affected_domains=2)
    profile = UserBehavioralProfile(user_id="u_rainy", min_observations_for_baseline=3)

    # Establish baseline
    for steps in [8000.0, 8500.0, 8200.0, 8300.0]:
        profile.update_with_features({
            "mean_daily_steps": steps,
            "mean_sleep_hours": 7.5,
            "mean_conversation_duration_minutes": 60.0,
        })
    assert profile.is_established

    # Single feature drop: steps drop massively (z < -3.0), but sleep and conversation remain normal
    rainy_day_features = {
        "mean_daily_steps": 2000.0,  # Extreme drop in activity domain
        "mean_sleep_hours": 7.5,    # Normal sleep
        "mean_conversation_duration_minutes": 60.0,  # Normal social
    }

    report = detector.detect("u_rainy", rainy_day_features, profile)

    # Must NOT declare anomaly due to single-feature constraint
    assert report.is_anomaly is False
    assert "mean_daily_steps" in report.anomalous_features
    assert "Single-feature non-inference constraint active" in report.clinical_notes
    assert "do not interpret as clinical diagnoses" in report.disclaimer


def test_multi_domain_disruption_convergence():
    """Multi-feature convergence test:
    Simultaneous deviation across Sleep AND Social domains triggers an anomaly.
    """
    detector = BehavioralAnomalyDetector(z_threshold=2.0, min_affected_domains=2)
    profile = UserBehavioralProfile(user_id="u_distress", min_observations_for_baseline=3)

    # Establish baseline
    for _ in range(5):
        profile.update_with_features({
            "mean_daily_steps": 8000.0,
            "mean_sleep_hours": 7.5,
            "mean_conversation_duration_minutes": 60.0,
        })
    # Add variation so std > 0
    profile.update_with_features({
        "mean_daily_steps": 8200.0,
        "mean_sleep_hours": 7.8,
        "mean_conversation_duration_minutes": 65.0,
    })
    profile.update_with_features({
        "mean_daily_steps": 7800.0,
        "mean_sleep_hours": 7.2,
        "mean_conversation_duration_minutes": 55.0,
    })

    # Convergent multi-domain collapse: sleep collapses AND social audio collapses
    disrupted_features = {
        "mean_daily_steps": 8000.0,
        "mean_sleep_hours": 3.5,  # Severe sleep drop (|z| > 2.0)
        "mean_conversation_duration_minutes": 10.0,  # Severe social audio drop (|z| > 2.0)
    }

    report = detector.detect("u_distress", disrupted_features, profile)
    assert report.is_anomaly is True
    assert report.disruption_index >= 2.0
    assert "Multi-feature behavioral deviation detected" in report.clinical_notes
    assert "sleep" in report.clinical_notes
    assert "social" in report.clinical_notes


# ==============================================================================
# 6. ANALYZER & PLACEHOLDER TESTS
# ==============================================================================

def test_behavior_placeholder_analyzer():
    analyzer = BehaviorPlaceholderAnalyzer()
    inp_without_meta = UserInput(session_id="s1", user_id="u1", text="Hello")
    res = analyzer.analyze(inp_without_meta)
    assert res.is_placeholder is True
    assert res.is_available is False

    inp_with_meta = UserInput(
        session_id="s1", user_id="u1", text="Hello", metadata={"behavior": {"steps": 5000}}
    )
    res_with_meta = analyzer.analyze(inp_with_meta)
    assert res_with_meta.is_placeholder is True
    assert res_with_meta.is_available is True


def test_behavioral_analyzer_strict_optionality():
    """Verify that when no behavioral data is provided, analyzer returns is_available=False cleanly."""
    analyzer = BehavioralAnalyzer()
    inp = UserInput(session_id="s_opt", user_id="u_opt", text="I feel a bit stressed today.")

    result = analyzer.analyze(inp)
    assert isinstance(result, BehavioralResult)
    assert result.is_available is False
    assert result.is_placeholder is False
    assert result.behavioral_anomaly_detected is False
    assert "No passive behavioral telemetry" in result.anomaly_notes


def test_behavioral_analyzer_with_metadata():
    """Verify analyzer processes inbound behavioral telemetry in user_input.metadata."""
    analyzer = BehavioralAnalyzer()
    inp = UserInput(
        session_id="s_telemetry",
        user_id="u_telemetry",
        text="Can we talk about my week?",
        metadata={
            "behavior": {
                "step_count": 10500,
                "sleep_duration_hours": 7.8,
                "conversation_duration_minutes": 55.0,
                "screen_unlock_count": 45,
            }
        }
    )

    result = analyzer.analyze(inp)
    assert result.is_available is True
    assert result.activity_level == ActivityLevel.ACTIVE.value
    assert result.sleep_duration_hours == 7.8
    assert result.social_conversation_minutes == 55.0
    assert result.phone_screen_unlocks == 45


def test_infer_activity_level():
    assert infer_activity_level(steps=1500, walking_mins=None) == ActivityLevel.SEDENTARY.value
    assert infer_activity_level(steps=4500, walking_mins=None) == ActivityLevel.LOW.value
    assert infer_activity_level(steps=7500, walking_mins=None) == ActivityLevel.MODERATE.value
    assert infer_activity_level(steps=12000, walking_mins=None) == ActivityLevel.ACTIVE.value
    assert infer_activity_level(steps=None, walking_mins=10.0) == ActivityLevel.SEDENTARY.value
    assert infer_activity_level(steps=None, walking_mins=25.0) == ActivityLevel.LOW.value
    assert infer_activity_level(steps=None, walking_mins=45.0) == ActivityLevel.MODERATE.value
    assert infer_activity_level(steps=None, walking_mins=75.0) == ActivityLevel.ACTIVE.value
    assert infer_activity_level(steps=None, walking_mins=None) is None
