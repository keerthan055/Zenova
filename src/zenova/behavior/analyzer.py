"""Concrete Behavioral and Passive-Sensing Analyzer for ZENOVA.

Implements BaseBehaviorAnalyzer with:
- Strict optionality: gracefully returns is_available=False if no behavioral telemetry exists.
- Ingestion of heterogeneous wearable/device data via WearableAdapterRegistry.
- Circadian 04:00 day-alignment and physiological bounds clamping via BehavioralPreprocessor.
- Feature extraction across physical activity, sleep, mobility, social audio, and phone usage.
- Longitudinal statistical baseline tracking via UserBehavioralProfile.
- Anomaly evaluation enforcing the single-feature non-inference constraint and observational disclaimers.
"""
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone

from zenova.core.interfaces import BaseBehaviorAnalyzer
from zenova.schemas.standard import UserInput, BehavioralResult, ConversationContext
from zenova.schemas.behavior import (
    BehavioralObservation,
    BehavioralAnomalyReport,
    ActivityLevel,
    DeviceType,
)
from zenova.behavior.preprocessor import BehavioralPreprocessor
from zenova.behavior.extractor import BehavioralFeatureExtractor
from zenova.behavior.model import UserBehavioralProfile
from zenova.behavior.anomaly import BehavioralAnomalyDetector
from zenova.behavior.wearable import WearableAdapterRegistry
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.analyzer")


def infer_activity_level(
    steps: Optional[int], walking_mins: Optional[float]
) -> Optional[str]:
    """Classify activity level into standard categories."""
    if steps is not None:
        if steps < 3000:
            return ActivityLevel.SEDENTARY.value
        elif steps < 6000:
            return ActivityLevel.LOW.value
        elif steps < 10000:
            return ActivityLevel.MODERATE.value
        else:
            return ActivityLevel.ACTIVE.value
    if walking_mins is not None:
        if walking_mins < 15.0:
            return ActivityLevel.SEDENTARY.value
        elif walking_mins < 30.0:
            return ActivityLevel.LOW.value
        elif walking_mins < 60.0:
            return ActivityLevel.MODERATE.value
        else:
            return ActivityLevel.ACTIVE.value
    return None


class BehavioralAnalyzer(BaseBehaviorAnalyzer):
    MODULE_NAME = "BehavioralAnalyzer"
    VERSION = "1.0.0"

    def __init__(
        self,
        min_observations_for_baseline: int = 3,
        z_threshold: float = 2.0,
        min_affected_domains: int = 2,
    ):
        self.preprocessor = BehavioralPreprocessor()
        self.extractor = BehavioralFeatureExtractor()
        self.anomaly_detector = BehavioralAnomalyDetector(
            z_threshold=z_threshold,
            min_affected_domains=min_affected_domains,
        )
        self.min_observations_for_baseline = min_observations_for_baseline
        # In-memory user profiles: user_id -> UserBehavioralProfile
        self.profiles: Dict[str, UserBehavioralProfile] = {}
        # User observation history: user_id -> List[BehavioralObservation]
        self.history: Dict[str, List[BehavioralObservation]] = {}

    def get_or_create_profile(self, user_id: str) -> UserBehavioralProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserBehavioralProfile(
                user_id=user_id,
                min_observations_for_baseline=self.min_observations_for_baseline,
            )
        return self.profiles[user_id]

    def record_observation(
        self, observation: BehavioralObservation
    ) -> BehavioralAnomalyReport:
        """Process and register a single observation for a user."""
        clamped = self.preprocessor.validate_and_clamp(observation)
        user_id = clamped.user_id

        self.history.setdefault(user_id, []).append(clamped)
        profile = self.get_or_create_profile(user_id)

        # Extract features for this observation
        features = self.extractor.extract_single_observation_features(clamped)

        # Anomaly detection against current baseline (before updating profile)
        report = self.anomaly_detector.detect(user_id, features, profile)

        # Update profile baseline with the new observation
        profile.update_with_features(features)

        return report

    def record_raw_payload(
        self, payload: Dict[str, Any], device_key: Optional[str] = None
    ) -> BehavioralAnomalyReport:
        """Parse raw device payload via adapter and register observation."""
        adapter = WearableAdapterRegistry.get_adapter(device_key)
        obs = adapter.parse_payload(payload)
        return self.record_observation(obs)

    def analyze(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None,
    ) -> BehavioralResult:
        """Analyze conversational turn for optional behavioral signals.
        
        Strict Optionality:
        If no behavioral telemetry is present in user_input.metadata or registered history,
        cleanly returns BehavioralResult(is_available=False).
        """
        user_id = user_input.user_id
        meta = user_input.metadata or {}

        # 1. Check if behavioral data was provided in inbound metadata
        behavior_payload = meta.get("behavior") or meta.get("behavioral_observation")
        device_key = meta.get("device_type")

        obs_to_evaluate: Optional[BehavioralObservation] = None

        if behavior_payload is not None:
            if isinstance(behavior_payload, BehavioralObservation):
                obs_to_evaluate = behavior_payload
            elif isinstance(behavior_payload, dict):
                # Ensure user_id is populated
                if "user_id" not in behavior_payload:
                    behavior_payload["user_id"] = user_id
                adapter = WearableAdapterRegistry.get_adapter(device_key)
                obs_to_evaluate = adapter.parse_payload(behavior_payload)
            
            if obs_to_evaluate:
                report = self.record_observation(obs_to_evaluate)
        else:
            # 2. Check if user has historical observations recorded via API
            user_history = self.history.get(user_id, [])
            if user_history:
                obs_to_evaluate = user_history[-1]
                profile = self.get_or_create_profile(user_id)
                feats = self.extractor.extract_single_observation_features(obs_to_evaluate)
                report = self.anomaly_detector.detect(user_id, feats, profile)
            else:
                report = None

        # 3. If no behavioral data exists, return optional unavailable result
        if obs_to_evaluate is None:
            return BehavioralResult(
                is_placeholder=False,
                module_version=self.VERSION,
                is_available=False,
                activity_level=None,
                sleep_duration_hours=None,
                social_conversation_minutes=None,
                phone_screen_unlocks=None,
                mobility_radius_km=None,
                behavioral_anomaly_detected=False,
                anomaly_notes="No passive behavioral telemetry available for this interaction.",
                metrics={},
                disclaimer="Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses.",
                timestamp=datetime.now(timezone.utc),
            )

        # 4. Behavioral data available: populate full BehavioralResult
        features = self.extractor.extract_single_observation_features(obs_to_evaluate)
        activity = infer_activity_level(
            obs_to_evaluate.step_count, obs_to_evaluate.walking_minutes
        )

        anomaly_flag = report.is_anomaly if report else False
        notes = (
            report.clinical_notes
            if report
            else "Behavioral patterns within normal individual variance."
        )

        return BehavioralResult(
            is_placeholder=False,
            module_version=self.VERSION,
            is_available=True,
            activity_level=activity,
            sleep_duration_hours=obs_to_evaluate.sleep_duration_hours,
            social_conversation_minutes=obs_to_evaluate.conversation_duration_minutes,
            phone_screen_unlocks=obs_to_evaluate.screen_unlock_count,
            mobility_radius_km=obs_to_evaluate.mobility_radius_km,
            behavioral_anomaly_detected=anomaly_flag,
            anomaly_notes=notes,
            metrics=features,
            disclaimer=(
                report.disclaimer
                if report
                else "Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses."
            ),
            timestamp=datetime.now(timezone.utc),
        )
