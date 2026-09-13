"""Personal Baseline Engine for ZENOVA.

Compares an individual's current emotional, symptomatic, and behavioral state
against their own historical baseline rather than static population-level thresholds.

Strict Clinical Boundary:
All deviations are observational heuristics. Statistical deviations from personal baseline
do NOT constitute a clinical diagnosis.
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from zenova.core.interfaces import BaseBaselineEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    ConversationContext
)
from zenova.schemas.baseline import (
    UserBaselineProfile,
    BaselineDeviationReport,
    FeatureDeviation,
    MultimodalObservation,
    BaselineStatus,
    DeviationInterpretation
)
from zenova.baseline.extractor import FeatureExtractor
from zenova.baseline.algorithms import BaseBaselineAlgorithm, BaselineAlgorithmFactory
from zenova.baseline.storage import BaselineStorageManager
from zenova.core.logging import get_logger

logger = get_logger("zenova.baseline.engine")

CLINICAL_DISCLAIMER = (
    "Statistical deviation from personal baseline; observational and non-diagnostic. "
    "Does not constitute clinical psychiatric assessment."
)


class PersonalBaselineEngine(BaseBaselineEngine):
    """Personalized longitudinal baseline engine with configurable algorithms."""

    MODULE_NAME = "PersonalBaselineEngine"
    VERSION = "personal_baseline-v1.0.0"

    def __init__(
        self,
        algorithm: str = "rolling_window",
        min_observations: int = 5,
        established_threshold: int = 15,
        window_size: int = 30,
        alpha: float = 0.15,
        storage_manager: Optional[BaselineStorageManager] = None,
        **algo_kwargs
    ):
        self.min_observations = min_observations
        self.established_threshold = established_threshold
        self.algorithm_name = algorithm
        self.storage = storage_manager or BaselineStorageManager(persist_to_db=True)

        # Initialize algorithm via factory
        kwargs = {"window_size": window_size} if algorithm in ("rolling", "rolling_window") else {"alpha": alpha}
        kwargs.update(algo_kwargs)
        self.algorithm: BaseBaselineAlgorithm = BaselineAlgorithmFactory.create(algorithm, **kwargs)

    def _determine_status_and_confidence(self, n_observations: int) -> Tuple[str, float]:
        """Determine cold-start establishment status and confidence score."""
        if n_observations < self.min_observations:
            status = BaselineStatus.INSUFFICIENT_DATA.value
            confidence = round((n_observations / self.min_observations) * 0.4, 4)
        elif n_observations < self.established_threshold:
            status = BaselineStatus.PROVISIONAL_BASELINE.value
            progress = (n_observations - self.min_observations) / (self.established_threshold - self.min_observations)
            confidence = round(0.40 + progress * 0.35, 4)
        else:
            status = BaselineStatus.ESTABLISHED_BASELINE.value
            extra = min((n_observations - self.established_threshold) / 35.0, 1.0)
            confidence = round(0.75 + extra * 0.25, 4)
        return status, confidence

    def _compute_interpretation(self, z: float) -> str:
        """Map z-score deviation to standardized qualitative categorization."""
        if z >= 3.0:
            return DeviationInterpretation.SUBSTANTIALLY_ABOVE.value
        elif z >= 2.0:
            return DeviationInterpretation.NOTABLY_ABOVE.value
        elif z >= 1.0:
            return DeviationInterpretation.MODERATELY_ABOVE.value
        elif z <= -3.0:
            return DeviationInterpretation.SUBSTANTIALLY_BELOW.value
        elif z <= -2.0:
            return DeviationInterpretation.NOTABLY_BELOW.value
        elif z <= -1.0:
            return DeviationInterpretation.MODERATELY_BELOW.value
        else:
            return DeviationInterpretation.WITHIN_NORMAL.value

    def evaluate(
        self,
        user_input: UserInput,
        emotion: Optional[EmotionResult] = None,
        context: Optional[ConversationContext] = None,
        symptom: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        extra_features: Optional[Dict[str, Any]] = None
    ) -> BaselineResult:
        """Evaluate inbound conversational turn against personal baseline."""
        # 1. Extract features from current turn
        features = FeatureExtractor.extract_from_turn(
            user_input=user_input,
            emotion=emotion,
            symptom=symptom,
            risk=risk,
            context=context,
            extra_features=extra_features
        )

        user_id = user_input.user_id
        profile = self.storage.get_or_create_profile(user_id)
        history = self.storage.get_observations(user_id)
        total_obs = len(history)

        status, confidence = self._determine_status_and_confidence(total_obs)

        # 2. Compute deviations against established feature baselines
        metric_deviations: Dict[str, Dict[str, Any]] = {}
        deviating_features: List[str] = []
        is_significant = False
        z_valence = 0.0

        if total_obs >= 2:
            for feat_name, curr_val in features.items():
                stats = profile.feature_baselines.get(feat_name)
                if stats and stats.count >= 2:
                    mean_val = stats.mean
                    # Floor standard deviation to prevent infinite z-scores on invariant features
                    std_val = max(stats.std, 0.05)
                    z = round((curr_val - mean_val) / std_val, 2)
                    interp = self._compute_interpretation(z)

                    dev_entry = {
                        "feature": feat_name,
                        "current_value": round(curr_val, 4),
                        "baseline_mean": round(mean_val, 4),
                        "baseline_std": round(stats.std, 4),
                        "deviation": z,
                        "interpretation": interp
                    }
                    metric_deviations[feat_name] = dev_entry

                    if feat_name == "valence_score":
                        z_valence = z

                    # Flag significant deviation if |z| >= 2.0 on primary wellbeing indicators
                    if abs(z) >= 2.0:
                        deviating_features.append(feat_name)
                        if feat_name in ("valence_score", "anxiety_score", "sleep_disturbance_score", "depressive_score", "risk_severity"):
                            is_significant = True

        # Generate contextual note
        if total_obs < self.min_observations:
            deviation_notes = f"Cold start ({total_obs}/{self.min_observations} interactions): baseline establishing."
        elif is_significant:
            deviation_notes = f"Meaningful shift detected on: {', '.join(deviating_features)}."
        else:
            deviation_notes = "Current state is within normal individual variation."

        # 3. Update baseline profile with current observation
        self.storage.add_observation(
            user_id=user_id,
            features=features,
            session_id=user_input.session_id,
            turn_id=getattr(user_input, "turn_id", None)
        )

        # Update running baselines across all historical observations
        all_obs = self.storage.get_observations(user_id)
        for f_name in features.keys():
            all_vals = [obs["features"][f_name] for obs in all_obs if f_name in obs.get("features", {})]
            if all_vals:
                updated_stats = self.algorithm.compute_baseline(
                    feature_name=f_name,
                    feature_values=all_vals,
                    existing_stats=profile.feature_baselines.get(f_name)
                )
                profile.feature_baselines[f_name] = updated_stats

        new_status, new_conf = self._determine_status_and_confidence(len(all_obs))
        profile.total_interactions_recorded = len(all_obs)
        profile.status = new_status
        profile.confidence = new_conf
        if "valence_score" in profile.feature_baselines:
            profile.baseline_valence_mean = profile.feature_baselines["valence_score"].mean
            profile.baseline_valence_std = profile.feature_baselines["valence_score"].std
        self.storage.save_profile(profile)

        return BaselineResult(
            is_placeholder=False,
            module_version=self.VERSION,
            user_id=user_id,
            status=status,
            confidence=confidence,
            total_observations=total_obs,
            z_score_valence=z_valence,
            is_significant_deviation=is_significant,
            deviation_notes=deviation_notes,
            metric_deviations=metric_deviations,
            deviating_features=deviating_features,
            disclaimer=CLINICAL_DISCLAIMER,
            timestamp=datetime.now(timezone.utc)
        )

    def evaluate_observation(self, obs: MultimodalObservation) -> BaselineDeviationReport:
        """Evaluate an explicit MultimodalObservation (e.g. from API or survey)."""
        # Convert observation model to flat numerical feature dictionary
        features: Dict[str, float] = {}
        if obs.valence is not None:
            features["valence_score"] = float(obs.valence)
        if obs.arousal is not None:
            features["arousal_score"] = float(obs.arousal)
        if obs.word_count is not None:
            features["word_count"] = float(obs.word_count)
        if obs.risk_severity_score is not None:
            features["risk_severity"] = float(obs.risk_severity_score)
        if obs.ema_mood is not None:
            features["ema_mood"] = float(obs.ema_mood)
        if obs.phq9_score is not None:
            features["phq9_score"] = float(obs.phq9_score)
        if obs.gad7_score is not None:
            features["gad7_score"] = float(obs.gad7_score)
        if obs.sleep_hours is not None:
            features["sleep_hours"] = float(obs.sleep_hours)
        if obs.pitch_mean is not None:
            features["pitch_mean"] = float(obs.pitch_mean)
        if obs.jitter is not None:
            features["jitter"] = float(obs.jitter)

        for s_lbl, s_conf in obs.symptom_scores.items():
            features[f"symptom_{s_lbl}"] = float(s_conf)
            if "anxiety" in s_lbl:
                features["anxiety_score"] = float(s_conf)

        for k, v in obs.extra_features.items():
            features[k] = float(v)

        user_id = obs.user_id
        profile = self.storage.get_or_create_profile(user_id)
        history = self.storage.get_observations(user_id)
        total_obs = len(history)

        status, confidence = self._determine_status_and_confidence(total_obs)

        dev_list: List[FeatureDeviation] = []
        metric_devs: Dict[str, float] = {}
        deviating_feats: List[str] = []
        is_sig = False
        z_valence = 0.0

        if total_obs >= 2:
            for feat_name, curr_val in features.items():
                stats = profile.feature_baselines.get(feat_name)
                if stats and stats.count >= 2:
                    mean_val = stats.mean
                    std_val = max(stats.std, 0.05)
                    z = round((curr_val - mean_val) / std_val, 2)
                    interp = self._compute_interpretation(z)

                    dev_obj = FeatureDeviation(
                        feature=feat_name,
                        current_value=round(curr_val, 4),
                        baseline_mean=round(mean_val, 4),
                        baseline_std=round(stats.std, 4),
                        deviation=z,
                        interpretation=interp
                    )
                    dev_list.append(dev_obj)
                    metric_devs[feat_name] = z

                    if feat_name == "valence_score":
                        z_valence = z

                    if abs(z) >= 2.0:
                        deviating_feats.append(feat_name)
                        is_sig = True

        # Ingest and update baseline
        self.storage.add_observation(
            user_id=user_id,
            features=features,
            session_id=obs.session_id,
            turn_id=obs.turn_id,
            timestamp=obs.timestamp
        )

        all_obs = self.storage.get_observations(user_id)
        for f_name in features.keys():
            all_vals = [o["features"][f_name] for o in all_obs if f_name in o.get("features", {})]
            if all_vals:
                profile.feature_baselines[f_name] = self.algorithm.compute_baseline(
                    feature_name=f_name,
                    feature_values=all_vals,
                    existing_stats=profile.feature_baselines.get(f_name)
                )

        new_status, new_conf = self._determine_status_and_confidence(len(all_obs))
        profile.total_interactions_recorded = len(all_obs)
        profile.status = new_status
        profile.confidence = new_conf
        self.storage.save_profile(profile)

        return BaselineDeviationReport(
            user_id=user_id,
            status=status,
            confidence=confidence,
            total_observations=total_obs,
            z_score_valence=z_valence,
            is_significant_deviation=is_sig,
            deviation_notes=(
                f"Significant deviation on {', '.join(deviating_feats)}"
                if is_sig else "Within normal baseline variation."
            ),
            deviations=dev_list,
            metric_deviations=metric_devs,
            deviating_features=deviating_feats,
            disclaimer=CLINICAL_DISCLAIMER,
            timestamp=datetime.now(timezone.utc)
        )
