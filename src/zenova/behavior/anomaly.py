"""Behavioral anomaly and multi-feature disruption detector.

CRITICAL CLINICAL BOUNDARIES:
1. SINGLE-FEATURE CONSTRAINT: Never infer sensitive clinical conclusions
   (e.g., depressive episode, social withdrawal crisis) directly from an isolated
   behavioral feature (e.g., just low steps on a rainy day). Multi-feature
   convergence across at least two distinct behavioral domains is strictly required
   to flag an anomaly.
2. OBSERVATIONAL DISCLAIMER: All behavioral signals are strictly observational
   proxies, never diagnostic claims.
"""
from typing import Dict, List, Set, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from zenova.schemas.behavior import BehavioralAnomalyReport
from zenova.behavior.model import UserBehavioralProfile
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.anomaly")

# Feature to Domain Taxonomy Mapping
FEATURE_DOMAINS: Dict[str, str] = {
    # Physical Activity
    "mean_daily_steps": "activity",
    "std_daily_steps": "activity",
    "mean_active_minutes": "activity",
    "mean_sedentary_minutes": "activity",
    "sedentary_ratio": "activity",
    # Sleep Proxies
    "mean_sleep_hours": "sleep",
    "sleep_hours_std": "sleep",
    "mean_sleep_disturbances": "sleep",
    # Mobility
    "mean_mobility_radius_km": "mobility",
    "mean_time_at_home_hours": "mobility",
    "mean_location_entropy": "mobility",
    # Social / Communication
    "mean_conversation_duration_minutes": "social",
    "mean_conversation_count": "social",
    # Digital Engagement
    "mean_screen_unlocks": "digital",
    "mean_screen_time_minutes": "digital",
    "mean_companion_interactions": "digital",
}


class BehavioralAnomalyDetector:
    """Evaluates multi-feature behavioral convergence and detects disruptions."""

    def __init__(
        self,
        z_threshold: float = 2.0,
        min_affected_domains: int = 2,
    ):
        """
        Args:
            z_threshold: Absolute Z-score threshold to consider an individual feature anomalous (|z| >= z_threshold).
            min_affected_domains: Minimum number of independent domains that must simultaneously
                                  exceed threshold before an anomaly is declared (default 2, enforcing
                                  the single-feature non-inference constraint).
        """
        self.z_threshold = z_threshold
        self.min_affected_domains = min_affected_domains

    def detect(
        self,
        user_id: str,
        current_features: Dict[str, float],
        profile: UserBehavioralProfile,
    ) -> BehavioralAnomalyReport:
        """Evaluate current features against personal baseline profile.
        
        Enforces:
        - Cold start safety (no anomaly alerts during calibration).
        - Multi-feature convergence (no single-feature anomaly triggers).
        """
        disclaimer = (
            "Passive behavioral signals are observational proxies; "
            "do not interpret as clinical diagnoses."
        )

        # 1. Cold-start check
        if not profile.is_established:
            return BehavioralAnomalyReport(
                user_id=user_id,
                is_anomaly=False,
                disruption_index=0.0,
                anomalous_features=[],
                feature_z_scores={},
                observation_count=profile.observation_count,
                clinical_notes=(
                    f"Baseline calibration in progress ({profile.observation_count}/"
                    f"{profile.min_observations_for_baseline} observations). "
                    "Insufficient longitudinal baseline to assess behavioral deviation."
                ),
                disclaimer=disclaimer,
                timestamp=datetime.now(timezone.utc),
            )

        # 2. Compute Z-scores for all current features
        z_scores = profile.get_z_scores(current_features)

        # 3. Identify features with |z| >= z_threshold and map to domains
        anomalous_features: List[str] = []
        affected_domains: Set[str] = set()
        domain_max_z: Dict[str, float] = {}

        for feat, z in z_scores.items():
            if abs(z) >= self.z_threshold:
                anomalous_features.append(feat)
                domain = FEATURE_DOMAINS.get(feat, "other")
                affected_domains.add(domain)
                abs_z = abs(z)
                if domain not in domain_max_z or abs_z > domain_max_z[domain]:
                    domain_max_z[domain] = abs_z

        # 4. Multi-domain convergence evaluation (SINGLE-FEATURE CONSTRAINT)
        # If fewer than min_affected_domains are involved, do NOT declare an anomaly.
        if len(affected_domains) >= self.min_affected_domains:
            is_anomaly = True
            # Composite Behavioral Disruption Index (BDI) is mean of peak |z| across affected domains
            disruption_index = round(sum(domain_max_z.values()) / len(domain_max_z), 2)
            domains_str = ", ".join(sorted(affected_domains))
            clinical_notes = (
                f"Multi-feature behavioral deviation detected across domains: [{domains_str}]. "
                f"Peak deviations: {', '.join(f'{k} (z={v:.2f})' for k, v in domain_max_z.items())}. "
                "Observational proxy variation only."
            )
        else:
            is_anomaly = False
            # If 1 feature/domain diverged, calculate nominal BDI without flagging anomaly
            if affected_domains:
                disruption_index = round(max(domain_max_z.values()) / 2.0, 2)
                clinical_notes = (
                    f"Isolated feature variation detected in domain(s): [{', '.join(affected_domains)}]. "
                    "Single-feature non-inference constraint active: no clinical anomaly declared "
                    "without multi-domain convergence."
                )
            else:
                disruption_index = 0.0
                clinical_notes = "Behavioral patterns within normal individual variance."

        return BehavioralAnomalyReport(
            user_id=user_id,
            is_anomaly=is_anomaly,
            disruption_index=disruption_index,
            anomalous_features=anomalous_features,
            feature_z_scores=z_scores,
            observation_count=profile.observation_count,
            clinical_notes=clinical_notes,
            disclaimer=disclaimer,
            timestamp=datetime.now(timezone.utc),
        )
