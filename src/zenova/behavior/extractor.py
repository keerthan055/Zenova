"""Feature extraction from longitudinal behavioral and passive-sensing observations.

Extracts domain-specific features across:
1. Physical Activity (steps, active minutes, sedentary ratio)
2. Sleep Patterns (duration, regularity/variance, disturbances)
3. Mobility & Space (mobility radius, time at home, location entropy)
4. Social / Audio (conversation duration, conversation frequency)
5. Phone Interaction (screen unlocks, screen-on duration)
"""
import math
from typing import List, Dict, Any, Optional
from zenova.schemas.behavior import BehavioralObservation
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.extractor")


class BehavioralFeatureExtractor:
    """Extracts summary and longitudinal statistical features from behavioral observations."""

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    @staticmethod
    def _std(values: List[float]) -> Optional[float]:
        if len(values) < 2:
            return 0.0 if values else None
        m = sum(values) / len(values)
        variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def extract_features(
        self, observations: List[BehavioralObservation]
    ) -> Dict[str, float]:
        """Extract multi-domain aggregated features from an observation series.
        
        Returns a dictionary of feature names to float values.
        Missing features (where no observation reported that signal) are omitted.
        """
        if not observations:
            return {}

        features: Dict[str, float] = {}

        # 1. Physical Activity
        steps = [float(o.step_count) for o in observations if o.step_count is not None]
        walking = [o.walking_minutes for o in observations if o.walking_minutes is not None]
        running = [o.running_minutes for o in observations if o.running_minutes is not None]
        sedentary = [o.sedentary_minutes for o in observations if o.sedentary_minutes is not None]

        if steps:
            features["mean_daily_steps"] = round(self._mean(steps), 2)
            features["std_daily_steps"] = round(self._std(steps), 2)

        active_mins = []
        for o in observations:
            w = o.walking_minutes or 0.0
            r = o.running_minutes or 0.0
            if o.walking_minutes is not None or o.running_minutes is not None:
                active_mins.append(w + r)
        if active_mins:
            features["mean_active_minutes"] = round(self._mean(active_mins), 2)

        if sedentary:
            features["mean_sedentary_minutes"] = round(self._mean(sedentary), 2)
            if active_mins and sum(active_mins) + sum(sedentary) > 0:
                tot_tracked = sum(active_mins) + sum(sedentary)
                features["sedentary_ratio"] = round(sum(sedentary) / tot_tracked, 3)

        # 2. Sleep Proxies
        sleep_hrs = [o.sleep_duration_hours for o in observations if o.sleep_duration_hours is not None]
        sleep_dists = [float(o.sleep_disturbances_count) for o in observations if o.sleep_disturbances_count is not None]

        if sleep_hrs:
            features["mean_sleep_hours"] = round(self._mean(sleep_hrs), 2)
            features["sleep_hours_std"] = round(self._std(sleep_hrs), 2)

        if sleep_dists:
            features["mean_sleep_disturbances"] = round(self._mean(sleep_dists), 2)

        # 3. Mobility & Location-derived
        radii = [o.mobility_radius_km for o in observations if o.mobility_radius_km is not None]
        home_hrs = [o.time_at_home_hours for o in observations if o.time_at_home_hours is not None]
        entropies = [o.location_entropy for o in observations if o.location_entropy is not None]

        if radii:
            features["mean_mobility_radius_km"] = round(self._mean(radii), 2)
        if home_hrs:
            features["mean_time_at_home_hours"] = round(self._mean(home_hrs), 2)
        if entropies:
            features["mean_location_entropy"] = round(self._mean(entropies), 3)

        # 4. Social Communication
        conv_dur = [o.conversation_duration_minutes for o in observations if o.conversation_duration_minutes is not None]
        conv_cnt = [float(o.conversation_count) for o in observations if o.conversation_count is not None]

        if conv_dur:
            features["mean_conversation_duration_minutes"] = round(self._mean(conv_dur), 2)
        if conv_cnt:
            features["mean_conversation_count"] = round(self._mean(conv_cnt), 2)

        # 5. Phone / Screen Interaction
        unlocks = [float(o.screen_unlock_count) for o in observations if o.screen_unlock_count is not None]
        screen_mins = [o.screen_time_minutes for o in observations if o.screen_time_minutes is not None]

        if unlocks:
            features["mean_screen_unlocks"] = round(self._mean(unlocks), 2)
        if screen_mins:
            features["mean_screen_time_minutes"] = round(self._mean(screen_mins), 2)

        # 6. Companion app interaction
        companion_cnt = [float(o.companion_interactions_count) for o in observations if o.companion_interactions_count is not None]
        if companion_cnt:
            features["mean_companion_interactions"] = round(self._mean(companion_cnt), 2)

        # 7. Additional extra metrics
        all_extra_keys = set()
        for o in observations:
            all_extra_keys.update(o.extra_metrics.keys())
        for k in all_extra_keys:
            vals = [o.extra_metrics[k] for o in observations if k in o.extra_metrics]
            if vals:
                features[f"mean_{k}"] = round(self._mean(vals), 2)

        return features

    def extract_single_observation_features(
        self, observation: BehavioralObservation
    ) -> Dict[str, float]:
        """Extract point-in-time features from a single observation."""
        return self.extract_features([observation])
