"""Behavioral data preprocessor for ZENOVA.

Handles:
- Circadian 04:00 day boundary alignment for passive sensing (preventing split nighttime sleep episodes).
- Range validation and physiological bounds clamping.
- Missing feature handling and imputation.
- Feature normalization and scaling.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from zenova.schemas.behavior import BehavioralObservation, DeviceType
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.preprocessor")

# Physiological upper/lower bounds for validation & clamping
BOUNDS = {
    "step_count": (0, 100000),
    "walking_minutes": (0.0, 1440.0),
    "running_minutes": (0.0, 1440.0),
    "sedentary_minutes": (0.0, 1440.0),
    "sleep_duration_hours": (0.0, 24.0),
    "sleep_disturbances_count": (0, 100),
    "mobility_radius_km": (0.0, 500.0),
    "location_entropy": (0.0, 5.0),
    "time_at_home_hours": (0.0, 24.0),
    "conversation_duration_minutes": (0.0, 1440.0),
    "conversation_count": (0, 500),
    "screen_unlock_count": (0, 1000),
    "screen_time_minutes": (0.0, 1440.0),
    "companion_interactions_count": (0, 200),
}


def get_circadian_day_key(dt: datetime, cutoff_hour: int = 4) -> str:
    """Map a timestamp to a circadian day key (YYYY-MM-DD) with a 04:00 boundary.
    
    Timestamps between 00:00 and 03:59 belong to the previous day's nocturnal cycle.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    shifted = dt - timedelta(hours=cutoff_hour)
    return shifted.strftime("%Y-%m-%d")


class BehavioralPreprocessor:
    """Preprocesses and validates raw behavioral observations."""

    def __init__(self, circadian_cutoff_hour: int = 4):
        self.circadian_cutoff_hour = circadian_cutoff_hour

    def validate_and_clamp_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp any physiologically extreme or erroneous values within valid ranges in a dictionary."""
        clamped_data = data.copy()
        for field_name, (min_val, max_val) in BOUNDS.items():
            val = clamped_data.get(field_name)
            if val is not None:
                if isinstance(min_val, int) and isinstance(max_val, int):
                    clamped = max(min_val, min(int(val), max_val))
                else:
                    clamped = max(float(min_val), min(float(val), float(max_val)))
                clamped_data[field_name] = clamped
        return clamped_data

    def validate_and_clamp(self, observation: BehavioralObservation) -> BehavioralObservation:
        """Clamp any physiologically extreme or erroneous values within valid ranges."""
        data = self.validate_and_clamp_dict(observation.model_dump())
        return BehavioralObservation(**data)

    def group_by_circadian_day(
        self, observations: List[BehavioralObservation]
    ) -> Dict[str, List[BehavioralObservation]]:
        """Group observations by their circadian 04:00 day."""
        days: Dict[str, List[BehavioralObservation]] = {}
        for obs in observations:
            key = get_circadian_day_key(obs.timestamp, cutoff_hour=self.circadian_cutoff_hour)
            days.setdefault(key, []).append(obs)
        return days

    def aggregate_day(self, observations: List[BehavioralObservation]) -> BehavioralObservation:
        """Aggregate multiple intraday observations into a single daily observation."""
        if not observations:
            raise ValueError("Cannot aggregate empty observation list.")
        if len(observations) == 1:
            return self.validate_and_clamp(observations[0])

        user_id = observations[0].user_id
        timestamp = observations[-1].timestamp
        device = observations[0].source_device

        # Sum cumulative counters
        def sum_field(field_name: str) -> Optional[float]:
            vals = [getattr(o, field_name) for o in observations if getattr(o, field_name) is not None]
            return sum(vals) if vals else None

        # Mean for rates/durations that represent a state
        def mean_field(field_name: str) -> Optional[float]:
            vals = [getattr(o, field_name) for o in observations if getattr(o, field_name) is not None]
            return (sum(vals) / len(vals)) if vals else None

        # Max for radius
        def max_field(field_name: str) -> Optional[float]:
            vals = [getattr(o, field_name) for o in observations if getattr(o, field_name) is not None]
            return max(vals) if vals else None

        steps = sum_field("step_count")
        walking = sum_field("walking_minutes")
        running = sum_field("running_minutes")
        sedentary = sum_field("sedentary_minutes")
        sleep_dur = max_field("sleep_duration_hours") or mean_field("sleep_duration_hours")
        sleep_dist = sum_field("sleep_disturbances_count")
        mobility_rad = max_field("mobility_radius_km")
        loc_ent = mean_field("location_entropy")
        time_home = mean_field("time_at_home_hours")
        conv_dur = sum_field("conversation_duration_minutes")
        conv_cnt = sum_field("conversation_count")
        screen_cnt = sum_field("screen_unlock_count")
        screen_dur = sum_field("screen_time_minutes")
        companion_cnt = sum_field("companion_interactions_count")

        # Merge extra metrics
        merged_extras: Dict[str, float] = {}
        for o in observations:
            merged_extras.update(o.extra_metrics)

        agg = BehavioralObservation(
            user_id=user_id,
            timestamp=timestamp,
            source_device=device,
            step_count=int(steps) if steps is not None else None,
            walking_minutes=walking,
            running_minutes=running,
            sedentary_minutes=sedentary,
            sleep_duration_hours=sleep_dur,
            sleep_disturbances_count=int(sleep_dist) if sleep_dist is not None else None,
            mobility_radius_km=mobility_rad,
            location_entropy=loc_ent,
            time_at_home_hours=time_home,
            conversation_duration_minutes=conv_dur,
            conversation_count=int(conv_cnt) if conv_cnt is not None else None,
            screen_unlock_count=int(screen_cnt) if screen_cnt is not None else None,
            screen_time_minutes=screen_dur,
            companion_interactions_count=int(companion_cnt) if companion_cnt is not None else None,
            extra_metrics=merged_extras,
        )
        return self.validate_and_clamp(agg)

    def preprocess_series(
        self, observations: List[BehavioralObservation]
    ) -> List[BehavioralObservation]:
        """Validate, align by circadian day, and produce aggregated daily observations sorted by timestamp."""
        if not observations:
            return []
        grouped = self.group_by_circadian_day(observations)
        daily_list: List[BehavioralObservation] = []
        for day_key in sorted(grouped.keys()):
            daily_obs = self.aggregate_day(grouped[day_key])
            daily_list.append(daily_obs)
        return daily_list
