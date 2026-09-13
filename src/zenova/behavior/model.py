"""Baseline behavioral profile model for individual users.

Maintains individual statistical distributions (mean, std, min, max, count)
for passive sensing features and provides calibrated deviation scoring (Z-scores).
Implements cold-start protection to avoid premature deviation alerts before
sufficient baseline data is accumulated.
"""
import math
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from zenova.core.logging import get_logger

logger = get_logger("zenova.behavior.model")


class BehavioralFeatureDistribution(BaseModel):
    """Statistical summary for a single behavioral feature."""
    feature_name: str
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def update(self, new_val: float) -> None:
        """Update distribution with a new observation using Welford's algorithm."""
        self.count += 1
        if self.count == 1:
            self.mean = new_val
            self.variance = 0.0
            self.min_val = new_val
            self.max_val = new_val
        else:
            delta = new_val - self.mean
            self.mean += delta / self.count
            delta2 = new_val - self.mean
            # Running sample variance
            self.variance = (self.variance * (self.count - 2) + delta * delta2) / (self.count - 1)
            self.min_val = min(self.min_val, new_val)
            self.max_val = max(self.max_val, new_val)

    def compute_z_score(self, val: float, min_std: float = 1e-4) -> float:
        """Compute standard Z-score against personal baseline."""
        if self.count < 2 or self.std < min_std:
            return 0.0
        return (val - self.mean) / max(self.std, min_std)


class UserBehavioralProfile(BaseModel):
    """User-specific longitudinal behavioral profile."""
    user_id: str
    observation_count: int = 0
    min_observations_for_baseline: int = 3
    distributions: Dict[str, BehavioralFeatureDistribution] = Field(default_factory=dict)

    @property
    def is_established(self) -> bool:
        return self.observation_count >= self.min_observations_for_baseline

    @property
    def confidence(self) -> float:
        if self.min_observations_for_baseline <= 0:
            return 1.0
        return min(1.0, round(self.observation_count / self.min_observations_for_baseline, 2))

    def update_with_features(self, features: Dict[str, float]) -> None:
        """Incorporate a new set of extracted features into the profile."""
        self.observation_count += 1
        for name, val in features.items():
            if name not in self.distributions:
                self.distributions[name] = BehavioralFeatureDistribution(feature_name=name)
            self.distributions[name].update(val)

    def get_z_scores(self, current_features: Dict[str, float]) -> Dict[str, float]:
        """Compute Z-scores for current features if baseline is established."""
        if not self.is_established:
            return {}
        scores = {}
        for name, val in current_features.items():
            if name in self.distributions:
                dist = self.distributions[name]
                scores[name] = round(dist.compute_z_score(val), 3)
        return scores
