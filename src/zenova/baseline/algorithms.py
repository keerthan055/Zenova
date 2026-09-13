"""Configurable baseline estimation algorithms for ZENOVA.

Provides:
- BaseBaselineAlgorithm (Interface)
- RollingWindowAlgorithm (Sliding window sample statistics)
- EWMAAlgorithm (Exponentially Weighted Moving Average with momentum variance)
- BayesianBaselineAlgorithm (Conjugate Normal-Inverse-Gamma prior updating)
- BaselineAlgorithmFactory (Dynamic algorithm factory)
"""
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from zenova.schemas.baseline import FeatureBaselineStats


class BaseBaselineAlgorithm(ABC):
    """Abstract baseline calculation algorithm."""

    @abstractmethod
    def compute_baseline(
        self,
        feature_name: str,
        feature_values: List[float],
        existing_stats: Optional[FeatureBaselineStats] = None
    ) -> FeatureBaselineStats:
        """Compute or update baseline statistics given historical values."""
        pass


class RollingWindowAlgorithm(BaseBaselineAlgorithm):
    """Computes baseline statistics using a fixed sliding window of observations."""

    def __init__(self, window_size: int = 30, min_variance_floor: float = 1e-4):
        self.window_size = window_size
        self.min_variance_floor = min_variance_floor

    def compute_baseline(
        self,
        feature_name: str,
        feature_values: List[float],
        existing_stats: Optional[FeatureBaselineStats] = None
    ) -> FeatureBaselineStats:
        if not feature_values:
            return FeatureBaselineStats(
                feature=feature_name,
                mean=0.0,
                std=math.sqrt(self.min_variance_floor),
                variance=self.min_variance_floor,
                min_val=0.0,
                max_val=0.0,
                count=0,
                last_value=0.0
            )

        window = feature_values[-self.window_size:]
        n = len(window)
        mean_val = sum(window) / n
        last_val = window[-1]
        min_val = min(window)
        max_val = max(window)

        if n > 1:
            var_val = sum((x - mean_val) ** 2 for x in window) / (n - 1)
        else:
            var_val = self.min_variance_floor

        var_val = max(var_val, self.min_variance_floor)
        std_val = math.sqrt(var_val)

        return FeatureBaselineStats(
            feature=feature_name,
            mean=round(mean_val, 4),
            std=round(std_val, 4),
            variance=round(var_val, 6),
            min_val=round(min_val, 4),
            max_val=round(max_val, 4),
            count=len(feature_values),
            last_value=round(last_val, 4),
            updated_at=datetime.now(timezone.utc)
        )


class EWMAAlgorithm(BaseBaselineAlgorithm):
    """Exponentially Weighted Moving Average (EWMA) tracking temporal momentum and variance."""

    def __init__(self, alpha: float = 0.15, min_variance_floor: float = 1e-4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EWMA alpha smoothing factor must be in (0.0, 1.0].")
        self.alpha = alpha
        self.min_variance_floor = min_variance_floor

    def compute_baseline(
        self,
        feature_name: str,
        feature_values: List[float],
        existing_stats: Optional[FeatureBaselineStats] = None
    ) -> FeatureBaselineStats:
        if not feature_values:
            return FeatureBaselineStats(
                feature=feature_name,
                mean=0.0,
                std=math.sqrt(self.min_variance_floor),
                variance=self.min_variance_floor,
                count=0,
                last_value=0.0
            )

        # Initialize from first observation
        ewma_mean = feature_values[0]
        ewma_var = self.min_variance_floor

        for x in feature_values[1:]:
            prev_mean = ewma_mean
            ewma_mean = self.alpha * x + (1.0 - self.alpha) * prev_mean
            # MacGregor & Harris (1993) EWMA variance update
            ewma_var = (1.0 - self.alpha) * (ewma_var + self.alpha * ((x - prev_mean) ** 2))

        ewma_var = max(ewma_var, self.min_variance_floor)
        ewma_std = math.sqrt(ewma_var)

        n = len(feature_values)
        return FeatureBaselineStats(
            feature=feature_name,
            mean=round(ewma_mean, 4),
            std=round(ewma_std, 4),
            variance=round(ewma_var, 6),
            min_val=round(min(feature_values), 4),
            max_val=round(max(feature_values), 4),
            count=n,
            last_value=round(feature_values[-1], 4),
            ewma_mean=round(ewma_mean, 4),
            ewma_variance=round(ewma_var, 6),
            updated_at=datetime.now(timezone.utc)
        )


class BayesianBaselineAlgorithm(BaseBaselineAlgorithm):
    """Conjugate Normal-Inverse-Gamma Bayesian baseline updating.
    
    Effectively prevents variance collapse and extreme estimates during early interactions
    by blending empirical user observations with an informed prior.
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_kappa: float = 2.0,     # Equivalent prior pseudo-observations for mean
        prior_alpha: float = 2.0,     # Shape parameter
        prior_beta: float = 0.08,     # Scale parameter (E[var] = beta / (alpha - 1) = 0.08)
        min_variance_floor: float = 1e-4
    ):
        self.prior_mean = prior_mean
        self.prior_kappa = prior_kappa
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_variance_floor = min_variance_floor

    def compute_baseline(
        self,
        feature_name: str,
        feature_values: List[float],
        existing_stats: Optional[FeatureBaselineStats] = None
    ) -> FeatureBaselineStats:
        n = len(feature_values)
        if n == 0:
            prior_var = self.prior_beta / max(self.prior_alpha - 1.0, 1e-4)
            return FeatureBaselineStats(
                feature=feature_name,
                mean=round(self.prior_mean, 4),
                std=round(math.sqrt(prior_var), 4),
                variance=round(prior_var, 6),
                count=0,
                last_value=0.0
            )

        sample_mean = sum(feature_values) / n
        sample_sum_sq_diff = sum((x - sample_mean) ** 2 for x in feature_values)

        # Posterior update equations for Normal-Inverse-Gamma conjugate model
        kappa_n = self.prior_kappa + n
        alpha_n = self.prior_alpha + (n / 2.0)
        mean_n = (self.prior_kappa * self.prior_mean + n * sample_mean) / kappa_n
        beta_n = (
            self.prior_beta
            + 0.5 * sample_sum_sq_diff
            + (self.prior_kappa * n * ((sample_mean - self.prior_mean) ** 2)) / (2.0 * kappa_n)
        )

        post_var = beta_n / max(alpha_n - 1.0, 1e-4)
        post_var = max(post_var, self.min_variance_floor)
        post_std = math.sqrt(post_var)

        return FeatureBaselineStats(
            feature=feature_name,
            mean=round(mean_n, 4),
            std=round(post_std, 4),
            variance=round(post_var, 6),
            min_val=round(min(feature_values), 4),
            max_val=round(max(feature_values), 4),
            count=n,
            last_value=round(feature_values[-1], 4),
            updated_at=datetime.now(timezone.utc)
        )


class BaselineAlgorithmFactory:
    """Factory for selecting and instantiating baseline algorithms."""

    _REGISTRY = {
        "rolling_window": RollingWindowAlgorithm,
        "rolling": RollingWindowAlgorithm,
        "ewma": EWMAAlgorithm,
        "bayesian": BayesianBaselineAlgorithm
    }

    @classmethod
    def create(cls, algorithm_name: str = "rolling_window", **kwargs) -> BaseBaselineAlgorithm:
        algo_cls = cls._REGISTRY.get(algorithm_name.lower())
        if not algo_cls:
            valid = list(cls._REGISTRY.keys())
            raise ValueError(f"Unknown baseline algorithm '{algorithm_name}'. Valid: {valid}")
        return algo_cls(**kwargs)
