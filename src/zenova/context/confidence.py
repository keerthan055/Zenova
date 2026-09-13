"""Confidence Propagation Engine for ZENOVA Step 9.

Propagates, weights, and aggregates confidence scores across heterogeneous
modalities to compute a robust, unified contextual reliability index.
"""
from typing import Dict, Any, List, Optional


DEFAULT_MODALITY_WEIGHTS = {
    "risk": 1.0,        # Critical safety weighting
    "symptoms": 0.85,   # High clinical importance
    "emotion": 0.80,    # Text affect
    "baseline": 0.75,   # Longitudinal consistency
    "voice": 0.70,      # Acoustic affect proxy
    "behavior": 0.65,   # Passive sensor telemetry
}


class ConfidencePropagator:
    """Computes propagated confidence metrics across available modalities."""

    def __init__(self, domain_weights: Optional[Dict[str, float]] = None):
        self.weights = domain_weights or DEFAULT_MODALITY_WEIGHTS

    def compute_propagation(
        self,
        modality_confidences: Dict[str, float]
    ) -> Dict[str, Any]:
        """Aggregate confidence scores from present modalities.

        Args:
            modality_confidences: Mapping of modality name to float confidence in [0.0, 1.0].
                                  Only available/present modalities should be included.

        Returns:
            Dictionary with overall_context_confidence, min_modality_confidence,
            harmonic_mean_confidence, modality_confidences, and normalized_weights.
        """
        if not modality_confidences:
            return {
                "overall_context_confidence": 0.0,
                "min_modality_confidence": 0.0,
                "harmonic_mean_confidence": 0.0,
                "modality_confidences": {},
                "active_weights": {}
            }

        # Filter and clamp confidences between 0.0 and 1.0
        clamped = {
            m: max(0.0, min(1.0, float(c)))
            for m, c in modality_confidences.items()
        }

        # 1. Minimum Modality Confidence (Conservative Floor)
        min_conf = min(clamped.values())

        # 2. Weighted Average
        total_weight = 0.0
        weighted_sum = 0.0
        active_weights = {}

        for mod, conf in clamped.items():
            w = self.weights.get(mod, 0.5)
            total_weight += w
            weighted_sum += conf * w
            active_weights[mod] = w

        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
            normalized_weights = {k: round(v / total_weight, 4) for k, v in active_weights.items()}
        else:
            weighted_avg = sum(clamped.values()) / len(clamped)
            normalized_weights = {k: 1.0 / len(clamped) for k in clamped}

        # 3. Harmonic Mean (heavily penalizes any single uncertain modality)
        eps = 1e-4
        inverted_sum = sum(1.0 / max(eps, c) for c in clamped.values())
        harmonic_mean = len(clamped) / inverted_sum if inverted_sum > 0 else 0.0
        harmonic_mean = max(0.0, min(1.0, harmonic_mean))

        return {
            "overall_context_confidence": round(weighted_avg, 4),
            "min_modality_confidence": round(min_conf, 4),
            "harmonic_mean_confidence": round(harmonic_mean, 4),
            "modality_confidences": {k: round(v, 4) for k, v in clamped.items()},
            "active_weights": normalized_weights
        }
