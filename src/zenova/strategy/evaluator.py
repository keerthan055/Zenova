"""Evaluation suite for emotional support strategy classifiers."""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from zenova.strategy.taxonomy import StrategyTaxonomy


class StrategyEvaluator:
    """Computes comprehensive evaluation metrics for support strategy models."""

    def __init__(self, taxonomy: Optional[StrategyTaxonomy] = None):
        self.taxonomy = taxonomy or StrategyTaxonomy.default()
        self.labels = list(self.taxonomy.strategies)

    def evaluate(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_prob: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Compute accuracy, macro precision/recall/F1, per-class F1, and confusion matrix."""
        acc = float(accuracy_score(y_true, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average="macro", zero_division=0
        )

        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i, lbl in enumerate(self.labels):
            per_class_metrics[lbl] = {
                "precision": round(float(p_class[i]), 4),
                "recall": round(float(r_class[i]), 4),
                "f1": round(float(f1_class[i]), 4),
                "support": int(support[i])
            }

        cm = confusion_matrix(y_true, y_pred, labels=self.labels).tolist()

        result: Dict[str, Any] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm,
            "labels": self.labels,
            "total_samples": len(y_true)
        }

        # Expected Calibration Error (ECE) if probabilities are provided
        if y_prob is not None and len(y_prob) == len(y_true):
            result["ece"] = self.compute_ece(y_true, y_pred, y_prob)

        return result

    def compute_ece(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_prob: List[Dict[str, float]],
        num_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE) across prediction confidences."""
        confidences = []
        accuracies = []

        for true_label, pred_label, prob_dict in zip(y_true, y_pred, y_prob):
            conf = prob_dict.get(pred_label, 0.0)
            correct = 1.0 if true_label == pred_label else 0.0
            confidences.append(conf)
            accuracies.append(correct)

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return round(float(ece), 4)
