"""Comprehensive evaluation suite for crisis risk models with False-Negative Analysis."""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
from zenova.schemas.standard import RiskLevel


class RiskModelEvaluator:
    """Evaluator emphasizing Sensitivity (Recall) and False-Negative minimization on crisis classes."""

    @staticmethod
    def evaluate(
        y_true: List[str],
        y_pred: List[str],
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute standard classification metrics and safety-critical False Negative analysis."""
        target_labels = labels or [
            RiskLevel.LOW.value,
            RiskLevel.MODERATE.value,
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value
        ]

        acc = float(accuracy_score(y_true, y_pred))
        macro_p = float(precision_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=target_labels).tolist()

        # Per-class metrics
        per_class = {}
        for lbl in target_labels:
            yt_bin = [1 if y == lbl else 0 for y in y_true]
            yp_bin = [1 if y == lbl else 0 for y in y_pred]
            lp = float(precision_score(yt_bin, yp_bin, zero_division=0))
            lr = float(recall_score(yt_bin, yp_bin, zero_division=0))
            lf = float(f1_score(yt_bin, yp_bin, zero_division=0))
            supp = int(sum(yt_bin))

            per_class[lbl] = {
                "precision": round(lp, 4),
                "recall": round(lr, 4),
                "f1": round(lf, 4),
                "support": supp
            }

        # Specificity for 'low' risk: True Negatives / (True Negatives + False Positives)
        yt_low = np.array([1 if y == RiskLevel.LOW.value else 0 for y in y_true])
        yp_low = np.array([1 if y == RiskLevel.LOW.value else 0 for y in y_pred])
        tn_low = int(np.sum((yt_low == 0) & (yp_low == 0)))
        fp_low = int(np.sum((yt_low == 0) & (yp_low == 1)))
        specificity_low = float(tn_low / (tn_low + fp_low)) if (tn_low + fp_low) > 0 else 1.0

        # Safety Sensitivity on High/Critical (fraction of true high/critical detected as high/critical)
        high_crit_trues = [i for i, y in enumerate(y_true) if y in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)]
        high_crit_detected = sum(1 for i in high_crit_trues if y_pred[i] in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value))
        safety_sensitivity = float(high_crit_detected / len(high_crit_trues)) if high_crit_trues else 1.0

        # False Negative Analysis on High/Critical
        false_negatives = []
        for i in high_crit_trues:
            if y_pred[i] not in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
                sample_text = texts[i] if texts and i < len(texts) else "N/A"
                false_negatives.append({
                    "index": i,
                    "text": sample_text,
                    "true_risk": y_true[i],
                    "predicted_risk": y_pred[i]
                })

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "specificity_low_risk": round(specificity_low, 4),
            "safety_sensitivity_high_critical": round(safety_sensitivity, 4),
            "total_false_negatives_count": len(false_negatives),
            "false_negative_cases": false_negatives,
            "confusion_matrix": {
                "labels": target_labels,
                "matrix": cm
            },
            "per_class": per_class
        }
