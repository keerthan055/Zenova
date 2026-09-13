"""Evaluation suite for emotion classification models."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class EmotionModelEvaluator:
    """Computes research-grade classification metrics across emotion categories."""

    @staticmethod
    def evaluate(
        y_true: List[str],
        y_pred: List[str],
        labels: List[str]
    ) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))

        # Macro metrics
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )

        # Per-class metrics
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        per_class_metrics = {}
        for i, lbl in enumerate(labels):
            per_class_metrics[lbl] = {
                "precision": round(float(p_class[i]), 4),
                "recall": round(float(r_class[i]), 4),
                "f1": round(float(f1_class[i]), 4),
                "support": int(support[i])
            }

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": cm,
            "labels": labels
        }
