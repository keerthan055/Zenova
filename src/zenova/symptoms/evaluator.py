"""Multi-label evaluation metrics for Symptom Signal Identification."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    accuracy_score
)


class SymptomModelEvaluator:
    """Evaluates multi-label symptom identification models."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str]
    ) -> Dict[str, Any]:
        """Compute standard multi-label classification metrics.
        
        Args:
            y_true: Binary matrix of shape (n_samples, n_labels)
            y_pred: Binary prediction matrix of shape (n_samples, n_labels)
            labels: List of label names corresponding to columns
        """
        assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

        # Macro metrics (average across labels)
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Micro metrics (aggregate globally)
        micro_p = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        micro_r = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        # Exact match (subset accuracy) & Hamming Loss
        subset_acc = float(accuracy_score(y_true, y_pred))
        h_loss = float(hamming_loss(y_true, y_pred))

        # Per-label metrics
        per_label = {}
        for col_idx, lbl in enumerate(labels):
            yt_col = y_true[:, col_idx]
            yp_col = y_pred[:, col_idx]

            lp = float(precision_score(yt_col, yp_col, zero_division=0))
            lr = float(recall_score(yt_col, yp_col, zero_division=0))
            lf = float(f1_score(yt_col, yp_col, zero_division=0))
            supp = int(np.sum(yt_col))

            per_label[lbl] = {
                "precision": round(lp, 4),
                "recall": round(lr, 4),
                "f1": round(lf, 4),
                "support": supp
            }

        return {
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "subset_accuracy": round(subset_acc, 4),
            "hamming_loss": round(h_loss, 4),
            "per_label": per_label
        }
