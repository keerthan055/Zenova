"""Cost-sensitive TF-IDF baseline classifier for crisis and suicide risk detection."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from zenova.core.logging import get_logger
from zenova.schemas.standard import RiskLevel

logger = get_logger("zenova.risk.baseline")


class RiskTfidfBaseline:
    """Cost-sensitive TF-IDF + Logistic Regression baseline penalizing False Negatives on crisis classes."""

    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or [
            RiskLevel.LOW.value,
            RiskLevel.MODERATE.value,
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value
        ]
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        # Asymmetric cost-sensitive class weights penalizing False Negatives on High/Critical
        class_weights = {
            self.label2id[RiskLevel.LOW.value]: 1.0,
            self.label2id[RiskLevel.MODERATE.value]: 1.5,
            self.label2id[RiskLevel.HIGH.value]: 3.5,
            self.label2id[RiskLevel.CRITICAL.value]: 5.0
        }
        self.classifier = LogisticRegression(
            class_weight=class_weights,
            max_iter=500,
            random_state=42
        )
        self.is_trained = False

    def train(self, texts: List[str], target_labels: List[str]):
        logger.info(f"Training cost-sensitive TF-IDF Risk Baseline on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        y = np.array([self.label2id[lbl] for lbl in target_labels])
        self.classifier.fit(X, y)
        self.is_trained = True
        logger.info("TF-IDF Risk Baseline training complete.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability distribution over 4 risk tiers (n_samples, 4)."""
        if not self.is_trained:
            raise RuntimeError("Baseline classifier must be trained before inference.")
        X = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(X)

        # Handle cases where training subset had missing classes
        if probs.shape[1] < len(self.labels):
            full_probs = np.zeros((len(texts), len(self.labels)), dtype=float)
            for local_idx, cls_id in enumerate(self.classifier.classes_):
                full_probs[:, cls_id] = probs[:, local_idx]
            return full_probs
        return probs

    def predict(self, texts: List[str], crisis_threshold: float = 0.35) -> List[str]:
        """Predict risk levels with safety decision boundary prioritizing high recall."""
        probs = self.predict_proba(texts)
        high_idx = self.label2id[RiskLevel.HIGH.value]
        crit_idx = self.label2id[RiskLevel.CRITICAL.value]

        predictions = []
        for p in probs:
            # If probability of High or Critical exceeds safety threshold, escalate
            if p[crit_idx] >= 0.25:
                predictions.append(RiskLevel.CRITICAL.value)
            elif (p[high_idx] + p[crit_idx]) >= crisis_threshold:
                predictions.append(RiskLevel.HIGH.value)
            else:
                top_idx = int(np.argmax(p))
                predictions.append(self.id2label[top_idx])
        return predictions

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "is_trained": self.is_trained
        }
        ckpt_path = out / "tfidf_baseline.joblib"
        joblib.dump(dump_data, ckpt_path)
        logger.info(f"Saved TF-IDF Risk Baseline checkpoint to {ckpt_path}")

    @classmethod
    def load(cls, output_dir: str) -> "RiskTfidfBaseline":
        ckpt_path = Path(output_dir) / "tfidf_baseline.joblib"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No baseline checkpoint found at {ckpt_path}")
        data = joblib.load(ckpt_path)
        instance = cls(labels=data["labels"])
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.label2id = data["label2id"]
        instance.id2label = data["id2label"]
        instance.is_trained = data["is_trained"]
        logger.info(f"Loaded TF-IDF Risk Baseline from {ckpt_path}")
        return instance
