"""Multi-label TF-IDF Baseline Classifier for Symptom Signal Identification."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from zenova.core.logging import get_logger

logger = get_logger("zenova.symptoms.baseline")


class SymptomTfidfBaseline:
    """Multi-label TF-IDF + Binary Relevance Logistic Regression baseline."""

    def __init__(self, labels: List[str]):
        self.labels = sorted(labels)
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.classifiers: Dict[int, Any] = {}
        self.constant_cols: Dict[int, float] = {}
        self.is_trained = False

    def _labels_to_binary_matrix(self, label_lists: List[List[str]]) -> np.ndarray:
        mat = np.zeros((len(label_lists), len(self.labels)), dtype=int)
        for row_idx, sample_labels in enumerate(label_lists):
            for lbl in sample_labels:
                if lbl in self.label2id:
                    mat[row_idx, self.label2id[lbl]] = 1
        return mat

    def train(self, texts: List[str], label_lists: List[List[str]]):
        logger.info(f"Training TF-IDF Baseline on {len(texts)} samples across {len(self.labels)} symptom classes...")
        X = self.vectorizer.fit_transform(texts)
        Y = self._labels_to_binary_matrix(label_lists)
        self.classifiers = {}
        self.constant_cols = {}

        for col_idx in range(len(self.labels)):
            y_col = Y[:, col_idx]
            unique_vals = np.unique(y_col)
            if len(unique_vals) <= 1:
                self.constant_cols[col_idx] = float(unique_vals[0]) if len(unique_vals) == 1 else 0.0
            else:
                clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
                clf.fit(X, y_col)
                self.classifiers[col_idx] = clf

        self.is_trained = True
        logger.info("TF-IDF Baseline training complete.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability matrix of shape (n_samples, n_labels)."""
        if not self.is_trained:
            raise RuntimeError("Baseline classifier must be trained before inference.")
        X = self.vectorizer.transform(texts)
        n_samples = len(texts)
        proba_mat = np.zeros((n_samples, len(self.labels)), dtype=float)

        for col_idx in range(len(self.labels)):
            if col_idx in self.constant_cols:
                proba_mat[:, col_idx] = self.constant_cols[col_idx]
            elif col_idx in self.classifiers:
                clf = self.classifiers[col_idx]
                col_probs = clf.predict_proba(X)
                if col_probs.shape[1] == 2:
                    proba_mat[:, col_idx] = col_probs[:, 1]
                else:
                    proba_mat[:, col_idx] = 0.0
        return proba_mat

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """Return binary prediction matrix (n_samples, n_labels)."""
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "vectorizer": self.vectorizer,
            "classifiers": self.classifiers,
            "constant_cols": self.constant_cols,
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "is_trained": self.is_trained
        }
        ckpt_path = out / "tfidf_baseline.joblib"
        joblib.dump(dump_data, ckpt_path)
        logger.info(f"Saved TF-IDF Baseline checkpoint to {ckpt_path}")

    @classmethod
    def load(cls, output_dir: str) -> "SymptomTfidfBaseline":
        ckpt_path = Path(output_dir) / "tfidf_baseline.joblib"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No baseline checkpoint found at {ckpt_path}")
        data = joblib.load(ckpt_path)
        instance = cls(labels=data["labels"])
        instance.vectorizer = data["vectorizer"]
        instance.classifiers = data.get("classifiers", {})
        instance.constant_cols = data.get("constant_cols", {})
        instance.label2id = data["label2id"]
        instance.id2label = data["id2label"]
        instance.is_trained = data["is_trained"]
        logger.info(f"Loaded TF-IDF Baseline from {ckpt_path}")
        return instance
