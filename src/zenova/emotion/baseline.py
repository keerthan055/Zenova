"""TF-IDF + LogisticRegression baseline emotion classifier."""
import os
import joblib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from zenova.core.logging import get_logger

logger = get_logger("zenova.emotion.baseline")


class EmotionTfidfBaseline:
    """Baseline model combining n-gram TF-IDF representations with regularized logistic regression."""

    def __init__(self, labels: List[str]):
        self.labels = labels
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]) -> None:
        logger.info(f"Training TF-IDF baseline on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        logger.info("TF-IDF baseline training complete.")

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X = self.vectorizer.transform(texts)
        return list(self.classifier.predict(X))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(X)
        classes = list(self.classifier.classes_)

        results = []
        for row in probs:
            results.append({cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, row)})
        return results

    def save(self, model_dir: str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.labels,
            "is_trained": self.is_trained
        }, p / "tfidf_baseline.joblib")
        logger.info(f"Saved TF-IDF baseline to {p / 'tfidf_baseline.joblib'}")

    @classmethod
    def load(cls, model_dir: str) -> "EmotionTfidfBaseline":
        p = Path(model_dir) / "tfidf_baseline.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found at {p}")
        data = joblib.load(p)
        inst = cls(labels=data["labels"])
        inst.vectorizer = data["vectorizer"]
        inst.classifier = data["classifier"]
        inst.is_trained = data["is_trained"]
        return inst
