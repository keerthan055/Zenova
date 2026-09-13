"""TF-IDF + LogisticRegression baseline support strategy classifier."""
import os
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.baseline")


class StrategyTfidfBaseline:
    """Baseline emotional support strategy classifier using n-gram TF-IDF and regularized multinomial logistic regression."""

    def __init__(self, taxonomy: Optional[StrategyTaxonomy] = None):
        self.taxonomy = taxonomy or StrategyTaxonomy.default()
        self.labels = list(self.taxonomy.strategies)
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Train baseline classifier on input context texts and target strategy labels."""
        logger.info(f"Training Strategy TF-IDF baseline on {len(texts)} samples across {len(self.labels)} classes...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        logger.info("Strategy TF-IDF baseline training complete.")

    def predict(self, texts: List[str]) -> List[str]:
        """Predict top-1 strategy label for each input text."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X = self.vectorizer.transform(texts)
        return list(self.classifier.predict(X))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict class probability distribution over all strategies."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        X = self.vectorizer.transform(texts)
        probs = self.classifier.predict_proba(X)
        classes = list(self.classifier.classes_)

        results = []
        for row in probs:
            row_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, row)}
            # Ensure all canonical taxonomy strategies exist in output dictionary
            for strat in self.labels:
                if strat not in row_dict:
                    row_dict[strat] = 0.0
            results.append(row_dict)
        return results

    def save(self, model_dir: str) -> None:
        """Save vectorizer and classifier to disk."""
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.labels,
            "taxonomy": self.taxonomy.to_dict(),
            "is_trained": self.is_trained
        }, p / "tfidf_baseline.joblib")
        logger.info(f"Saved Strategy TF-IDF baseline to {p / 'tfidf_baseline.joblib'}")

    @classmethod
    def load(cls, model_dir: str) -> "StrategyTfidfBaseline":
        """Load trained baseline from disk."""
        p = Path(model_dir) / "tfidf_baseline.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found at {p}")
        data = joblib.load(p)
        tax_data = data.get("taxonomy")
        taxonomy = StrategyTaxonomy(
            taxonomy_name=tax_data.get("taxonomy_name", "ESConv-8"),
            strategies=tax_data.get("strategies"),
            descriptions=tax_data.get("descriptions")
        ) if tax_data else StrategyTaxonomy.default()

        inst = cls(taxonomy=taxonomy)
        inst.vectorizer = data["vectorizer"]
        inst.classifier = data["classifier"]
        inst.labels = data["labels"]
        inst.is_trained = data["is_trained"]
        return inst
