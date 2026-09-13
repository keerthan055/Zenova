"""Dense TF-IDF and Latent Semantic Embedding model for zero-download offline execution."""
import os
import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from zenova.rag.embeddings.base import BaseEmbeddingModel
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.embeddings.dense_tfidf")


class TfidfDenseEmbedding(BaseEmbeddingModel):
    """Offline, deterministic dense embedding model using LSA / TF-IDF projection.

    Guarantees zero-network download, instant execution, and portable cosine similarity.
    """

    def __init__(self, dimension: int = 48):
        self._dim = dimension
        self._model_name = "dense-tfidf-lsa-v1"
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.svd: Optional[TruncatedSVD] = None
        self.is_fitted = False

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def fit(self, texts: List[str]) -> None:
        """Fit vocabulary and SVD projection on knowledge corpus."""
        logger.info(f"Fitting TfidfDenseEmbedding on {len(texts)} passages...")
        tfidf_mat = self.vectorizer.fit_transform(texts)
        actual_components = min(self._dim, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
        if actual_components < 2:
            actual_components = min(self._dim, tfidf_mat.shape[1])
            self.svd = None
            self._dim = tfidf_mat.shape[1]
        else:
            self.svd = TruncatedSVD(n_components=actual_components, random_state=42)
            self.svd.fit(tfidf_mat)
            self._dim = actual_components

        self.is_fitted = True
        logger.info(f"TfidfDenseEmbedding fitted with dimension={self._dim}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            # If not yet fitted, fit dynamically on batch
            self.fit(texts)

        tfidf_mat = self.vectorizer.transform(texts)
        if self.svd is not None:
            dense_mat = self.svd.transform(tfidf_mat)
        else:
            dense_mat = tfidf_mat.toarray()

        normed = normalize(dense_mat, norm="l2", axis=1)
        return normed.tolist()

    def save(self, filepath: str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "dim": self._dim,
            "is_fitted": self.is_fitted,
            "model_name": self._model_name
        }, p)
        logger.info(f"Saved TfidfDenseEmbedding model to {p}")

    def load(self, filepath: str) -> None:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Embedding checkpoint not found at {p}")
        data = joblib.load(p)
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]
        self._dim = data["dim"]
        self.is_fitted = data["is_fitted"]
        self._model_name = data.get("model_name", "dense-tfidf-lsa-v1")
        logger.info(f"Loaded TfidfDenseEmbedding model from {p}")
