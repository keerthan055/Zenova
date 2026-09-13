"""Base interface for pluggable embedding models."""
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract interface decoupling RAG from specific embedding models or providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for a batch of texts."""
        pass

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings as a 2D NumPy array."""
        return np.array(self.embed_batch(texts), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Generate normalized embedding as a 1D NumPy array."""
        return np.array(self.embed_text(text), dtype=np.float32)

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Serialize embedding parameters if stateful."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load embedding parameters from disk."""
        pass
