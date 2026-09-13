"""Base interface for pluggable vector databases."""
from abc import ABC, abstractmethod
from typing import List, Tuple
from zenova.schemas.rag import KnowledgeChunk


class BaseVectorStore(ABC):
    """Abstract vector store interface decoupling RAG from vector database implementations."""

    @abstractmethod
    def add_chunks(self, chunks: List[KnowledgeChunk], embeddings: List[List[float]]) -> None:
        """Add chunks and corresponding dense vector embeddings to index."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """Search vector database by dense cosine similarity."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist vector index to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load vector index from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total indexed passages."""
        pass
