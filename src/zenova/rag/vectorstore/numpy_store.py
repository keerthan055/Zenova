"""In-Memory NumPy vector store using exact normalized cosine search."""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from zenova.schemas.rag import KnowledgeChunk
from zenova.rag.vectorstore.base import BaseVectorStore
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.vectorstore.numpy_store")


class InMemoryVectorStore(BaseVectorStore):
    """Pure NumPy vector store using normalized matrix dot products for exact cosine search.

    Completely portable across platforms, zero cloud dependencies, and fast for
    thousands of curated clinical chunks.
    """

    def __init__(self):
        self.chunks: List[KnowledgeChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[KnowledgeChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})")

        new_mat = np.array(embeddings, dtype=np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_mat = new_mat / norms

        if self.embeddings is None:
            self.embeddings = new_mat
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, new_mat])
            self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to InMemoryVectorStore. Total indexed: {len(self.chunks)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[KnowledgeChunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Matrix dot product computes cosine similarity directly
        scores = np.dot(self.embeddings, q_vec)

        # Sort descending
        top_indices = np.argsort(scores)[::-1]

        results: List[Tuple[KnowledgeChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            results.append((self.chunks[idx], round(score, 4)))
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else []
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"Saved {len(self.chunks)} vector store chunks to {p}")

    def load(self, filepath: str) -> None:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Vector store index not found at {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [KnowledgeChunk(**c) for c in data.get("chunks", [])]
        raw_embeds = data.get("embeddings", [])
        if raw_embeds:
            self.embeddings = np.array(raw_embeds, dtype=np.float32)
        else:
            self.embeddings = None
        logger.info(f"Loaded {len(self.chunks)} vector store chunks from {p}")

    def count(self) -> int:
        return len(self.chunks)
