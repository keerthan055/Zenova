"""Factory for pluggable RAG embedding models."""
import os
from typing import Optional
from zenova.rag.embeddings.base import BaseEmbeddingModel
from zenova.rag.embeddings.dense_tfidf import TfidfDenseEmbedding
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.embeddings.factory")


def get_embedding_model(model_type: Optional[str] = None) -> BaseEmbeddingModel:
    """Resolve and instantiate configured embedding model."""
    name = (model_type or os.getenv("RAG_EMBEDDING_MODEL", "tfidf")).lower().strip()

    if name in ("tfidf", "dense_tfidf", "local", "default"):
        return TfidfDenseEmbedding()

    logger.warning(f"Embedding model '{name}' not directly available. Falling back to TfidfDenseEmbedding.")
    return TfidfDenseEmbedding()
