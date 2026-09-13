"""Factory for pluggable vector store backends."""
import os
from typing import Optional
from zenova.rag.vectorstore.base import BaseVectorStore
from zenova.rag.vectorstore.numpy_store import InMemoryVectorStore
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.vectorstore.factory")


def get_vector_store(store_type: Optional[str] = None) -> BaseVectorStore:
    """Resolve and instantiate configured vector store."""
    name = (store_type or os.getenv("RAG_VECTOR_STORE", "numpy")).lower().strip()

    if name in ("numpy", "memory", "inmemory", "default"):
        return InMemoryVectorStore()

    logger.warning(f"Vector store '{name}' not directly available. Falling back to InMemoryVectorStore.")
    return InMemoryVectorStore()
