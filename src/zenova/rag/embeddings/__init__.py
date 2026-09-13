"""Pluggable embedding backends for ZENOVA RAG."""
from zenova.rag.embeddings.base import BaseEmbeddingModel
from zenova.rag.embeddings.dense_tfidf import TfidfDenseEmbedding
from zenova.rag.embeddings.factory import get_embedding_model

__all__ = [
    "BaseEmbeddingModel",
    "TfidfDenseEmbedding",
    "get_embedding_model",
]
