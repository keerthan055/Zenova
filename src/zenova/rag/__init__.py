"""ZENOVA Retrieval-Augmented Generation (RAG) Subpackage.

Curated evidence retrieval, sliding-window chunking, pluggable dense embeddings,
in-memory vector storage, hybrid reranking, and anti-hallucination confidence gating.
"""
from zenova.schemas.rag import (
    SourceType,
    EvidenceTier,
    ReviewStatus,
    ReliabilityMetadata,
    KnowledgeSource,
    KnowledgeChunk,
    RAGSearchResult,
    RAGQueryResult
)
from zenova.rag.chunker import DocumentChunker
from zenova.rag.embeddings import (
    BaseEmbeddingModel,
    TfidfDenseEmbedding,
    get_embedding_model
)
from zenova.rag.vectorstore import (
    BaseVectorStore,
    InMemoryVectorStore,
    get_vector_store
)
from zenova.rag.reranker import HybridReranker
from zenova.rag.engine import KnowledgeRetrievalEngine
from zenova.rag.placeholder import RAGPlaceholderEngine

__all__ = [
    "SourceType",
    "EvidenceTier",
    "ReviewStatus",
    "ReliabilityMetadata",
    "KnowledgeSource",
    "KnowledgeChunk",
    "RAGSearchResult",
    "RAGQueryResult",
    "DocumentChunker",
    "BaseEmbeddingModel",
    "TfidfDenseEmbedding",
    "get_embedding_model",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    "HybridReranker",
    "KnowledgeRetrievalEngine",
    "RAGPlaceholderEngine",
]
