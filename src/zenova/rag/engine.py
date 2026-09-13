"""Curated Knowledge Retrieval Engine with anti-hallucination gating and provenance tracking."""
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from zenova.schemas.rag import RAGQueryResult, RAGSearchResult
from zenova.rag.embeddings import BaseEmbeddingModel, get_embedding_model
from zenova.rag.vectorstore import BaseVectorStore, get_vector_store
from zenova.rag.reranker import HybridReranker
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.engine")


class KnowledgeRetrievalEngine:
    """Production RAG retrieval engine enforcing strict evidence confidence gating and provenance tracking."""

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        min_confidence_threshold: float = 0.30,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.min_confidence_threshold = min_confidence_threshold
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store()
        self.reranker = HybridReranker()
        self.is_initialized = False

        self._initialize()

    def _initialize(self) -> None:
        """Load persisted embeddings and vector index if present."""
        embed_ckpt = self.knowledge_dir / "embeddings" / "embedding_model.joblib"
        index_json = self.knowledge_dir / "embeddings" / "index.json"

        if embed_ckpt.exists() and index_json.exists():
            try:
                self.embedding_model.load(str(embed_ckpt))
                self.vector_store.load(str(index_json))
                self.is_initialized = True
                logger.info(
                    f"KnowledgeRetrievalEngine loaded {self.vector_store.count()} indexed chunks from {self.knowledge_dir}"
                )
            except Exception as e:
                logger.warning(f"Failed to load existing knowledge index: {e}")
        else:
            logger.info(f"No existing knowledge index found in {self.knowledge_dir}. Engine ready for indexing.")

    def query(
        self,
        text: str,
        top_k: int = 3,
        threshold_override: Optional[float] = None
    ) -> RAGQueryResult:
        """Query knowledge base with strict anti-hallucination confidence gating."""
        t0 = time.time()
        threshold = threshold_override if threshold_override is not None else self.min_confidence_threshold

        clean_query = text.strip()
        if not clean_query or not self.is_initialized or self.vector_store.count() == 0:
            return RAGQueryResult(
                query=clean_query,
                is_confident=False,
                confidence_score=0.0,
                results=[],
                formatted_context=[],
                cited_sources=[],
                status_message="NO_VERIFIED_KNOWLEDGE_FOUND",
                retrieval_latency_ms=round((time.time() - t0) * 1000, 2)
            )

        # 1. Generate query embedding
        query_vec = self.embedding_model.embed_text(clean_query)

        # 2. Search vector store for top candidates
        candidates = self.vector_store.search(query_vec, top_k=top_k * 3, min_score=0.0)
        if not candidates:
            return RAGQueryResult(
                query=clean_query,
                is_confident=False,
                confidence_score=0.0,
                results=[],
                formatted_context=[],
                cited_sources=[],
                status_message="NO_VERIFIED_KNOWLEDGE_FOUND",
                retrieval_latency_ms=round((time.time() - t0) * 1000, 2)
            )

        # 3. Hybrid Reranking (Semantic + Lexical + Clinical Authority)
        reranked = self.reranker.rerank(query=clean_query, candidates=candidates, top_n=top_k)

        top_score = reranked[0].score if reranked else 0.0

        # 4. Anti-Hallucination Low-Confidence Gate
        if top_score < threshold:
            logger.info(
                f"RAG query '{clean_query[:40]}' score {top_score:.3f} below threshold {threshold:.3f}. Gating answer to prevent hallucination."
            )
            return RAGQueryResult(
                query=clean_query,
                is_confident=False,
                confidence_score=top_score,
                highest_score=top_score,
                threshold_applied=threshold,
                results=[],
                formatted_context=[],
                cited_sources=[],
                status_message="NO_VERIFIED_KNOWLEDGE_FOUND",
                retrieval_latency_ms=round((time.time() - t0) * 1000, 2)
            )

        # 5. Build attributed formatted context and source citations
        formatted_context: List[str] = []
        cited_sources: List[Dict[str, str]] = []
        seen_sources = set()

        for res in reranked:
            chunk = res.chunk
            formatted_context.append(
                f"[{chunk.title} | {chunk.publisher} - Section: {chunk.section}]\n\"{chunk.content}\""
            )
            if chunk.source_id not in seen_sources:
                seen_sources.add(chunk.source_id)
                cited_sources.append({
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "publisher": chunk.publisher,
                    "url": chunk.url,
                    "citation": chunk.citation
                })

        latency = (time.time() - t0) * 1000
        logger.info(
            f"RAG retrieved {len(reranked)} verified passages for query '{clean_query[:40]}' (top_score={top_score:.3f}, latency={latency:.1f}ms)"
        )

        return RAGQueryResult(
            query=clean_query,
            is_confident=True,
            confidence_score=top_score,
            highest_score=top_score,
            threshold_applied=threshold,
            results=reranked,
            formatted_context=formatted_context,
            cited_sources=cited_sources,
            total_chunks_evaluated=len(candidates),
            status_message="VERIFIED_KNOWLEDGE_RETRIEVED",
            retrieval_latency_ms=round(latency, 2)
        )
