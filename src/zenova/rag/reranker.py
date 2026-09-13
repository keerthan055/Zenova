"""Hybrid reranker combining dense semantic similarity, lexical overlap, and clinical authority weighting."""
import re
from typing import List, Tuple, Optional, Any, Union
from zenova.schemas.rag import KnowledgeChunk, RAGSearchResult


class HybridReranker:
    """Reranks vector search candidates using hybrid semantic, lexical, and authority scoring."""

    def __init__(
        self,
        semantic_weight: float = 0.60,
        lexical_weight: float = 0.30,
        authority_weight: float = 0.10
    ):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.authority_weight = authority_weight

    def _compute_lexical_overlap(self, query: str, document: str) -> float:
        """Compute word overlap Jaccard coefficient between query and document."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        d_tokens = set(re.findall(r"\w+", document.lower()))
        if not q_tokens or not d_tokens:
            return 0.0

        intersection = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens)
        return float(intersection) / float(union) if union > 0 else 0.0

    def rerank(
        self,
        query: str,
        candidates: Optional[List[Tuple[KnowledgeChunk, float]]] = None,
        top_n: int = 3,
        results: Optional[List[Any]] = None,
        top_k: Optional[int] = None
    ) -> List[RAGSearchResult]:
        """Score and rerank search candidates."""
        input_candidates = candidates or []
        if not input_candidates and results:
            for item in results:
                if isinstance(item, tuple):
                    input_candidates.append(item)
                elif isinstance(item, RAGSearchResult):
                    input_candidates.append((item.chunk, item.score))

        target_n = top_k if top_k is not None else top_n
        output_results: List[RAGSearchResult] = []

        for chunk, sem_score in input_candidates:
            lex_score = self._compute_lexical_overlap(query, chunk.content)
            # Authority multiplier normalized around 1.0 (range 0.8 to 1.2)
            auth_norm = (chunk.authority_weight - 1.0) * 0.5  # [-0.1 to +0.1]
            auth_bonus = max(-0.1, min(0.1, auth_norm))

            composite = (
                self.semantic_weight * sem_score +
                self.lexical_weight * min(1.0, lex_score * 3.0) +
                self.authority_weight * (0.5 + auth_bonus)
            )

            citation_str = (
                f"[Source: {chunk.title} | Publisher: {chunk.publisher} | "
                f"Citation: {chunk.citation} | URL: {chunk.url}]"
            )

            output_results.append(
                RAGSearchResult(
                    chunk=chunk,
                    score=round(composite, 4),
                    semantic_score=round(sem_score, 4),
                    lexical_score=round(lex_score, 4),
                    authority_bonus=round(auth_bonus, 4),
                    citation_text=citation_str
                )
            )

        output_results.sort(key=lambda r: r.score, reverse=True)
        return output_results[:target_n]
