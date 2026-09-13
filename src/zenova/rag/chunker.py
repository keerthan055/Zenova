"""Document chunker and provenance validator for curated healthcare sources."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from zenova.schemas.rag import (
    KnowledgeSource,
    KnowledgeChunk,
    ReliabilityMetadata,
    ReviewStatus,
    SourceType,
    EvidenceTier
)
from zenova.core.logging import get_logger

logger = get_logger("zenova.rag.chunker")


class DocumentChunker:
    """Validates source provenance and chunks authoritative documents into attributed passages."""

    def __init__(
        self,
        max_words: int = 180,
        overlap_words: int = 25,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.max_words = chunk_size if chunk_size is not None else max_words
        self.overlap_words = chunk_overlap if chunk_overlap is not None else overlap_words

    def validate_source(self, data: Dict[str, Any]) -> KnowledgeSource:
        """Enforce strict provenance, review status, and metadata requirements."""
        required = ["source_id", "title", "publisher", "url", "publication_date", "source_type", "citation", "reliability"]
        for field in required:
            if field not in data or not data[field]:
                raise ValueError(f"Knowledge source missing mandatory field: '{field}'")

        rel = data["reliability"]
        if isinstance(rel, dict):
            status = str(rel.get("review_status", ReviewStatus.PENDING.value)).upper()
            if status not in (ReviewStatus.APPROVED.value, "APPROVED"):
                raise PermissionError(
                    f"Source '{data.get('source_id')}' has review status '{status}'. Only APPROVED sources can be indexed."
                )

        source = KnowledgeSource(**data)
        return source

    def chunk_source(self, data: Dict[str, Any]) -> List[KnowledgeChunk]:
        """Validate and chunk an authoritative source into attributed KnowledgeChunk instances."""
        source = self.validate_source(data)
        sections = data.get("sections", [])
        chunks: List[KnowledgeChunk] = []

        chunk_idx = 0
        for sec in sections:
            heading = sec.get("heading", "General")
            content = sec.get("content", "").strip()
            if not content:
                continue

            words = content.split()
            if len(words) <= self.max_words:
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{source.source_id}_chk_{chunk_idx:02d}",
                        source_id=source.source_id,
                        title=source.title,
                        source_title=source.title,
                        publisher=source.publisher,
                        url=source.url,
                        citation=source.citation,
                        section=heading,
                        section_heading=heading,
                        content=content,
                        token_count=int(len(words) * 1.3),
                        word_count=len(words),
                        source_type=source.source_type,
                        evidence_tier=source.reliability.evidence_tier,
                        authority_weight=source.reliability.authority_weight
                    )
                )
                chunk_idx += 1
            else:
                # Sliding window chunking with overlap
                start = 0
                while start < len(words):
                    end = min(start + self.max_words, len(words))
                    passage = " ".join(words[start:end])
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=f"{source.source_id}_chk_{chunk_idx:02d}",
                            source_id=source.source_id,
                            title=source.title,
                            source_title=source.title,
                            publisher=source.publisher,
                            url=source.url,
                            citation=source.citation,
                            section=heading,
                            section_heading=heading,
                            content=passage,
                            token_count=int((end - start) * 1.3),
                            word_count=end - start,
                            source_type=source.source_type,
                            evidence_tier=source.reliability.evidence_tier,
                            authority_weight=source.reliability.authority_weight
                        )
                    )
                    chunk_idx += 1
                    if end == len(words):
                        break
                    start += self.max_words - self.overlap_words

        logger.info(f"Chunked source '{source.source_id}' ({source.title}) into {len(chunks)} passages.")
        return chunks

    def chunk_file(self, filepath: str) -> List[KnowledgeChunk]:
        """Load and chunk a JSON source file, safely handling rejection of unapproved sources."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self.chunk_source(data)
        except (PermissionError, ValueError) as e:
            logger.warning(f"Rejected unapproved or invalid source file '{filepath}': {e}")
            return []
