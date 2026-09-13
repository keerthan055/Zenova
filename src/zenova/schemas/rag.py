"""Data models and schemas for Curated Retrieval-Augmented Generation (RAG)."""
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    GUIDELINE = "clinical_guideline"
    CLINICAL_GUIDELINE = "clinical_guideline"
    PSYCHOEDUCATION = "psychoeducation"
    COPING_STRATEGY = "coping_strategy"
    CRISIS_RESOURCE = "crisis_resource"
    CLINICAL_REVIEW = "clinical_review"
    GOVERNMENT_RESOURCE = "government_resource"


class EvidenceTier(str, Enum):
    TIER_1_GLOBAL_HEALTH_AGENCY = "TIER_1_GLOBAL_HEALTH_AGENCY"  # WHO, CDC
    TIER_1_CLINICAL_PEER_REVIEW = "TIER_1_CLINICAL_PEER_REVIEW"  # APA, NIMH
    TIER_2_GOVERNMENT_HEALTH_SERVICE = "TIER_2_GOVERNMENT_HEALTH_SERVICE"  # NHS, SAMHSA
    TIER_1_CLINICAL_GUIDELINE = "TIER_1_GLOBAL_HEALTH_AGENCY"
    TIER_3_PEER_REVIEWED_LITERATURE = "TIER_1_CLINICAL_PEER_REVIEW"


class ReviewStatus(str, Enum):
    APPROVED = "APPROVED"
    PENDING = "PENDING_REVIEW"
    REJECTED = "REJECTED"
    DRAFT = "draft"


class ReliabilityMetadata(BaseModel):
    evidence_tier: EvidenceTier
    peer_reviewed: bool = True
    reviewed_by: str = "ZENOVA Clinical Advisory Review"
    last_reviewed_date: str = "2024-01-01"
    review_status: ReviewStatus = ReviewStatus.APPROVED
    clinical_guideline_aligned: bool = True
    authority_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    license: str = "Open Access / Public Domain / Educational Fair Use"


class KnowledgeSource(BaseModel):
    """Metadata schema required for every source admitted to the curated knowledge base."""
    source_id: str
    title: str
    publisher: str
    url: str
    publication_date: str = "2023-01-01"
    last_reviewed_date: Optional[str] = "2024-01-01"
    source_type: SourceType
    citation: str
    scope: List[str] = Field(default_factory=list)
    reliability: ReliabilityMetadata
    sections: Optional[List[Dict[str, Any]]] = None
    raw_content: Optional[str] = None


class KnowledgeChunk(BaseModel):
    """Passage-level semantic chunk with full provenance and source attribution."""
    chunk_id: str
    source_id: str
    title: str = ""
    source_title: Optional[str] = None
    publisher: str
    url: str = ""
    citation: str = ""
    section: str = "General"
    section_heading: Optional[str] = None
    content: str
    token_count: int = 0
    word_count: Optional[int] = None
    source_type: SourceType = SourceType.COPING_STRATEGY
    evidence_tier: EvidenceTier = EvidenceTier.TIER_1_GLOBAL_HEALTH_AGENCY
    authority_weight: float = 1.0

    @model_validator(mode="after")
    def populate_aliases(self) -> "KnowledgeChunk":
        if not self.title and self.source_title:
            self.title = self.source_title
        if not self.source_title and self.title:
            self.source_title = self.title
        if self.section_heading and self.section == "General":
            self.section = self.section_heading
        if not self.section_heading and self.section:
            self.section_heading = self.section
        if self.word_count is None and self.content:
            self.word_count = len(self.content.split())
        return self


class RAGSearchResult(BaseModel):
    """Individual retrieved passage with scoring breakdown."""
    chunk: KnowledgeChunk
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    authority_bonus: float = 0.0
    citation_text: str = ""


class RAGQueryResult(BaseModel):
    """Complete structured response from KnowledgeRetrievalEngine."""
    query: str
    is_confident: bool
    confidence_score: float
    highest_score: Optional[float] = None
    threshold_applied: float = 0.30
    results: List[RAGSearchResult] = Field(default_factory=list)
    formatted_context: List[str] = Field(default_factory=list)
    cited_sources: List[Dict[str, str]] = Field(default_factory=list)
    total_chunks_evaluated: int = 0
    status_message: str
    retrieval_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def set_highest_score(self) -> "RAGQueryResult":
        if self.highest_score is None:
            self.highest_score = self.confidence_score
        return self
