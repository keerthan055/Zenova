"""Response generation and RAG grounding schemas for ZENOVA."""
from typing import List, Optional, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from zenova.schemas.strategy import SupportStrategy


class GenerationResult(BaseModel):
    generated_text: str
    strategy_applied: SupportStrategy
    model_name: str
    rag_grounded: bool = False
    retrieved_sources: List[str] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    generation_metadata: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
