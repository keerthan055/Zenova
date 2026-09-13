"""Mental health symptom and signal schemas for ZENOVA."""
from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SymptomSeverity(str, Enum):
    NONE = "none"
    SUBCLINICAL = "subclinical"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class SymptomSignal(BaseModel):
    marker_name: str = Field(..., description="E.g., sleep_disturbance, anhedonia, fatigue, rumination")
    severity: SymptomSeverity = SymptomSeverity.SUBCLINICAL
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: List[str] = Field(default_factory=list, description="Verbatim spans in text exhibiting signal")
    clinical_disclaimer: str = (
        "Signal extraction is an observational heuristic for supportive dialogue planning "
        "and does not constitute clinical diagnosis."
    )


class SymptomAnalysisResult(BaseModel):
    signals: List[SymptomSignal] = Field(default_factory=list)
    aggregate_severity: SymptomSeverity = SymptomSeverity.NONE
    notes: Optional[str] = None
    disclaimer: str = "Informational signals only; not a formal psychiatric diagnosis."
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
