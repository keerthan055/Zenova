"""Crisis and high-risk safety assessment schemas for ZENOVA."""
from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CrisisCategory(str, Enum):
    NONE = "none"
    SELF_HARM = "self_harm"
    SUICIDAL_IDEATION = "suicidal_ideation"
    IMMINENT_DANGER = "imminent_danger"
    VIOLENCE = "violence"
    DELUSIONAL_BREAK = "delusional_break"


class CrisisRiskAssessment(BaseModel):
    risk_level: RiskLevel = RiskLevel.LOW
    crisis_category: CrisisCategory = CrisisCategory.NONE
    confidence: float = Field(ge=0.0, le=1.0)
    trigger_cues: List[str] = Field(default_factory=list, description="Detected cues triggering risk assessment")
    requires_immediate_escalation: bool = False
    escalation_action: Optional[str] = Field(
        default=None,
        description="E.g., provide_crisis_hotline, notify_on_call_clinician, lock_dialogue"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or self.requires_immediate_escalation
