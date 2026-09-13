"""Support strategy planner schemas for ZENOVA based on Hill's Helping Skills & ESConv."""
from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SupportStrategy(str, Enum):
    QUESTION = "Question"
    RESTATEMENT_OR_PARAPHRASING = "Restatement or Paraphrasing"
    REFLECTION_OF_FEELINGS = "Reflection of feelings"
    AFFIRMATION_AND_REASSURANCE = "Affirmation and Reassurance"
    SELF_DISCLOSURE = "Self-disclosure"
    PROVIDING_SUGGESTIONS = "Providing Suggestions"
    INFORMATION = "Information"
    OTHERS = "Others"


class DialogStage(str, Enum):
    EXPLORATION = "Exploration"
    COMFORTING = "Comforting"
    ACTION = "Action"


class StrategyPrediction(BaseModel):
    selected_strategy: SupportStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    stage: DialogStage = DialogStage.COMFORTING
    ranked_strategies: List[Dict[str, float]] = Field(default_factory=list)
    rationale: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
