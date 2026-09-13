"""Emotion schema definition for ZENOVA."""
from enum import Enum
from typing import Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class EmotionCategory(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    ANXIETY = "anxiety"
    FEAR = "fear"
    NEUTRAL = "neutral"
    FRUSTRATION = "frustration"
    HOPE = "hope"
    GRIEF = "grief"
    GUILT = "guilt"
    SHAME = "shame"


class EmotionAnalysisResult(BaseModel):
    primary_emotion: EmotionCategory
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    valence: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0, description="Pleasantness: -1.0 to +1.0")
    arousal: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0, description="Activation energy: -1.0 to +1.0")
    dominance: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0, description="Control: -1.0 to +1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {k} must be between 0.0 and 1.0, got {prob}")
        return v
