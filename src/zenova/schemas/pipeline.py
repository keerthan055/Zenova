"""End-to-end pipeline schemas for ZENOVA orchestration."""
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from zenova.schemas.emotion import EmotionAnalysisResult
from zenova.schemas.symptoms import SymptomAnalysisResult
from zenova.schemas.risk import CrisisRiskAssessment
from zenova.schemas.baseline import BaselineDeviationReport
from zenova.schemas.strategy import StrategyPrediction
from zenova.schemas.safety import SafetyGateResult


class ZenovaUserTurn(BaseModel):
    session_id: str
    user_id: str
    turn_id: int
    text: str
    audio_features: Optional[Dict[str, float]] = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ZenovaPipelineOutput(BaseModel):
    session_id: str
    turn_id: int
    user_input: str
    emotion: Optional[EmotionAnalysisResult] = None
    symptoms: Optional[SymptomAnalysisResult] = None
    risk: Optional[CrisisRiskAssessment] = None
    baseline_deviation: Optional[BaselineDeviationReport] = None
    strategy: Optional[StrategyPrediction] = None
    safety: Optional[SafetyGateResult] = None
    final_response: str
    escalated_to_human: bool = False
    escalation_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
