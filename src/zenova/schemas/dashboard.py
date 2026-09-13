"""Pydantic schemas for the Clinician Web Dashboard."""
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

CLINICAL_DISCLAIMER_NOTICE = (
    "CLINICAL DECISION-SUPPORT ONLY: Algorithmic predictions represent statistical pattern proxies "
    "and interaction strategies. They DO NOT constitute psychiatric diagnoses, DSM-5 medical evaluations, "
    "or emergency dispatch directives. Professional clinical judgment always supersedes automated outputs."
)


class MLExplainabilityCard(BaseModel):
    """Transparent explainability card for a specific machine learning component."""
    model_name: str = Field(..., description="Human-readable model name")
    model_version: str = Field(..., description="Model version or checkpoint identifier")
    modality: str = Field(..., description="Modality (text, speech, behavior, context, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    prediction_summary: str = Field(..., description="Concise human-readable prediction summary")
    relevant_signals: List[str] = Field(default_factory=list, description="Key features, keywords, cues, or biometric deviations")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Prediction timestamp")
    limitations: str = Field(..., description="Explicit clinical and algorithmic limitations")
    clinical_notice: str = Field(default=CLINICAL_DISCLAIMER_NOTICE, description="Mandatory non-diagnosis clinical notice")


class PatientSummary(BaseModel):
    """High-level patient status for dashboard roster."""
    user_id: str
    current_risk_level: str = "no_risk"
    baseline_status: str = "insufficient_data"
    baseline_deviation_score: float = 0.0
    open_alerts_count: int = 0
    total_turns: int = 0
    total_sessions: int = 0
    last_active_at: Optional[datetime] = None


class PatientListResponse(BaseModel):
    """Roster of active users/patients for clinician navigation."""
    count: int
    patients: List[PatientSummary]
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class EmotionTrendPoint(BaseModel):
    turn_id: int
    timestamp: Optional[str] = None
    primary_emotion: str
    valence: float
    arousal: float
    confidence: float = 1.0


class SymptomTrendPoint(BaseModel):
    turn_id: int
    timestamp: Optional[str] = None
    detected_symptoms: List[str] = Field(default_factory=list)
    highest_severity: str = "none"


class RiskTrendPoint(BaseModel):
    turn_id: int
    timestamp: Optional[str] = None
    risk_level: str
    crisis_category: str = "none"
    confidence: float = 1.0


class StrategyTrendPoint(BaseModel):
    turn_id: int
    timestamp: Optional[str] = None
    strategy: str
    confidence: float = 1.0


class PatientTrendsResponse(BaseModel):
    """Longitudinal time-series trajectories for an individual patient."""
    user_id: str
    emotion_trends: List[EmotionTrendPoint] = Field(default_factory=list)
    symptom_trends: List[SymptomTrendPoint] = Field(default_factory=list)
    risk_trends: List[RiskTrendPoint] = Field(default_factory=list)
    strategy_trends: List[StrategyTrendPoint] = Field(default_factory=list)
    baseline_metrics: Dict[str, Any] = Field(default_factory=dict)
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class TimelineEvent(BaseModel):
    """Unified longitudinal event item in chronological timeline."""
    event_id: str
    event_type: str = Field(..., description="Event category: conversation, behavior, voice, escalation, baseline")
    timestamp: datetime
    title: str
    description: str
    severity: str = "info"
    data: Dict[str, Any] = Field(default_factory=dict)
    explainability: Optional[MLExplainabilityCard] = None


class PatientTimelineResponse(BaseModel):
    """Chronological longitudinal timeline response."""
    user_id: str
    events_count: int
    events: List[TimelineEvent]
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class ConversationTurnDisplay(BaseModel):
    """Turn details with parsed ML predictions and explainability."""
    turn_id: int
    session_id: str
    speaker: str
    content: str
    created_at: Optional[str] = None
    emotion: Optional[Dict[str, Any]] = None
    symptoms: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    strategy: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None
    behavior: Optional[Dict[str, Any]] = None
    explainability: Optional[List[MLExplainabilityCard]] = None


class PatientConversationsResponse(BaseModel):
    """Patient conversation history with explainability."""
    user_id: str
    turns_count: int
    turns: List[ConversationTurnDisplay]
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class PatientExplainabilityResponse(BaseModel):
    """Granular ML explainability cards for all active models."""
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_cards: List[MLExplainabilityCard]
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class DashboardOverviewResponse(BaseModel):
    """Overview triage metrics, alerts, risk distribution, and system status."""
    active_alerts_count: int
    alerts_by_severity: Dict[str, int]
    risk_distribution: Dict[str, int]
    recent_escalations: List[Dict[str, Any]] = Field(default_factory=list)
    active_users_count: int
    total_sessions_count: int
    system_status: Dict[str, Any]
    clinical_notice: str = CLINICAL_DISCLAIMER_NOTICE


class DashboardModuleDescriptor(BaseModel):
    """Descriptor for an extensible dashboard widget/module."""
    module_id: str
    name: str
    category: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True


class DashboardModulesResponse(BaseModel):
    count: int
    modules: List[DashboardModuleDescriptor]


class DashboardAccessAuditEntry(BaseModel):
    access_id: str
    actor_id: str
    actor_role: str
    target_user_id: Optional[str] = None
    endpoint: str
    action: str
    ip_address: str
    timestamp: datetime


class DashboardAuditLogsResponse(BaseModel):
    count: int
    logs: List[DashboardAccessAuditEntry]
