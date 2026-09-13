"""Standard typed data contracts for ZENOVA.

Includes:
- UserInput
- ConversationTurn
- ConversationContext
- EmotionResult
- SymptomResult
- RiskResult
- BaselineResult
- StrategyResult
- GeneratedResponse
- SafetyResult
- EscalationEvent
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# ENUMS
# ==============================================================================

class ModalityType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    MULTIMODAL = "multimodal"


class SpeakerRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CLINICIAN = "clinician"


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


class SymptomSeverity(str, Enum):
    NONE = "none"
    SUBCLINICAL = "subclinical"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


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


class SafetyAction(str, Enum):
    ALLOW = "allow"
    REVISE = "revise"
    BLOCK_AND_ESCALATE = "block_and_escalate"


class EscalationStatus(str, Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


# ==============================================================================
# SCHEMAS
# ==============================================================================

class UserInput(BaseModel):
    """Inbound user turn with multimodal support and session metadata."""
    session_id: str = Field(..., description="Unique active session identifier")
    user_id: str = Field(..., description="Anonymized user identifier")
    text: str = Field(..., min_length=1, description="Raw textual input from user")
    modality: ModalityType = Field(default=ModalityType.TEXT)
    audio_features: Optional[Dict[str, float]] = Field(default=None, description="Acoustic features if voice")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmotionResult(BaseModel):
    """Emotion analysis output schema."""
    is_placeholder: bool = Field(default=False, description="Flag indicating placeholder vs trained model")
    module_version: str = Field(default="1.0.0")
    primary_emotion: EmotionCategory
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Pleasantness: -1.0 to 1.0")
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0, description="Activation energy: -1.0 to 1.0")
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Perceived control: -1.0 to 1.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability for {k} must be in [0, 1], got {p}")
        return v


class SymptomSignal(BaseModel):
    """Observational symptom signal with evidence."""
    marker_name: str
    severity: SymptomSeverity = SymptomSeverity.SUBCLINICAL
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: List[str] = Field(default_factory=list)
    clinical_disclaimer: str = (
        "Observational marker only; does not constitute psychiatric diagnosis."
    )


class SymptomResult(BaseModel):
    """Symptom and signal analysis result."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    signals: List[SymptomSignal] = Field(default_factory=list)
    aggregate_severity: SymptomSeverity = SymptomSeverity.NONE
    disclaimer: str = "Informational signals only; not a formal psychiatric diagnosis."
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskResult(BaseModel):
    """Crisis and high-risk safety assessment result."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    risk_level: RiskLevel = RiskLevel.LOW
    crisis_category: CrisisCategory = CrisisCategory.NONE
    confidence: float = Field(ge=0.0, le=1.0)
    requires_immediate_escalation: bool = False
    requires_escalation: Optional[bool] = None
    trigger_cues: List[str] = Field(default_factory=list)
    escalation_action: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or self.requires_immediate_escalation


class BaselineResult(BaseModel):
    """Longitudinal baseline deviation comparison."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    user_id: str
    status: str = Field(
        default="insufficient_data",
        description="Establishment status: insufficient_data, provisional_baseline, or established_baseline"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_observations: int = Field(default=0, ge=0)
    z_score_valence: float = Field(default=0.0)
    is_significant_deviation: bool = False
    deviation_notes: Optional[str] = None
    notes: Optional[str] = None
    metric_deviations: Dict[str, Any] = Field(default_factory=dict)
    deviating_features: List[str] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Statistical deviation from personal baseline; observational and non-diagnostic."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategyResult(BaseModel):
    """Support strategy planner recommendation based on Hill's helping skills."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    selected_strategy: SupportStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    stage: DialogStage = DialogStage.COMFORTING
    rationale: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeneratedResponse(BaseModel):
    """Conversational response generation result."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    response_text: str
    strategy_applied: SupportStrategy
    model_name: str
    rag_sources: List[str] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    validation_passed: bool = True
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyResult(BaseModel):
    """Safety gate evaluation outcome."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    is_safe: bool = True
    action: SafetyAction = SafetyAction.ALLOW
    violated_policies: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    toxicity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    modified_text: Optional[str] = None
    override_applied: bool = False
    audit_id: Optional[str] = None
    disclaimer: str = "ZENOVA is an AI wellbeing support companion, not a clinical provider."
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EscalationEvent(BaseModel):
    """Clinician escalation incident record."""
    event_id: str
    session_id: str
    user_id: str
    risk_level: RiskLevel
    crisis_category: CrisisCategory
    trigger_cues: List[str] = Field(default_factory=list)
    status: EscalationStatus = EscalationStatus.PENDING
    assigned_clinician_id: Optional[str] = None
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BehavioralResult(BaseModel):
    """Optional passive behavioral and sensing analysis outcome."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    is_available: bool = Field(default=False, description="True if behavioral data was provided for this evaluation")
    activity_level: Optional[str] = None
    sleep_duration_hours: Optional[float] = None
    social_conversation_minutes: Optional[float] = None
    phone_screen_unlocks: Optional[int] = None
    mobility_radius_km: Optional[float] = None
    behavioral_anomaly_detected: bool = False
    anomaly_notes: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VoiceResult(BaseModel):
    """Optional voice and acoustic emotion analysis outcome."""
    is_placeholder: bool = Field(default=False)
    module_version: str = Field(default="1.0.0")
    is_available: bool = Field(default=False, description="True if voice audio was provided for this evaluation")
    primary_emotion: Optional[EmotionCategory] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    acoustic_features: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationTurn(BaseModel):
    """Comprehensive record of a conversational turn."""
    turn_id: int
    session_id: str
    speaker: SpeakerRole
    content: str
    emotion: Optional[EmotionResult] = None
    symptoms: Optional[SymptomResult] = None
    risk: Optional[RiskResult] = None
    baseline: Optional[BaselineResult] = None
    behavior: Optional[BehavioralResult] = None
    voice: Optional[VoiceResult] = None
    strategy: Optional[StrategyResult] = None
    safety: Optional[SafetyResult] = None
    context: Optional[Dict[str, Any]] = None
    user_feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationContext(BaseModel):
    """Longitudinal session context carrying history and state across turns."""
    session_id: str
    user_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    active_stage: DialogStage = DialogStage.EXPLORATION
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

from zenova.schemas.context import (
    MultimodalContext,
    ConversationContextBlock,
    EmotionContextBlock,
    SymptomsContextBlock,
    RiskContextBlock,
    BaselineContextBlock,
    BehaviorContextBlock,
    VoiceContextBlock,
    HistoryContextBlock,
    MetadataContextBlock,
    UserFeedback,
    TurnOutcome,
    PrivacyLevel,
    ValenceTrend
)
