"""Multimodal Context Engine schemas for ZENOVA Step 9.

Defines the normalized 9-block context schema:
1. conversation
2. emotion
3. symptoms
4. risk
5. baseline
6. behavior
7. voice
8. history
9. metadata

Enforces deterministic versioning, strict typing, confidence propagation,
missing-data explicitness, and privacy-aware attributes.
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator


class DialogStage(str, Enum):
    EXPLORATION = "Exploration"
    COMFORTING = "Comforting"
    ACTION = "Action"


class PrivacyLevel(str, Enum):
    STANDARD = "STANDARD"
    ANONYMIZED = "ANONYMIZED"
    EPHEMERAL = "EPHEMERAL"


class ValenceTrend(str, Enum):
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"
    INSUFFICIENT_DATA = "insufficient_data"


class UserFeedback(BaseModel):
    """Explicit feedback submitted by the user on a specific turn."""
    turn_id: int
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="1-5 star user rating")
    is_helpful: Optional[bool] = None
    feedback_text: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TurnOutcome(BaseModel):
    """Estimated or evaluated response outcome."""
    turn_id: int
    outcome_type: str = Field(default="affective_shift", description="e.g. affective_shift, de_escalation, engagement")
    sentiment_delta: float = Field(default=0.0, description="Change in valence post-response")
    outcome_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- 1. Conversation Context Block ---
class ConversationContextBlock(BaseModel):
    session_id: str
    user_id: str
    turn_id: int = 1
    current_text: str = ""
    speaker: str = "user"
    dialog_stage: DialogStage = DialogStage.EXPLORATION
    turn_count: int = 1
    language: str = "en"
    input_modalities: List[str] = Field(default_factory=lambda: ["text"])
    is_crisis_bypass: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- 2. Emotion Context Block ---
class EmotionContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    primary_emotion: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    source: str = "unavailable"
    is_discrepancy_detected: bool = False
    disclaimer: str = "Acoustic and textual emotion signals are observational affect proxies; do not interpret as clinical diagnoses."
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "modality_omitted"
        return self


# --- 3. Symptoms Context Block ---
class SymptomsContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    signal_count: int = 0
    primary_signals: List[str] = Field(default_factory=list)
    max_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    disclaimer: str = "Identified symptom signals are observational patterns and do NOT constitute clinical diagnosis."
    model_version: Optional[str] = None
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "modality_omitted"
        return self


# --- 4. Risk Context Block ---
class RiskContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    risk_level: str = "low"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    crisis_category: str = "none"
    is_high_risk: bool = False
    requires_escalation: bool = False
    trigger_cues: List[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "modality_omitted"
        return self


# --- 5. Baseline Context Block ---
class BaselineContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    status: str = "insufficient_data"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    total_observations: int = 0
    active_features: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    anomalous_features: List[str] = Field(default_factory=list)
    interpretation_summary: Optional[str] = None
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "insufficient_data"
        return self


# --- 6. Behavior Context Block ---
class BehaviorContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    source_device: Optional[str] = None
    domains: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    bdi: float = Field(default=0.0, ge=0.0)
    is_anomalous: bool = False
    affected_domains: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    disclaimer: str = "Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses."
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "passive_sensing_omitted"
        return self


# --- 7. Voice Context Block ---
class VoiceContextBlock(BaseModel):
    is_available: bool = False
    reason: Optional[str] = None
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    predicted_emotion: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)
    acoustic_summary: Dict[str, float] = Field(default_factory=dict)
    disclaimer: str = "Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses."
    timestamp: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.is_available and not self.reason:
            self.reason = "audio_omitted"
        return self


# --- 8. History Context Block ---
class HistoryContextBlock(BaseModel):
    turn_count: int = 0
    recent_turns: List[Dict[str, Any]] = Field(default_factory=list)
    previous_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    previous_outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    affective_trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    valence_trend: ValenceTrend = ValenceTrend.INSUFFICIENT_DATA
    dominant_themes: List[str] = Field(default_factory=list)
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list)


# --- 9. Metadata Context Block ---
class MetadataContextBlock(BaseModel):
    schema_version: str = "1.0.0"
    engine_version: str = "zenova-context-v1.0.0"
    context_id: str
    context_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    confidence_propagation: Dict[str, Any] = Field(default_factory=dict)
    missing_modalities: List[str] = Field(default_factory=list)
    available_modalities: List[str] = Field(default_factory=list)
    fused_multimodal_state: Optional[Dict[str, Any]] = None
    privacy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "privacy_level": PrivacyLevel.STANDARD.value,
            "is_redacted": False,
            "redacted_fields_count": 0,
            "retention_policy": "standard_retention"
        }
    )


# --- Root Normalized Multimodal Context ---
class MultimodalContext(BaseModel):
    """The master normalized context object for ZENOVA Step 9.

    Contains exactly the 9 required blocks:
    - conversation
    - emotion
    - symptoms
    - risk
    - baseline
    - behavior
    - voice
    - history
    - metadata
    """
    conversation: ConversationContextBlock
    emotion: EmotionContextBlock
    symptoms: SymptomsContextBlock
    risk: RiskContextBlock
    baseline: BaselineContextBlock
    behavior: BehaviorContextBlock
    voice: VoiceContextBlock
    history: HistoryContextBlock
    metadata: MetadataContextBlock
