"""ZENOVA User-Facing Application Domain Schemas & Contracts."""
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


USER_DISCLAIMER_NOTICE = (
    "IMPORTANT NOTICE: ZENOVA is an AI-powered conversational companion designed exclusively for "
    "general emotional support, self-reflection, and wellbeing tracking. ZENOVA is NOT a licensed healthcare "
    "provider, therapist, psychiatrist, or emergency service. It does not provide medical diagnoses, "
    "clinical psychiatric assessments, medication recommendations, or crisis intervention therapy. "
    "If you are in immediate danger or experiencing a mental health emergency, please call or text 988 "
    "(Suicide & Crisis Lifeline in US/Canada) or reach out to local emergency services immediately."
)


class PrivacyLevel(str, Enum):
    """User-selected privacy and data retention mode."""
    STANDARD = "standard"          # History preserved for user's personal longitudinal reflection
    ANONYMIZED = "anonymized"      # Personally identifiable entities scrubbed before storage
    EPHEMERAL = "ephemeral"        # Zero conversational turns or observations persisted to disk


class CommunicationStyle(str, Enum):
    """User-preferred conversational pacing and tone."""
    WARM_EMPATHIC = "warm_empathic"
    SOLUTION_FOCUSED = "solution_focused"
    REFLECTIVE = "reflective"
    GENTLE_MINIMAL = "gentle_minimal"


class UserPreferences(BaseModel):
    """User settings for data privacy, input modalities, and communication preferences."""
    user_id: str
    save_history: bool = True
    enable_voice: bool = True
    enable_wearables: bool = False
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    preferred_language: str = "en"
    communication_style: CommunicationStyle = CommunicationStyle.WARM_EMPATHIC
    allow_clinician_sharing: bool = False
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UserPreferencesUpdateRequest(BaseModel):
    """Partial update payload for user preferences."""
    save_history: Optional[bool] = None
    enable_voice: Optional[bool] = None
    enable_wearables: Optional[bool] = None
    privacy_level: Optional[PrivacyLevel] = None
    preferred_language: Optional[str] = None
    communication_style: Optional[CommunicationStyle] = None
    allow_clinician_sharing: Optional[bool] = None


class WellbeingCheckinRequest(BaseModel):
    """User submission for a daily or periodic wellbeing check-in."""
    user_id: str
    mood_score: int = Field(ge=1, le=10, description="Overall mood score from 1 (very low) to 10 (flourishing)")
    valence: float = Field(ge=-1.0, le=1.0, default=0.0, description="Affective valence from -1.0 (unpleasant) to +1.0 (pleasant)")
    sleep_hours: float = Field(ge=0.0, le=24.0, default=7.0, description="Hours of sleep in past 24 hours")
    stress_level: int = Field(ge=1, le=5, default=3, description="Subjective stress from 1 (calm) to 5 (overwhelmed)")
    energy_level: int = Field(ge=1, le=5, default=3, description="Subjective energy from 1 (exhausted) to 5 (vital)")
    notes: Optional[str] = Field(default=None, max_length=1000, description="Optional personal reflection or context note")


class WellbeingCheckinResponse(BaseModel):
    """Structured response confirming a recorded wellbeing check-in."""
    checkin_id: str
    user_id: str
    mood_score: int
    valence: float
    sleep_hours: float
    stress_level: int
    energy_level: int
    notes: Optional[str] = None
    created_at: str
    feedback_message: str


class CheckinHistoryItem(BaseModel):
    """Single historical check-in data point."""
    checkin_id: str
    mood_score: int
    valence: float
    sleep_hours: float
    stress_level: int
    energy_level: int
    notes: Optional[str] = None
    created_at: str


class CheckinHistoryResponse(BaseModel):
    """Historical timeline of check-ins with summary statistics."""
    user_id: str
    total_checkins: int
    average_mood: float
    average_sleep_hours: float
    average_stress: float
    checkins: List[CheckinHistoryItem]


class SupportResource(BaseModel):
    """Information for an official crisis, mental health, or community support resource."""
    name: str
    category: str
    description: str
    phone: Optional[str] = None
    text_sms: Optional[str] = None
    website: Optional[str] = None
    hours: str = "24/7/365"
    availability: str = "Free & Confidential"
    country: str = "US / International"


class SupportResourcesResponse(BaseModel):
    """Curated directory of verified support resources and grounding exercises."""
    hotlines: List[SupportResource]
    grounding_techniques: List[Dict[str, Any]]
    disclaimer: str = USER_DISCLAIMER_NOTICE


class SessionSummaryItem(BaseModel):
    """High-level summary of a user's conversational session."""
    session_id: str
    user_id: str
    turn_count: int
    first_message_preview: Optional[str] = None
    created_at: str
    updated_at: str


class UserSessionsResponse(BaseModel):
    """List of past conversation sessions for the user."""
    user_id: str
    total_sessions: int
    sessions: List[SessionSummaryItem]


class UserDataExportResponse(BaseModel):
    """Comprehensive GDPR/HIPAA-compliant export of all user data stored in ZENOVA."""
    export_id: str
    user_id: str
    exported_at: str
    preferences: Dict[str, Any]
    checkins: List[Dict[str, Any]]
    sessions_count: int
    sessions: List[Dict[str, Any]]
    observations_count: int
    disclaimer: str = USER_DISCLAIMER_NOTICE


class DataPurgeResponse(BaseModel):
    """Confirmation of complete data deletion under the Right to be Forgotten."""
    user_id: str
    purged_at: str
    deleted_sessions: int
    deleted_turns: int
    deleted_checkins: int
    deleted_observations: int
    status: str = "PURGED_SUCCESSFULLY"
    message: str = "All personal conversational turns, check-ins, baselines, and observations have been permanently deleted."
