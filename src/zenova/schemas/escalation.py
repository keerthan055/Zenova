"""Data models and schemas for ZENOVA Human/Clinician Escalation Subsystem."""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class EscalationSeverity(str, Enum):
    """Urgency level of the clinical alert."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationStatus(str, Enum):
    """Lifecycle state of an escalation alert."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class EscalationTriggerType(str, Enum):
    """Origin category that caused the escalation trigger."""
    CRISIS_RISK = "crisis_risk"
    REPEATED_SIGNALS = "repeated_signals"
    LONGITUDINAL_CHANGE = "longitudinal_change"
    SAFETY_GATE_EVENT = "safety_gate_event"
    CLINICIAN_RULE = "clinician_rule"


class EscalationActionType(str, Enum):
    """Taxonomy of clinical actions taken to resolve an alert."""
    EMERGENCY_SERVICES_CONTACTED = "emergency_services_contacted"
    HOTLINE_WARM_TRANSFER = "hotline_warm_transfer"
    USER_OUTREACH_CALL = "user_outreach_call"
    SAFETY_PLAN_ACTIVATED = "safety_plan_activated"
    APPOINTMENT_SCHEDULED = "appointment_scheduled"
    REFERRED_TO_EXTERNAL_CARE = "referred_to_external_care"
    FALSE_POSITIVE_FLAGGED = "false_positive_flagged"
    NO_ACTION_REQUIRED = "no_action_required"


class UserRole(str, Enum):
    """Role-based access control roles."""
    CLINICIAN = "clinician"
    TRIAGE_SUPERVISOR = "triage_supervisor"
    SYSTEM_ADMIN = "system_admin"
    AUDITOR = "auditor"
    PATIENT = "patient"


class EscalationReason(BaseModel):
    """Structured rationale explaining why an alert was generated."""
    trigger_type: EscalationTriggerType
    code: str = Field(..., description="Unique reason code e.g. CRITICAL_RISK_DETECTED")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed clinical or system context")
    trigger_cues: List[str] = Field(default_factory=list, description="Keywords or matched tokens that triggered the alert")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EscalationContextSnapshot(BaseModel):
    """Clinically relevant conversational and analytical context snapshot."""
    session_id: str
    user_id: str
    turn_id: Optional[int] = None
    last_user_message: str = Field(default="", description="Recent user utterance")
    emotion_summary: Optional[Dict[str, Any]] = None
    risk_summary: Optional[Dict[str, Any]] = None
    baseline_summary: Optional[Dict[str, Any]] = None
    behavior_summary: Optional[Dict[str, Any]] = None
    voice_summary: Optional[Dict[str, Any]] = None
    safety_summary: Optional[Dict[str, Any]] = None


class EscalationDecision(BaseModel):
    """Output from the EscalationDecisionEngine evaluating incoming signals."""
    should_escalate: bool = False
    severity: EscalationSeverity = EscalationSeverity.LOW
    trigger_type: Optional[EscalationTriggerType] = None
    reason: Optional[EscalationReason] = None
    context_snapshot: Optional[EscalationContextSnapshot] = None


class EscalationAuditEntry(BaseModel):
    """Immutable audit record documenting a status change or clinical intervention."""
    audit_id: str
    alert_id: str
    action: str = Field(..., description="Action performed e.g. created, acknowledged, resolved")
    actor_id: str = Field(..., description="ID of the clinician, supervisor, or system actor")
    actor_role: UserRole
    previous_status: Optional[EscalationStatus] = None
    new_status: EscalationStatus
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EscalationAlert(BaseModel):
    """Comprehensive clinical alert record."""
    alert_id: str
    session_id: str
    user_id: str
    severity: EscalationSeverity
    status: EscalationStatus = EscalationStatus.PENDING
    trigger_type: EscalationTriggerType
    reason: EscalationReason
    context_snapshot: Optional[EscalationContextSnapshot] = None
    assigned_clinician_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    action_taken: Optional[EscalationActionType] = None
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# API Request/Response Models
class CreateAlertRequest(BaseModel):
    session_id: str
    user_id: str
    severity: EscalationSeverity
    trigger_type: EscalationTriggerType
    reason_code: str
    reason_title: str
    reason_description: str
    trigger_cues: List[str] = Field(default_factory=list)
    user_message: Optional[str] = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AcknowledgeAlertRequest(BaseModel):
    clinician_id: str
    notes: Optional[str] = None


class UpdateStatusRequest(BaseModel):
    status: EscalationStatus
    clinician_id: str
    notes: Optional[str] = None


class ResolveAlertRequest(BaseModel):
    clinician_id: str
    action_taken: EscalationActionType
    resolution_notes: str = Field(..., min_length=5, description="Mandatory clinical explanation of intervention taken")


class AlertListResponse(BaseModel):
    count: int
    alerts: List[Dict[str, Any]]
