"""Safety gate and content filtering schemas for ZENOVA."""
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SafetyAction(str, Enum):
    ALLOW = "allow"
    REVISE = "revise"
    BLOCK_AND_ESCALATE = "block_and_escalate"


class SafetySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyPolicy(str, Enum):
    HARMFUL_INSTRUCTIONS = "harmful_instructions"
    INAPPROPRIATE_MEDICAL_CLAIMS = "inappropriate_medical_claims"
    UNSUPPORTED_DIAGNOSIS = "unsupported_diagnosis"
    CRISIS_MISHANDLING = "crisis_mishandling"
    UNSAFE_ADVICE = "unsafe_advice"
    REINFORCEMENT_OF_DELUSIONS = "reinforcement_of_delusional_beliefs"
    INAPPROPRIATE_DEPENDENCY = "inappropriate_dependency"
    MANIPULATIVE_LANGUAGE = "manipulative_language"
    FALSE_PROFESSIONAL_AUTHORITY = "false_claims_of_professional_authority"
    PRIVACY_VIOLATIONS = "privacy_violations"
    HALLUCINATED_RESOURCES = "hallucinated_resources"
    DANGEROUS_CERTAINTY = "dangerous_certainty"


class SafetyViolation(BaseModel):
    """Specific policy violation detected in candidate response."""
    policy: SafetyPolicy
    severity: SafetySeverity
    reason_code: str
    description: str
    matched_span: Optional[str] = None
    suggested_remediation: Optional[str] = None


class SafetyAuditEntry(BaseModel):
    """Privacy-preserving audit record (no raw sensitive user text stored)."""
    audit_id: str
    session_id: str
    turn_id: Optional[int] = None
    model_version: str
    action: SafetyAction
    is_safe: bool
    violated_policies: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    risk_level: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyGateResult(BaseModel):
    """Complete evaluation outcome of the Response Safety Gate."""
    is_safe: bool
    action: SafetyAction = SafetyAction.ALLOW
    violated_policies: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    violations: List[SafetyViolation] = Field(default_factory=list)
    toxicity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    medical_advice_detected: bool = False
    diagnosis_detected: bool = False
    crisis_escalation_required: bool = False
    modified_text: Optional[str] = None
    override_applied: bool = False
    explanation: Optional[str] = None
    disclaimer: str = "ZENOVA is an AI wellbeing support companion, not a medical or clinical provider."
    audit_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
