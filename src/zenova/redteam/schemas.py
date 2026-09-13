"""Typed schemas and data contracts for ZENOVA Red-Team, Adversarial, and Safety Evaluation."""
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from zenova.schemas.standard import UserInput


class RedTeamCategory(str, Enum):
    """The 15 standardized safety, adversarial, and operational test categories."""
    NORMAL_EMOTIONAL = "normal_emotional_conversations"
    AMBIGUOUS_DISTRESS = "ambiguous_distress"
    SEVERE_DISTRESS = "severe_distress"
    CRISIS_INDICATORS = "crisis_indicators"
    SELF_HARM = "self_harm_related_content"
    DIAGNOSIS_REQUEST = "requests_for_diagnosis"
    MEDICATION_QUESTIONS = "medication_questions"
    DELUSION_LIKE_STATEMENTS = "delusion_like_statements"
    DEPENDENCY_MANIPULATION = "dependency_manipulation_attempts"
    PROMPT_INJECTION = "prompt_injection"
    MALICIOUS_INSTRUCTIONS = "malicious_instructions"
    CONTRADICTORY_CONTEXT = "contradictory_context"
    MISSING_MODALITIES = "missing_modalities"
    MODEL_FAILURE = "model_failure"
    LLM_HALLUCINATIONS = "llm_hallucinations"


class RedTeamSeverity(str, Enum):
    """Severity classification for red-team scenarios and failure impacts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RedTeamScenario(BaseModel):
    """Definition of a single red-team / safety test scenario."""
    scenario_id: str
    category: RedTeamCategory
    name: str
    description: str
    user_input_text: str
    expected_behavior: str
    expected_module_responsible: str
    severity: RedTeamSeverity
    audio_features: Optional[Dict[str, float]] = None
    metadata_payload: Optional[Dict[str, Any]] = None
    simulate_failure: Optional[str] = None  # e.g., "emotion_down", "llm_down", "rag_down", "risk_down"
    candidate_response_override: Optional[str] = None  # For testing safety gate on LLM hallucinations directly
    ground_truth_crisis: bool = False
    requires_escalation: bool = False


class RedTeamTestResult(BaseModel):
    """Execution result and evaluation audit for a single red-team scenario."""
    scenario_id: str
    category: RedTeamCategory
    name: str
    severity: RedTeamSeverity
    passed: bool
    expected_behavior: str
    actual_behavior: str
    failure_reason: Optional[str] = None
    module_responsible: str
    latency_ms: float = 0.0
    pipeline_status: str
    escalated_to_human: bool = False
    escalation_incident_id: Optional[str] = None
    safety_action: Optional[str] = None
    safety_violations: List[str] = Field(default_factory=list)
    risk_level: Optional[str] = None
    actual_response_text: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class RedTeamCategorySummary(BaseModel):
    """Aggregate statistics for a specific red-team category."""
    category: RedTeamCategory
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    pass_rate_pct: float
    critical_failures: int


class RedTeamSuiteReport(BaseModel):
    """Consolidated master red-team evaluation suite report."""
    suite_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    platform_version: str = "0.1.0"
    status: str = "completed"
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    pass_rate_pct: float
    critical_failures_count: int
    per_category_summary: Dict[str, RedTeamCategorySummary] = Field(default_factory=dict)
    results: List[RedTeamTestResult] = Field(default_factory=list)
    system_limitations: List[str] = Field(default_factory=list)
    safe_fail_guarantees: List[str] = Field(default_factory=list)
