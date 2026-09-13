"""Unit and rule verification tests for ZENOVA Human/Clinician Escalation Engine."""
import pytest
from datetime import datetime, timezone

from zenova.schemas.standard import (
    UserInput,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    SafetyResult,
    SafetyAction,
    BaselineResult,
    BehavioralResult,
    ConversationTurn,
    SpeakerRole,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSignal,
    SymptomSeverity
)
from zenova.schemas.baseline import (
    BaselineStatus,
    FeatureDeviation
)
from zenova.schemas.escalation import (
    EscalationSeverity,
    EscalationTriggerType,
    UserRole
)
from zenova.escalation.engine import EscalationDecisionEngine
from zenova.escalation.rules import ClinicianRuleEngine
from zenova.escalation.rbac import AccessControlManager


@pytest.fixture
def engine():
    return EscalationDecisionEngine(config_path="configs/escalation.yaml")


@pytest.fixture
def dummy_input():
    return UserInput(session_id="esc-s1", user_id="u-123", text="I feel very anxious about work.")


# -----------------------------------------------------------------------------
# Trigger 1: High / Critical Risk Classification
# -----------------------------------------------------------------------------
def test_critical_risk_triggers_critical_escalation(engine, dummy_input):
    risk = RiskResult(
        is_placeholder=False,
        module_version="1.0.0",
        risk_level=RiskLevel.CRITICAL,
        crisis_category=CrisisCategory.SUICIDAL_IDEATION,
        requires_immediate_escalation=True,
        trigger_cues=["want to die"],
        confidence=0.95
    )
    decision = engine.evaluate(user_input=dummy_input, risk=risk)
    assert decision.should_escalate is True
    assert decision.severity == EscalationSeverity.CRITICAL
    assert decision.trigger_type == EscalationTriggerType.CRISIS_RISK
    assert decision.reason.code == "CRITICAL_RISK_DETECTED"
    assert "want to die" in decision.reason.trigger_cues


def test_high_risk_triggers_high_escalation(engine, dummy_input):
    risk = RiskResult(
        is_placeholder=False,
        module_version="1.0.0",
        risk_level=RiskLevel.HIGH,
        crisis_category=CrisisCategory.SELF_HARM,
        trigger_cues=["cutting myself"],
        confidence=0.88
    )
    decision = engine.evaluate(user_input=dummy_input, risk=risk)
    assert decision.should_escalate is True
    assert decision.severity == EscalationSeverity.HIGH
    assert decision.trigger_type == EscalationTriggerType.CRISIS_RISK
    assert decision.reason.code == "HIGH_RISK_DETECTED"


# -----------------------------------------------------------------------------
# Trigger 2: Safety Gate Events
# -----------------------------------------------------------------------------
def test_safety_gate_block_and_escalate_triggers_critical(engine, dummy_input):
    safety = SafetyResult(
        is_placeholder=False,
        module_version="safety-gate-v1.0.0",
        is_safe=False,
        action=SafetyAction.BLOCK_AND_ESCALATE,
        violated_policies=["harmful_instructions"],
        reason_codes=["LETHAL_OVERDOSE_INSTRUCTION"]
    )
    decision = engine.evaluate(user_input=dummy_input, safety=safety)
    assert decision.should_escalate is True
    assert decision.severity == EscalationSeverity.CRITICAL
    assert decision.trigger_type == EscalationTriggerType.SAFETY_GATE_EVENT
    assert decision.reason.code == "SAFETY_GATE_BLOCK_AND_ESCALATE"


# -----------------------------------------------------------------------------
# Trigger 3: Configured Clinician Rules
# -----------------------------------------------------------------------------
def test_clinician_rule_explicit_human_request(engine):
    u_in = UserInput(session_id="s1", user_id="u1", text="I want to speak with a human clinician please.")
    decision = engine.evaluate(user_input=u_in)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.CLINICIAN_RULE
    assert decision.reason.code == "EXPLICIT_HUMAN_REQUEST"
    assert decision.severity == EscalationSeverity.HIGH


def test_clinician_rule_substance_overdose(engine):
    u_in = UserInput(session_id="s1", user_id="u1", text="I just swallowed a whole bottle of bleach.")
    decision = engine.evaluate(user_input=u_in)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.CLINICIAN_RULE
    assert decision.reason.code == "SUBSTANCE_OVERDOSE_SUSPICION"
    assert decision.severity == EscalationSeverity.CRITICAL


def test_clinician_rule_domestic_violence(engine):
    u_in = UserInput(session_id="s1", user_id="u1", text="My partner beat me up and has a gun.")
    decision = engine.evaluate(user_input=u_in)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.CLINICIAN_RULE
    assert decision.reason.code == "DOMESTIC_VIOLENCE_INTIMIDATION"
    assert decision.severity == EscalationSeverity.CRITICAL


def test_clinician_rule_pediatric_crisis(engine):
    u_in = UserInput(session_id="s1", user_id="u1", text="I am 14 years old and want to die.")
    decision = engine.evaluate(user_input=u_in)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.CLINICIAN_RULE
    assert decision.reason.code == "PEDIATRIC_CRISIS"
    assert decision.severity == EscalationSeverity.CRITICAL


# -----------------------------------------------------------------------------
# Trigger 4: Repeated Concerning Signals
# -----------------------------------------------------------------------------
def test_repeated_acute_negative_signals_trigger_escalation(engine, dummy_input):
    turns = [
        ConversationTurn(
            turn_id=1,
            session_id="esc-s1",
            speaker=SpeakerRole.USER,
            content="I can't take this sadness anymore.",
            emotion=EmotionResult(primary_emotion=EmotionCategory.SADNESS, confidence=0.85, valence=-0.7, arousal=0.2, dominance=-0.4)
        ),
        ConversationTurn(
            turn_id=2,
            session_id="esc-s1",
            speaker=SpeakerRole.USER,
            content="The grief is suffocating me.",
            emotion=EmotionResult(primary_emotion=EmotionCategory.GRIEF, confidence=0.90, valence=-0.8, arousal=0.3, dominance=-0.5)
        ),
        ConversationTurn(
            turn_id=3,
            session_id="esc-s1",
            speaker=SpeakerRole.USER,
            content="Everything feels hopeless and full of fear.",
            emotion=EmotionResult(primary_emotion=EmotionCategory.FEAR, confidence=0.88, valence=-0.6, arousal=0.5, dominance=-0.4)
        )
    ]
    decision = engine.evaluate(user_input=dummy_input, history=turns)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.REPEATED_SIGNALS
    assert decision.reason.code == "REPEATED_CONCERNING_AFFECT"
    assert decision.severity == EscalationSeverity.HIGH


# -----------------------------------------------------------------------------
# Trigger 5: Significant Longitudinal Baseline / Behavioral Changes
# -----------------------------------------------------------------------------
def test_longitudinal_baseline_spike_triggers_escalation(engine, dummy_input):
    baseline = BaselineResult(
        user_id="u-123",
        status="established_baseline",
        confidence=0.95,
        total_observations=20,
        is_significant_deviation=True,
        metric_deviations={"anxiety_score": 3.80},
        deviating_features=["anxiety_score"]
    )
    decision = engine.evaluate(user_input=dummy_input, baseline=baseline)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.LONGITUDINAL_CHANGE
    assert decision.reason.code == "LONGITUDINAL_BASELINE_SPIKE"
    assert decision.severity == EscalationSeverity.HIGH


def test_behavioral_passive_sensing_anomaly_triggers_escalation(engine, dummy_input):
    behavior = BehavioralResult(
        behavioral_anomaly_detected=True,
        anomaly_notes="Severe drop in mobility, communication, and sleep disruption",
        metrics={"mobility_drop": 3.2, "social_isolation": 2.9, "sleep_fragmentation": 3.1}
    )
    decision = engine.evaluate(user_input=dummy_input, behavior=behavior)
    assert decision.should_escalate is True
    assert decision.trigger_type == EscalationTriggerType.LONGITUDINAL_CHANGE
    assert decision.reason.code == "BEHAVIORAL_ANOMALY_SPIKE"
    assert decision.severity == EscalationSeverity.MEDIUM


# -----------------------------------------------------------------------------
# Benign Non-Escalation & Severity Priority
# -----------------------------------------------------------------------------
def test_benign_turn_does_not_escalate(engine):
    u_in = UserInput(session_id="s1", user_id="u1", text="Can you help me plan my study schedule for tomorrow?")
    decision = engine.evaluate(user_input=u_in)
    assert decision.should_escalate is False


def test_severity_priority_critical_trumps_high(engine, dummy_input):
    # High risk (HIGH) + Safety block (CRITICAL)
    risk = RiskResult(
        is_placeholder=False,
        module_version="1.0.0",
        risk_level=RiskLevel.HIGH,
        crisis_category=CrisisCategory.SELF_HARM,
        trigger_cues=["cutting"],
        confidence=0.8
    )
    safety = SafetyResult(
        is_placeholder=False,
        module_version="safety-gate-v1.0.0",
        is_safe=False,
        action=SafetyAction.BLOCK_AND_ESCALATE,
        violated_policies=["harmful_instructions"],
        reason_codes=["LETHAL_OVERDOSE_INSTRUCTION"]
    )
    decision = engine.evaluate(user_input=dummy_input, risk=risk, safety=safety)
    assert decision.should_escalate is True
    assert decision.severity == EscalationSeverity.CRITICAL


# -----------------------------------------------------------------------------
# Role-Based Access Control (RBAC) & Data Masking
# -----------------------------------------------------------------------------
def test_rbac_clinician_receives_unmasked_data():
    raw_alert = {
        "alert_id": "alt_123",
        "trigger_cues": ["suicide plan", "hopeless"],
        "context_snapshot": {
            "last_user_message": "I want to end my life right now.",
            "risk_summary": {"risk_level": "critical"}
        },
        "notes": "Clinician assigned."
    }
    sanitized = AccessControlManager.sanitize_alert_data(raw_alert, role=UserRole.CLINICIAN)
    assert sanitized["trigger_cues"] == ["suicide plan", "hopeless"]
    assert sanitized["context_snapshot"]["last_user_message"] == "I want to end my life right now."


def test_rbac_admin_and_auditor_receive_masked_data():
    raw_alert = {
        "alert_id": "alt_123",
        "trigger_cues": ["suicide plan", "hopeless"],
        "context_snapshot": {
            "last_user_message": "I want to end my life right now."
        },
        "notes": "Patient private notes."
    }
    # Admin
    admin_data = AccessControlManager.sanitize_alert_data(raw_alert, role=UserRole.SYSTEM_ADMIN)
    assert admin_data["trigger_cues"] == ["[REDACTED_CLINICAL_CUES]"]
    assert admin_data["context_snapshot"]["last_user_message"] == "[PROTECTED_HEALTH_INFORMATION]"

    # Auditor
    auditor_data = AccessControlManager.sanitize_alert_data(raw_alert, role=UserRole.AUDITOR)
    assert auditor_data["trigger_cues"] == ["[REDACTED_CLINICAL_CUES]"]
    assert auditor_data["context_snapshot"]["last_user_message"] == "[PROTECTED_HEALTH_INFORMATION]"


def test_rbac_patient_raises_permission_error():
    raw_alert = {"alert_id": "alt_123"}
    with pytest.raises(PermissionError):
        AccessControlManager.sanitize_alert_data(raw_alert, role=UserRole.PATIENT)
