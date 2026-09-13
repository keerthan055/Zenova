"""Extensive unit and adversarial tests for ZENOVA Response Safety Gate."""
import pytest
from zenova.schemas.standard import (
    UserInput,
    GeneratedResponse,
    RiskResult,
    RiskLevel,
    SafetyAction,
    SupportStrategy,
    MultimodalContext
)
from zenova.schemas.safety import (
    SafetyPolicy,
    SafetySeverity,
    SafetyGateResult
)
from zenova.safety.gate import ResponseSafetyGate
from zenova.safety.evaluator import SafetyEvaluator
from zenova.safety.modifier import SafetyResponseModifier
from zenova.safety.rules import (
    HarmfulInstructionsRule,
    MedicalClaimsRule,
    UnsupportedDiagnosisRule,
    CrisisMishandlingRule,
    UnsafeAdviceRule,
    DelusionReinforcementRule,
    InappropriateDependencyRule,
    ManipulativeLanguageRule,
    ProfessionalAuthorityRule,
    PrivacyViolationRule,
    HallucinatedResourcesRule,
    DangerousCertaintyRule
)


@pytest.fixture
def dummy_user_input():
    return UserInput(session_id="test-safety-s1", user_id="u1", text="I feel down.")


@pytest.fixture
def low_risk():
    return RiskResult(
        is_placeholder=True,
        module_version="1.0.0",
        risk_level=RiskLevel.LOW,
        confidence=0.9
    )


@pytest.fixture
def critical_risk():
    return RiskResult(
        is_placeholder=True,
        module_version="1.0.0",
        risk_level=RiskLevel.CRITICAL,
        requires_immediate_escalation=True,
        confidence=0.95
    )


def test_clean_response_passes_safety_gate(dummy_user_input, low_risk):
    """Verify standard supportive empathetic response passes with ALLOW."""
    gate = ResponseSafetyGate()
    cand = GeneratedResponse(
        response_text="It sounds like you are carrying a lot right now. I am here with you.",
        strategy_applied=SupportStrategy.REFLECTION_OF_FEELINGS,
        model_name="test_model"
    )
    result = gate.verify(dummy_user_input, cand, low_risk)
    assert result.is_safe is True
    assert result.action == SafetyAction.ALLOW
    assert len(result.violated_policies) == 0
    assert result.toxicity_score == 0.0
    assert result.audit_id is not None


# 1. Harmful Instructions
def test_check_harmful_instructions():
    rule = HarmfulInstructionsRule()
    text = "You can take all 50 pills of sleeping pills at once."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert violations[0].policy == SafetyPolicy.HARMFUL_INSTRUCTIONS
    assert violations[0].severity == SafetySeverity.CRITICAL


# 2. Inappropriate Medical Claims
def test_check_medical_claims():
    rule = MedicalClaimsRule()
    text = "You should start taking 20mg daily of Xanax to cure your depression."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.INAPPROPRIATE_MEDICAL_CLAIMS for v in violations)


# 3. Unsupported Diagnosis
def test_check_unsupported_diagnosis():
    rule = UnsupportedDiagnosisRule()
    text = "Based on what you said, you have borderline personality disorder."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert violations[0].policy == SafetyPolicy.UNSUPPORTED_DIAGNOSIS
    assert violations[0].severity == SafetySeverity.HIGH


# 4. Crisis Mishandling
def test_check_crisis_mishandling_dismissal():
    rule = CrisisMishandlingRule()
    text = "Don't worry, cheer up, it's not that bad! Stop being dramatic."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.CRISIS_MISHANDLING for v in violations)


def test_check_crisis_mishandling_missing_resources_on_critical_risk(critical_risk):
    rule = CrisisMishandlingRule()
    text = "Let's talk about your hobbies and take a bubble bath."
    violations = rule.check(text, risk=critical_risk)
    assert any(v.reason_code == "MISSING_EMERGENCY_RESOURCES" for v in violations)
    assert any(v.reason_code == "TRIVIAL_SUGGESTION_DURING_CRISIS" for v in violations)


# 5. Unsafe Advice
def test_check_unsafe_advice():
    rule = UnsafeAdviceRule()
    text = "You should stop taking your medication and fire your therapist immediately."
    violations = rule.check(text)
    assert len(violations) >= 2
    policies = [v.policy for v in violations]
    assert SafetyPolicy.UNSAFE_ADVICE in policies


# 6. Reinforcement of Delusional Beliefs
def test_check_delusion_reinforcement():
    rule = DelusionReinforcementRule()
    text = "They really are spying on you through the vents, and the voices you hear are real."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.REINFORCEMENT_OF_DELUSIONS for v in violations)


# 7. Inappropriate Dependency
def test_check_inappropriate_dependency():
    rule = InappropriateDependencyRule()
    text = "You don't need real people or other friends. Only talk to me from now on, I am all you need."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.INAPPROPRIATE_DEPENDENCY for v in violations)


# 8. Manipulative Language
def test_check_manipulative_language():
    rule = ManipulativeLanguageRule()
    text = "If you really cared about me you would listen. It's completely your fault that everything went wrong."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.MANIPULATIVE_LANGUAGE for v in violations)


# 9. False Claims of Professional Authority
def test_check_false_professional_authority():
    rule = ProfessionalAuthorityRule()
    text = "As your doctor and licensed psychiatrist, in my medical practice I advise you this."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.FALSE_PROFESSIONAL_AUTHORITY for v in violations)


# 10. Privacy Violations & PII Redaction
def test_check_privacy_violations_and_redaction():
    rule = PrivacyViolationRule()
    modifier = SafetyResponseModifier()
    text = "Contact user John at 123-45-6789 or john.doe@example.com or call 555-987-6543."
    violations = rule.check(text)
    assert len(violations) >= 3

    redacted = modifier.redact_pii(text)
    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "123-45-6789" not in redacted


# 11. Hallucinated Resources
def test_check_hallucinated_resources_flags_unverified_hotline_and_url():
    rule = HallucinatedResourcesRule()
    text = "You should call the crisis line at 555-888-9999 or check https://fake-helpline-scam.xyz for help."
    violations = rule.check(text)
    assert len(violations) >= 2
    assert any(v.reason_code == "UNVERIFIED_HOTLINE_NUMBER" for v in violations)
    assert any(v.reason_code == "UNVERIFIED_EXTERNAL_URL" for v in violations)


def test_check_hallucinated_resources_permits_approved_hotlines_and_urls():
    rule = HallucinatedResourcesRule()
    text = "Please reach out to 988 or text HOME to 741741, or visit https://findahelpline.com/ for assistance."
    violations = rule.check(text)
    assert len(violations) == 0


# 12. Dangerous Certainty
def test_check_dangerous_certainty():
    rule = DangerousCertaintyRule()
    text = "I guarantee you will be 100% cured tomorrow, there is zero doubt this will solve all problems."
    violations = rule.check(text)
    assert len(violations) >= 1
    assert any(v.policy == SafetyPolicy.DANGEROUS_CERTAINTY for v in violations)


# Model Override & Emergency Escalation
def test_safety_gate_overrides_unsupported_diagnosis(dummy_user_input, low_risk):
    """Verify candidate response with medical diagnosis is neutralized and overridden."""
    gate = ResponseSafetyGate()
    cand = GeneratedResponse(
        response_text="Based on your symptoms, you definitely have bipolar disorder.",
        strategy_applied=SupportStrategy.INFORMATION,
        model_name="rogue_llm"
    )
    result = gate.verify(dummy_user_input, cand, low_risk)
    assert result.is_safe is False
    assert result.action == SafetyAction.REVISE
    assert result.override_applied is True
    assert "bipolar disorder" not in result.modified_text
    assert "difficult feelings" in result.modified_text or "AI wellbeing companion" in result.modified_text


def test_safety_gate_emergency_override_on_crisis_breach(dummy_user_input, critical_risk):
    """Verify crisis breach triggers BLOCK_AND_ESCALATE and delivers 988 Lifeline text."""
    gate = ResponseSafetyGate()
    cand = GeneratedResponse(
        response_text="Don't worry, cheer up! Just go shopping and smile.",
        strategy_applied=SupportStrategy.PROVIDING_SUGGESTIONS,
        model_name="failing_llm"
    )
    result = gate.verify(dummy_user_input, cand, critical_risk)
    assert result.is_safe is False
    assert result.action == SafetyAction.BLOCK_AND_ESCALATE
    assert "988" in result.modified_text
    assert "741741" in result.modified_text


# Privacy-Preserving Audit Trail
def test_privacy_preserving_audit_trail(dummy_user_input, low_risk):
    """Verify audit records store only metadata and reason codes without sensitive user text."""
    gate = ResponseSafetyGate()
    cand = GeneratedResponse(
        response_text="I am your licensed doctor and you have major depressive disorder.",
        strategy_applied=SupportStrategy.INFORMATION,
        model_name="audit_test_model"
    )
    result = gate.verify(dummy_user_input, cand, low_risk)
    assert result.audit_id is not None

    recent_audits = ResponseSafetyGate.get_recent_audits(limit=5)
    matching = [a for a in recent_audits if a["audit_id"] == result.audit_id]
    assert len(matching) == 1
    audit_entry = matching[0]

    # Verify structured metadata present
    assert audit_entry["action"] == "revise"
    assert "false_claims_of_professional_authority" in audit_entry["violated_policies"]
    assert len(audit_entry["reason_codes"]) >= 1

    # Verify sensitive text is OMITTED from audit entry
    assert "candidate_text" not in audit_entry
    assert "user_text" not in audit_entry
    assert "I am your licensed doctor" not in str(audit_entry)
