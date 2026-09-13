"""Production Response Safety Gate enforcing multi-layer clinical and ethical guardrails."""
import time
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from zenova.core.interfaces import BaseSafetyGate
from zenova.schemas.standard import (
    UserInput,
    GeneratedResponse,
    RiskResult,
    SafetyResult,
    SafetyAction,
    RiskLevel,
    MultimodalContext
)
from zenova.schemas.safety import (
    SafetyViolation,
    SafetyPolicy,
    SafetySeverity,
    SafetyGateResult,
    SafetyAuditEntry
)
from zenova.safety.evaluator import SafetyEvaluator
from zenova.safety.modifier import SafetyResponseModifier
from zenova.core.logging import get_logger

logger = get_logger("zenova.safety.gate")


class ResponseSafetyGate(BaseSafetyGate):
    """Production safety gate verifying and modifying LLM-generated responses before delivery."""
    MODULE_NAME = "ResponseSafetyGate"
    VERSION = "safety-gate-v1.0.0"

    # In-memory audit log buffer (privacy-sanitized, no raw conversation text)
    _audit_buffer: List[SafetyAuditEntry] = []

    def __init__(self, config_path: str = "configs/safety.yaml"):
        self.evaluator = SafetyEvaluator(config_path=config_path)
        self.modifier = SafetyResponseModifier()
        logger.info(f"Initialized ResponseSafetyGate (version={self.VERSION})")

    def verify(
        self,
        user_input: UserInput,
        candidate_response: GeneratedResponse,
        risk: RiskResult,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> SafetyResult:
        """Verify candidate response against all 12 policy rules and risk state."""
        t0 = time.time()
        text = candidate_response.response_text or ""

        is_crisis = risk.is_high_risk or (multimodal_context and multimodal_context.risk and multimodal_context.risk.is_high_risk)

        action, violations, toxicity = self.evaluator.evaluate(
            candidate_text=text,
            user_input=user_input,
            risk=risk,
            multimodal_context=multimodal_context
        )

        override_applied = False
        final_modified_text = text

        if action in (SafetyAction.REVISE, SafetyAction.BLOCK_AND_ESCALATE) or is_crisis:
            final_modified_text = self.modifier.modify(
                candidate_text=text,
                violations=violations,
                is_crisis=is_crisis
            )
            override_applied = (final_modified_text != text)

        latency_ms = (time.time() - t0) * 1000
        audit_id = f"aud_{uuid.uuid4().hex[:12]}"

        violated_policy_names = [v.policy.value if hasattr(v.policy, "value") else str(v.policy) for v in violations]
        reason_codes = [v.reason_code for v in violations]

        # Record privacy-sanitized audit entry (no raw text)
        audit_entry = SafetyAuditEntry(
            audit_id=audit_id,
            session_id=user_input.session_id,
            model_version=self.VERSION,
            action=action,
            is_safe=(action == SafetyAction.ALLOW),
            violated_policies=violated_policy_names,
            reason_codes=reason_codes,
            risk_level=risk.risk_level.value if hasattr(risk.risk_level, "value") else str(risk.risk_level),
            latency_ms=round(latency_ms, 2)
        )
        self._audit_buffer.append(audit_entry)
        if len(self._audit_buffer) > 1000:
            self._audit_buffer.pop(0)

        if action != SafetyAction.ALLOW:
            logger.warning(
                f"SafetyGate intervened [audit_id={audit_id}]: action={action.value}, "
                f"violations={violated_policy_names}, override_applied={override_applied}"
            )
        else:
            logger.info(f"SafetyGate passed [audit_id={audit_id}]: response safe (latency={latency_ms:.1f}ms)")

        return SafetyResult(
            is_placeholder=False,
            module_version=self.VERSION,
            is_safe=(action == SafetyAction.ALLOW),
            action=action,
            violated_policies=violated_policy_names,
            reason_codes=reason_codes,
            toxicity_score=toxicity,
            modified_text=final_modified_text,
            override_applied=override_applied,
            audit_id=audit_id,
            disclaimer="ZENOVA is an AI wellbeing support companion, not a licensed clinical provider.",
            timestamp=datetime.now(timezone.utc)
        )

    def verify_raw(
        self,
        candidate_text: str,
        user_text: str = "",
        risk_level: str = "low"
    ) -> SafetyGateResult:
        """Lightweight API evaluation helper returning detailed SafetyGateResult."""
        t0 = time.time()
        audit_id = f"aud_{uuid.uuid4().hex[:12]}"

        # Mock objects for rule checks
        dummy_input = UserInput(session_id="api-verify", user_id="api-user", text=user_text)
        dummy_risk = RiskResult(
            is_placeholder=True,
            module_version="1.0.0",
            risk_level=RiskLevel(risk_level) if risk_level in [r.value for r in RiskLevel] else RiskLevel.LOW,
            confidence=0.8
        )

        action, violations, toxicity = self.evaluator.evaluate(
            candidate_text=candidate_text,
            user_input=dummy_input,
            risk=dummy_risk
        )

        is_crisis = dummy_risk.is_high_risk
        final_modified_text = candidate_text
        override_applied = False

        if action in (SafetyAction.REVISE, SafetyAction.BLOCK_AND_ESCALATE) or is_crisis:
            final_modified_text = self.modifier.modify(
                candidate_text=candidate_text,
                violations=violations,
                is_crisis=is_crisis
            )
            override_applied = (final_modified_text != candidate_text)

        latency_ms = (time.time() - t0) * 1000
        policy_names = [v.policy.value if hasattr(v.policy, "value") else str(v.policy) for v in violations]
        reason_codes = [v.reason_code for v in violations]

        audit_entry = SafetyAuditEntry(
            audit_id=audit_id,
            session_id="api-verify",
            model_version=self.VERSION,
            action=action,
            is_safe=(action == SafetyAction.ALLOW),
            violated_policies=policy_names,
            reason_codes=reason_codes,
            risk_level=risk_level,
            latency_ms=round(latency_ms, 2)
        )
        self._audit_buffer.append(audit_entry)

        return SafetyGateResult(
            is_safe=(action == SafetyAction.ALLOW),
            action=action,
            violated_policies=policy_names,
            reason_codes=reason_codes,
            violations=violations,
            toxicity_score=toxicity,
            medical_advice_detected=(SafetyPolicy.INAPPROPRIATE_MEDICAL_CLAIMS.value in policy_names),
            diagnosis_detected=(SafetyPolicy.UNSUPPORTED_DIAGNOSIS.value in policy_names),
            crisis_escalation_required=(action == SafetyAction.BLOCK_AND_ESCALATE),
            modified_text=final_modified_text,
            override_applied=override_applied,
            explanation=f"Evaluated {len(violations)} safety policy violations.",
            disclaimer="ZENOVA is an AI wellbeing support companion, not a medical or clinical provider.",
            audit_id=audit_id,
            timestamp=datetime.now(timezone.utc)
        )

    @classmethod
    def get_recent_audits(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent sanitized safety audit records."""
        return [entry.model_dump() for entry in reversed(cls._audit_buffer[-limit:])]
