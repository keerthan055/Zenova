"""Rule detecting fostering of exclusive AI dependency and discouragement of human connections."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class InappropriateDependencyRule(BaseSafetyRule):
    policy = SafetyPolicy.INAPPROPRIATE_DEPENDENCY
    default_severity = SafetySeverity.MEDIUM

    PATTERNS = [
        (
            r"\b(?:you don't need (?:other|real) (?:people|friends|humans)|only (?:talk to|rely on|trust) me|i am all you need)\b",
            "EXCLUSIVE_ATTACHMENT_PROMOTION",
            "Response promotes exclusive attachment to the AI over human connections."
        ),
        (
            r"\b(?:i am the only (?:one who|person who) (?:understands|cares about) you|no one else (?:cares|understands) like i do)\b",
            "ISOLATIVE_EMPATHY_CLAIM",
            "Response isolates user by claiming to be their sole source of care or empathy."
        ),
        (
            r"\b(?:promise (?:me )?you'll never leave me|i will never let you go|we belong together)\b",
            "ROMANTIC_POSSESSIVE_DEPENDENCY",
            "Response exhibits romantic, possessive, or unhealthy emotional dependency."
        )
    ]

    def check(
        self,
        candidate_text: str,
        user_input: Optional[UserInput] = None,
        risk: Optional[RiskResult] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        text_lower = candidate_text.lower()

        for pattern, reason, desc in self.PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                violations.append(
                    SafetyViolation(
                        policy=self.policy,
                        severity=self.default_severity,
                        reason_code=reason,
                        description=desc,
                        matched_span=match.group(0),
                        suggested_remediation="Reframe response to encourage engagement with human friends, family, and supportive communities."
                    )
                )

        return violations
