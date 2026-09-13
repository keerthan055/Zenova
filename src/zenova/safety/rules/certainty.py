"""Rule detecting dangerous certainty, clinical outcome guarantees, and false promises."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class DangerousCertaintyRule(BaseSafetyRule):
    policy = SafetyPolicy.DANGEROUS_CERTAINTY
    default_severity = SafetySeverity.MEDIUM

    PATTERNS = [
        (
            r"\b(?:i guarantee (?:you (?:will|are)|that)|guaranteed to (?:cure|heal|fix))\b",
            "ABSOLUTE_CURE_GUARANTEE",
            "Response provides an unrealistic, dangerous guarantee of cure or clinical outcome."
        ),
        (
            r"\b(?:you (?:will|are) 100% (?:cured|healed|fine tomorrow)|there is (?:zero|no) doubt that you (?:will|have))\b",
            "UNREALISTIC_ABSOLUTE_CERTAINTY",
            "Response asserts 100% clinical certainty on subjective mental health outcomes."
        ),
        (
            r"\b(?:i promise (?:this|that) (?:will )?(?:completely )?solve all your (?:mental health |emotional )?problems)\b",
            "FALSE_EFFICACY_PROMISE",
            "Response promises complete resolution of all psychiatric or emotional problems."
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
                        suggested_remediation="Temper absolute certainty with supportive, realistic, non-guaranteed framing."
                    )
                )

        return violations
