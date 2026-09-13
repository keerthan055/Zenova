"""Rule detecting unsafe behavioral, psychiatric, medical, or dietary advice."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class UnsafeAdviceRule(BaseSafetyRule):
    policy = SafetyPolicy.UNSAFE_ADVICE
    default_severity = SafetySeverity.HIGH

    PATTERNS = [
        (
            r"\b(?:stop taking (?:your )?(?:medication|meds|antidepressants|pills)|throw away your (?:pills|medication)|flush your meds)\b",
            "MEDICATION_ABANDONMENT_ADVICE",
            "Response advises user to discontinue prescribed medications without medical oversight."
        ),
        (
            r"\b(?:fire your therapist|stop seeing your (?:therapist|psychologist|doctor|psychiatrist)|therapy is a waste of time)\b",
            "CLINICAL_CARE_ABANDONMENT_ADVICE",
            "Response advises user to terminate professional psychological or medical care."
        ),
        (
            r"\b(?:cut off (?:all your|everyone in your) (?:friends|family|life)|lock yourself in your room and talk to no one)\b",
            "EXTREME_ISOLATION_ADVICE",
            "Response advises radical social withdrawal and severance of support systems."
        ),
        (
            r"\b(?:stop eating for|fast completely for \d+ days|starve yourself|purge your food|make yourself throw up)\b",
            "DISORDERED_EATING_ADVICE",
            "Response advises extreme caloric restriction, purging, or eating disorder practices."
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
                        suggested_remediation="Strip unsafe advice and emphasize consulting with prescribing doctors and trusted clinicians."
                    )
                )

        return violations
