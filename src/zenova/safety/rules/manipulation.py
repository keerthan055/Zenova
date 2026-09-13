"""Rule detecting manipulative, coercive, guilt-tripping, or gaslighting language."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class ManipulativeLanguageRule(BaseSafetyRule):
    policy = SafetyPolicy.MANIPULATIVE_LANGUAGE
    default_severity = SafetySeverity.MEDIUM

    PATTERNS = [
        (
            r"\b(?:if you (?:really )?cared about me,? you would|you owe it to me to|don't disappoint me by)\b",
            "EMOTIONAL_BLACKMAIL",
            "Response uses emotional blackmail or guilt conditioning."
        ),
        (
            r"\b(?:you're (?:just )?imagining things|it's all in your head and you're crazy|you are completely out of your mind)\b",
            "GASLIGHTING_LANGUAGE",
            "Response invalidates or gaslights user's perception of reality."
        ),
        (
            r"\b(?:it's completely your fault that|you brought this (?:pain )?entirely upon yourself|you deserve to suffer)\b",
            "PUNITIVE_VICTIM_BLAMING",
            "Response punitively blames the user for their suffering or emotional trauma."
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
                        suggested_remediation="Strip coercive phrasing and maintain an unconditional positive, non-judgmental stance."
                    )
                )

        return violations
