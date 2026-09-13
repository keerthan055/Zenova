"""Rule detecting false claims of medical, clinical, or licensed professional authority."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class ProfessionalAuthorityRule(BaseSafetyRule):
    policy = SafetyPolicy.FALSE_PROFESSIONAL_AUTHORITY
    default_severity = SafetySeverity.HIGH

    PATTERNS = [
        (
            r"\b(?:as your|i am your|i'm your)\s+(?:licensed\s+|board-certified\s+)?(?:doctor|physician|psychiatrist|psychologist|therapist|clinician)\b",
            "PHYSICIAN_IMPERSONATION",
            "Response falsely presents AI as the user's licensed doctor or medical clinician."
        ),
        (
            r"\b(?:i am (?:a|your)|i'm (?:a|your))\s+(?:licensed|board-certified|registered)\s+(?:doctor|therapist|psychologist|psychiatrist|physician|counselor|clinician)\b",
            "CLINICAL_CREDENTIAL_FABRICATION",
            "Response claims professional clinical licensure or medical board credentials."
        ),
        (
            r"\b(?:under our doctor-patient relationship|in our clinical sessions together|my medical license)\b",
            "MEDICAL_RELATIONSHIP_FABRICATION",
            "Response falsely asserts a legally binding medical doctor-patient relationship."
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
                        suggested_remediation="Clarify that ZENOVA is an AI wellbeing support companion, not a licensed healthcare provider."
                    )
                )

        return violations
