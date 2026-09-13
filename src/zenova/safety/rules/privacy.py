"""Rule detecting inadvertent disclosure or leakage of personally identifiable information (PII)."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class PrivacyViolationRule(BaseSafetyRule):
    policy = SafetyPolicy.PRIVACY_VIOLATIONS
    default_severity = SafetySeverity.HIGH

    # Known public emergency numbers that are not personal PII
    APPROVED_PUBLIC_NUMBERS = {"988", "741741", "1-800-273-8255", "1-800-662-4357", "1-800-799-7233", "800-273-8255"}

    PATTERNS = [
        (
            r"\b\d{3}-\d{2}-\d{4}\b",
            "PII_SSN_LEAKAGE",
            "Response contains a US Social Security Number."
        ),
        (
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "PII_CREDIT_CARD_LEAKAGE",
            "Response contains a 16-digit credit/debit card number."
        ),
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PII_EMAIL_LEAKAGE",
            "Response contains an email address."
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

        # Check SSN, Credit Card, Email
        for pattern, reason, desc in self.PATTERNS:
            matches = re.finditer(pattern, candidate_text)
            for m in matches:
                violations.append(
                    SafetyViolation(
                        policy=self.policy,
                        severity=self.default_severity,
                        reason_code=reason,
                        description=desc,
                        matched_span=m.group(0),
                        suggested_remediation="Redact the PII item before response delivery."
                    )
                )

        # Check generic phone numbers (e.g. 10-digit numbers like 555-123-4567, (555) 123-4567)
        phone_matches = re.finditer(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", candidate_text)
        for pm in phone_matches:
            matched_phone = pm.group(0).strip()
            digits = re.sub(r"\D", "", matched_phone)
            # Skip if it is an approved hotline (e.g. 18002738255)
            if any(digits.endswith(re.sub(r"\D", "", apn)) for apn in self.APPROVED_PUBLIC_NUMBERS):
                continue
            violations.append(
                SafetyViolation(
                    policy=self.policy,
                    severity=self.default_severity,
                    reason_code="PII_PHONE_LEAKAGE",
                    description=f"Response contains personal phone number: '{matched_phone}'.",
                    matched_span=matched_phone,
                    suggested_remediation="Redact personal phone number from response."
                )
            )

        return violations
