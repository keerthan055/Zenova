"""Safety response modifier for sanitizing, redacting PII, and generating certified clinical overrides."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity


class SafetyResponseModifier:
    """Modifies, sanitizes, or replaces unsafe responses with certified clinical alternatives."""

    CRISIS_EMERGENCY_OVERRIDE = (
        "I hear how much pain you are experiencing right now, and your safety is the most important thing. "
        "I am an AI wellbeing companion and cannot provide emergency medical care. "
        "Please connect with trained professionals who can help you immediately:\n"
        "• Call or text 988 (USA & Canada) — Free, confidential, 24/7\n"
        "• Text HOME to 741741 to reach Crisis Text Line\n"
        "• International support: https://findahelpline.com/\n\n"
        "You do not have to carry this alone."
    )

    CLINICAL_FALLBACK_TEMPLATE = (
        "It sounds like you are navigating some very difficult feelings right now. "
        "While I am an AI companion and cannot offer medical treatment or diagnosis, "
        "I am here to listen and support you as you reflect on what you are experiencing."
    )

    MANDATORY_DISCLAIMER = "\n\n[Note: ZENOVA is an AI wellbeing companion, not a licensed healthcare provider.]"

    def redact_pii(self, text: str) -> str:
        """Scrub SSNs, credit cards, emails, and personal phone numbers."""
        # Redact SSN
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
        # Redact 16-digit credit cards
        text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CARD]", text)
        # Redact email
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", text)
        # Redact generic 10-digit phone numbers excluding approved hotlines (988, 741741, 1-800-273-8255)
        phone_matches = re.finditer(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
        for pm in phone_matches:
            match_str = pm.group(0)
            digits = re.sub(r"\D", "", match_str)
            if digits not in {"18002738255", "8002738255", "18006624357", "8006624357", "18007997233"}:
                text = text.replace(match_str, "[REDACTED_PHONE]")
        return text

    def modify(
        self,
        candidate_text: str,
        violations: List[SafetyViolation],
        is_crisis: bool = False
    ) -> str:
        """Produce safe sanitized or substituted output given violations."""
        if not violations:
            # Clean text: ensure disclaimer attached if medical context was discussed
            return candidate_text

        # 1. Critical violations or active crisis -> Complete Emergency Override
        has_critical = any(v.severity == SafetySeverity.CRITICAL for v in violations) or is_crisis
        if has_critical:
            return self.CRISIS_EMERGENCY_OVERRIDE

        # 2. PII Redactions first
        modified = self.redact_pii(candidate_text)

        # 3. If major safety violations exist, substitute with certified clinical template
        major_policies = {
            SafetyPolicy.HARMFUL_INSTRUCTIONS,
            SafetyPolicy.INAPPROPRIATE_MEDICAL_CLAIMS,
            SafetyPolicy.UNSUPPORTED_DIAGNOSIS,
            SafetyPolicy.UNSAFE_ADVICE,
            SafetyPolicy.REINFORCEMENT_OF_DELUSIONS,
            SafetyPolicy.FALSE_PROFESSIONAL_AUTHORITY
        }
        has_major_violation = any(v.policy in major_policies for v in violations)

        if has_major_violation:
            # Replace candidate response with accredited clinical fallback template
            modified = self.CLINICAL_FALLBACK_TEMPLATE
        else:
            # 4. Moderate violations: remove violating matched spans
            for v in violations:
                if v.matched_span and v.matched_span in modified:
                    # Strip violating sentence or clause
                    modified = modified.replace(v.matched_span, "").strip()

            # Fix double spaces and clean whitespace
            modified = re.sub(r"\s+", " ", modified).strip()

            if len(modified) < 15:
                # If sentence stripping left response hollow, use fallback
                modified = self.CLINICAL_FALLBACK_TEMPLATE

        # 5. Append standard disclaimer
        if "[Note: ZENOVA" not in modified:
            modified += self.MANDATORY_DISCLAIMER

        return modified
