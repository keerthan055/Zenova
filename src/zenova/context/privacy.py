"""Privacy and Redaction Engine for ZENOVA Step 9.

Ensures privacy-aware handling and storage of multimodal context objects:
- PII detection and scrubbing (emails, phones, SSNs, credit cards, IPs)
- Configurable Privacy Levels: STANDARD, ANONYMIZED, EPHEMERAL
- Pseudonymization of user and session identifiers
- Safe serialization for database persistence
"""
import re
import hmac
import hashlib
from typing import Dict, Any, Tuple, Optional
from zenova.schemas.context import PrivacyLevel


# Regular expressions for identifying common PII tokens
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
IP_ADDRESS_REGEX = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')

DEFAULT_SALT = "zenova_privacy_salt_2026"


class PrivacyEngine:
    """Privacy enforcement, PII scrubbing, and anonymization engine."""

    def __init__(self, salt: str = DEFAULT_SALT):
        self.salt = salt

    def scrub_text(self, text: str) -> Tuple[str, int]:
        """Scrub PII tokens from free text.

        Returns:
            (sanitized_text, count_of_redactions)
        """
        if not text or not isinstance(text, str):
            return text, 0

        redactions = 0

        def _replace_email(match):
            nonlocal redactions
            redactions += 1
            return "[REDACTED_EMAIL]"

        def _replace_phone(match):
            nonlocal redactions
            redactions += 1
            return "[REDACTED_PHONE]"

        def _replace_ssn(match):
            nonlocal redactions
            redactions += 1
            return "[REDACTED_SSN]"

        def _replace_cc(match):
            nonlocal redactions
            redactions += 1
            return "[REDACTED_CARD]"

        def _replace_ip(match):
            nonlocal redactions
            redactions += 1
            return "[REDACTED_IP]"

        scrubbed = EMAIL_REGEX.sub(_replace_email, text)
        scrubbed = PHONE_REGEX.sub(_replace_phone, scrubbed)
        scrubbed = SSN_REGEX.sub(_replace_ssn, scrubbed)
        scrubbed = CREDIT_CARD_REGEX.sub(_replace_cc, scrubbed)
        scrubbed = IP_ADDRESS_REGEX.sub(_replace_ip, scrubbed)

        return scrubbed, redactions

    def pseudonymize_id(self, raw_id: str) -> str:
        """Deterministically pseudonymize user or session ID via HMAC-SHA256."""
        if not raw_id:
            return ""
        digest = hmac.new(
            self.salt.encode("utf-8"),
            raw_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"anon_{digest[:16]}"

    def apply_privacy(
        self,
        context_dict: Dict[str, Any],
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    ) -> Tuple[Dict[str, Any], int]:
        """Apply the specified privacy level transformations to a context dictionary.

        Returns:
            (transformed_context_dict, total_redactions)
        """
        import copy
        ctx = copy.deepcopy(context_dict)
        total_redactions = 0

        # 1. Scrub conversation text
        conv = ctx.get("conversation", {})
        if "current_text" in conv:
            scrubbed_text, count = self.scrub_text(conv["current_text"])
            conv["current_text"] = scrubbed_text
            total_redactions += count

        # 2. Scrub history text
        hist = ctx.get("history", {})
        for turn in hist.get("recent_turns", []):
            if "text" in turn:
                turn["text"], count = self.scrub_text(turn["text"])
                total_redactions += count
        for feedback in hist.get("user_feedback", []):
            if "feedback_text" in feedback and feedback["feedback_text"]:
                feedback["feedback_text"], count = self.scrub_text(feedback["feedback_text"])
                total_redactions += count

        # 3. If ANONYMIZED, pseudonymize IDs
        if privacy_level == PrivacyLevel.ANONYMIZED:
            if "user_id" in conv:
                conv["user_id"] = self.pseudonymize_id(conv["user_id"])
            if "session_id" in conv:
                conv["session_id"] = self.pseudonymize_id(conv["session_id"])

        # 4. Update metadata privacy block
        meta = ctx.get("metadata", {})
        meta["privacy"] = {
            "privacy_level": privacy_level.value,
            "is_redacted": total_redactions > 0 or privacy_level == PrivacyLevel.ANONYMIZED,
            "redacted_fields_count": total_redactions,
            "retention_policy": "zero_retention" if privacy_level == PrivacyLevel.EPHEMERAL else (
                "anonymized_research" if privacy_level == PrivacyLevel.ANONYMIZED else "standard_retention"
            )
        }

        return ctx, total_redactions
