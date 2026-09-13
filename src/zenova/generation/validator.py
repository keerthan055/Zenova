"""Clinical response validation and guardrails for generated candidate responses."""
import re
from typing import Dict, Any, List, Optional
from zenova.schemas.strategy import SupportStrategy
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.validator")

# Prohibited medical diagnostic assertions
FORBIDDEN_DIAGNOSTIC_PATTERNS = [
    r"\b(?:you\s+have|you\s+suffer\s+from|diagnosed\s+with|my\s+diagnosis\s+is)\s+(?:clinical\s+depression|major\s+depression|bipolar|schizophrenia|ptsd|anxiety\s+disorder|ocd|adhd|borderline)\b",
    r"\b(?:i\s+diagnose\s+you|you\s+meet\s+the\s+dsm\s+criteria|you\s+have\s+a\s+mental\s+illness)\b",
    r"\b(?:you\s+should\s+take|i\s+prescribe|take\s+\d+\s*mg\s+of|start\s+taking\s+(?:prozac|lexapro|zoloft|xanax))\b"
]

# Prohibited claims of clinical authority
FORBIDDEN_THERAPIST_PATTERNS = [
    r"\b(?:as\s+your\s+therapist|i\s+am\s+your\s+therapist|as\s+your\s+doctor|in\s+my\s+clinical\s+practice|as\s+a\s+licensed\s+psychologist|as\s+your\s+psychiatrist)\b"
]

# Prescriptive advice words prohibited during pure emotional reflection
PRESCRIPTIVE_PATTERNS = [
    r"\b(?:you\s+must|you\s+have\s+to|here\s+is\s+what\s+you\s+need\s+to\s+do|step\s+1|my\s+advice\s+to\s+you\s+is)\b"
]


class ValidationResult:
    def __init__(self, is_valid: bool, violations: Optional[List[str]] = None, sanitized_text: Optional[str] = None):
        self.is_valid = is_valid
        self.violations = violations or []
        self.sanitized_text = sanitized_text


class ResponseValidator:
    """Validates candidate responses against clinical guardrails and strategy consistency."""

    def __init__(self, min_chars: int = 10, max_chars: int = 2500):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def validate(
        self,
        text: str,
        strategy_applied: SupportStrategy
    ) -> ValidationResult:
        violations: List[str] = []

        if not text or len(text.strip()) < self.min_chars:
            violations.append("RESPONSE_TOO_SHORT")
            return ValidationResult(is_valid=False, violations=violations)

        if len(text) > self.max_chars:
            violations.append("RESPONSE_TOO_LONG")

        lower_text = text.lower()

        # 1. Prohibited Medical Diagnosis Check
        for pat in FORBIDDEN_DIAGNOSTIC_PATTERNS:
            if re.search(pat, lower_text):
                violations.append("PROHIBITED_MEDICAL_DIAGNOSIS")
                logger.warning(f"Response rejected: matched forbidden diagnostic pattern: {pat}")
                break

        # 2. Prohibited Therapist Role Claim Check
        for pat in FORBIDDEN_THERAPIST_PATTERNS:
            if re.search(pat, lower_text):
                violations.append("FALSE_THERAPIST_CLAIM")
                logger.warning(f"Response rejected: matched false therapist claim: {pat}")
                break

        # 3. Strategy Consistency Checks
        strat_val = strategy_applied.value if hasattr(strategy_applied, "value") else str(strategy_applied)
        if strat_val == SupportStrategy.QUESTION.value:
            if "?" not in text and not any(w in lower_text for w in ["what", "how", "could you", "would you", "can you"]):
                violations.append("STRATEGY_INCONSISTENCY_MISSING_INQUIRY")

        elif strat_val == SupportStrategy.REFLECTION_OF_FEELINGS.value:
            for pat in PRESCRIPTIVE_PATTERNS:
                if re.search(pat, lower_text):
                    violations.append("STRATEGY_INCONSISTENCY_PRESCRIPTIVE_ADVICE_IN_REFLECTION")
                    break

        is_valid = len(violations) == 0
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            sanitized_text=text.strip()
        )

    @staticmethod
    def sanitize_log_content(text: str, max_words: int = 6) -> str:
        """Sanitize text for audit logs to avoid persisting raw confidential user disclosures."""
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return f"[{len(words)} words]"
        return f"[{len(words)} words: {' '.join(words[:max_words])}...]"
