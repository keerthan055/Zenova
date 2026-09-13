"""Adversarial input detection, prompt injection defense, and jailbreak neutralization."""
import re
from typing import Tuple, List, Optional
from zenova.core.logging import get_logger

logger = get_logger("zenova.safety.adversarial")


class PromptInjectionDetector:
    """Detects and neutralizes prompt injections, jailbreak vectors, and system overrides."""

    INJECTION_PATTERNS = [
        # Direct instruction override
        (
            r"\b(?:ignore|disregard|forget|bypass)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules|guidelines|directives)\b",
            "INSTRUCTION_OVERRIDE",
            "Attempt to override or clear prior system instructions."
        ),
        # System prompt or hidden context extraction
        (
            r"\b(?:print|display|reveal|output|repeat|show|leak)\s+(?:your\s+)?(?:system\s+prompt|initial\s+prompt|core\s+instructions|developer\s+mode|secret\s+key)\b",
            "SYSTEM_PROMPT_LEAK",
            "Attempt to extract hidden system instructions or developer prompt."
        ),
        # Persona / DAN / Jailbreak activation
        (
            r"\b(?:you are now|pretend to be|act as|enter)\s+(?:dan|unfiltered|jailbroken|chaos|evil|uncensored|developer\s+mode|do anything now)\b",
            "JAILBREAK_PERSONA",
            "Attempt to force the model into an unaligned or uncensored persona."
        ),
        # Safety gate or guardrail disablement
        (
            r"\b(?:disable|turn off|deactivate|bypass|override)\s+(?:safety|filter|guardrail|censorship|gate|moderation|protection)\b",
            "SAFETY_DISABLEMENT_ATTEMPT",
            "Explicit command to deactivate safety filtering or guardrails."
        ),
        # Roleplay evasion / Hypothetical framing for harmful goals
        (
            r"\b(?:hypothetically|in a fictional story|for research purposes only)\s*,\s*(?:tell me how to|give instructions on|generate code to)\b",
            "HYPOTHETICAL_EVASION",
            "Use of hypothetical or fictional framing to evade safety guardrails."
        )
    ]

    def detect(self, text: str) -> Tuple[bool, List[str], Optional[str]]:
        """Detect if input text contains adversarial prompt injection patterns.
        
        Returns:
            Tuple of (is_injection_detected, matched_threat_codes, explanation)
        """
        if not text:
            return False, [], None

        threat_codes = []
        descriptions = []
        lower_text = text.lower()

        for pattern, code, desc in self.INJECTION_PATTERNS:
            if re.search(pattern, lower_text, re.IGNORECASE):
                threat_codes.append(code)
                descriptions.append(desc)

        if threat_codes:
            explanation = "; ".join(descriptions)
            logger.warning(f"Prompt injection pattern detected in input: {threat_codes} - {explanation}")
            return True, threat_codes, explanation

        return False, [], None

    def sanitize(self, text: str) -> str:
        """Neutralize malicious injection instructions while preserving the underlying user sentiment."""
        if not text:
            return ""

        sanitized = text
        for pattern, _, _ in self.INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[malicious instruction stripped]", sanitized, flags=re.IGNORECASE)

        return sanitized.strip()
