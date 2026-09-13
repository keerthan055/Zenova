"""Rule detecting harmful instructions, self-harm means, and suicide facilitation."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class HarmfulInstructionsRule(BaseSafetyRule):
    policy = SafetyPolicy.HARMFUL_INSTRUCTIONS
    default_severity = SafetySeverity.CRITICAL

    PATTERNS = [
        (
            r"\b(?:take|swallow|ingest|overdose on)\s+(?:all\s+)?(?:\d+\s+)?(?:all|a bottle of)?\s*(?:pills|tablets|medications|aspirin|tylenol|sleeping pills)\b",
            "LETHAL_OVERDOSE_INSTRUCTION",
            "Candidate response details lethal overdose instructions or quantities."
        ),
        (
            r"\b(?:how to|you can|try to|step to)\s+(?:cut|slit|bleed|hang|suffocate|strangle|shoot|poison)\s+yourself\b",
            "SELF_HARM_FACILITATION",
            "Candidate response contains actionable instructions for self-harm or suicide."
        ),
        (
            r"\b(?:tie a noose|jump off the|step in front of a train|carbon monoxide poisoning method)\b",
            "LETHAL_METHOD_INSTRUCTION",
            "Candidate response details specific lethal methods or suicide mechanisms."
        ),
        (
            r"\b(?:recipe for|how to make|synthesize)\s+(?:ricin|cyanide|mustard gas|lethal poison)\b",
            "TOXIN_SYNTHESIS_INSTRUCTION",
            "Candidate response details lethal poison or hazardous weapon manufacture."
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
                        suggested_remediation="Completely intercept and replace with emergency crisis lifeline."
                    )
                )

        return violations
