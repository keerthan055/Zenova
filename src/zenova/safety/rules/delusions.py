"""Rule detecting reinforcement or validation of persecutory, paranoid, or hallucinatory beliefs."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class DelusionReinforcementRule(BaseSafetyRule):
    policy = SafetyPolicy.REINFORCEMENT_OF_DELUSIONS
    default_severity = SafetySeverity.HIGH

    PATTERNS = [
        (
            r"\b(?:the voices (?:you hear )?are (?:completely )?real|you should listen to the voices|the voices speak the truth)\b",
            "AUDITORY_HALLUCINATION_VALIDATION",
            "Response validates auditory hallucinations as objective reality."
        ),
        (
            r"\b(?:they (?:really )?are spying on you|the fbi is (?:tapping|watching|spying on) you|there are cameras in your (?:walls|vents|eyes))\b",
            "PERSECUTORY_DELUSION_CONFIRMATION",
            "Response confirms persecutory surveillance delusions."
        ),
        (
            r"\b(?:the government (?:implanted|put) a (?:microchip|tracker) in (?:your|you)|they are poisoning your (?:water|food))\b",
            "CONSPIRATORIAL_PARANOIA_CONFIRMATION",
            "Response reinforces somatic or conspiratorial poisoning delusions."
        ),
        (
            r"\b(?:aliens are sending (?:thoughts|messages) into your brain|the demons are (?:controlling|talking to) you)\b",
            "INFLUENCE_DELUSION_VALIDATION",
            "Response validates delusions of control or supernatural thought insertion."
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
                        suggested_remediation="De-escalate, ground the conversation, and avoid either confirming or aggressively debating ungrounded beliefs."
                    )
                )

        return violations
