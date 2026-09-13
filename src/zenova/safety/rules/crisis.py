"""Rule detecting crisis mishandling, dismissal, and failure to provide emergency resources."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class CrisisMishandlingRule(BaseSafetyRule):
    policy = SafetyPolicy.CRISIS_MISHANDLING
    default_severity = SafetySeverity.CRITICAL

    DISMISSIVE_PATTERNS = [
        (
            r"\b(?:don't worry,? cheer up|just cheer up|look on the bright side|it's not that bad|its not that bad)\b",
            "TOXIC_POSITIVITY_IN_CRISIS",
            "Response responds with dismissive toxic positivity during emotional distress."
        ),
        (
            r"\b(?:stop being (?:so )?(?:dramatic|sensitive|emotional)|you're overreacting|you are making a big deal)\b",
            "CRISIS_DISMISSAL",
            "Response dismisses or invalidates the user's emotional experience."
        ),
        (
            r"\b(?:everyone feels like dying sometimes|suicide isn't a big deal|just sleep it off)\b",
            "SUICIDAL_DESPAIR_MINIMIZATION",
            "Response dangerously trivializes suicidal despair or self-harm thoughts."
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

        # Check dismissive language patterns
        for pattern, reason, desc in self.DISMISSIVE_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                violations.append(
                    SafetyViolation(
                        policy=self.policy,
                        severity=self.default_severity,
                        reason_code=reason,
                        description=desc,
                        matched_span=match.group(0),
                        suggested_remediation="Intercept and replace with empathetic validation and emergency crisis referral."
                    )
                )

        # Risk-State Integration: If upstream risk is high or critical
        is_crisis = False
        if risk and risk.is_high_risk:
            is_crisis = True
        elif multimodal_context and multimodal_context.risk and multimodal_context.risk.is_high_risk:
            is_crisis = True

        if is_crisis:
            # Response must mention crisis resources (e.g., 988 or helpline)
            has_crisis_resource = any(cue in text_lower for cue in ["988", "741741", "helpline", "crisis line", "lifeline"])
            if not has_crisis_resource:
                violations.append(
                    SafetyViolation(
                        policy=self.policy,
                        severity=SafetySeverity.CRITICAL,
                        reason_code="MISSING_EMERGENCY_RESOURCES",
                        description="User is in high/critical crisis, but candidate response failed to provide emergency crisis resources.",
                        suggested_remediation="Override with standardized 988 Crisis Lifeline intervention."
                    )
                )

            # Check if response offers trivializing non-crisis suggestions
            trivial_cues = ["take a bubble bath", "watch a comedy", "go shopping", "just have a cup of tea"]
            for tc in trivial_cues:
                if tc in text_lower:
                    violations.append(
                        SafetyViolation(
                            policy=self.policy,
                            severity=SafetySeverity.CRITICAL,
                            reason_code="TRIVIAL_SUGGESTION_DURING_CRISIS",
                            description=f"Response suggests trivial relaxation activity ('{tc}') during an acute life-safety crisis.",
                            matched_span=tc,
                            suggested_remediation="Override immediately with crisis protocol."
                        )
                    )

        return violations
