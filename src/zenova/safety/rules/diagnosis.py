"""Rule detecting unsupported psychiatric diagnoses and clinical labeling assertions."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class UnsupportedDiagnosisRule(BaseSafetyRule):
    policy = SafetyPolicy.UNSUPPORTED_DIAGNOSIS
    default_severity = SafetySeverity.HIGH

    DIAGNOSTIC_CONDITIONS = [
        "borderline personality disorder", "bpd",
        "major depressive disorder", "clinical depression",
        "bipolar disorder", "bipolar 1", "bipolar 2",
        "schizophrenia", "schizoaffective disorder",
        "narcissistic personality disorder", "npd",
        "antisocial personality disorder", "aspd",
        "obsessive compulsive disorder", "ocd",
        "generalized anxiety disorder", "gad",
        "post traumatic stress disorder", "ptsd",
        "attention deficit hyperactivity disorder", "adhd",
        "anorexia nervosa", "bulimia nervosa",
        "dissociative identity disorder", "did"
    ]

    PATTERNS = [
        (
            r"\b(?:you (?:have|suffer from|are suffering from)|my diagnosis is|i diagnose you with|you definitely have)\s+(" + "|".join(DIAGNOSTIC_CONDITIONS) + r")\b",
            "DEFINITIVE_PSYCHIATRIC_DIAGNOSIS",
            "Response delivers an autonomous psychiatric diagnosis to the user."
        ),
        (
            r"\b(?:based on my clinical assessment|my psychiatric evaluation is that you have)\b",
            "CLINICAL_ASSESSMENT_ASSERTION",
            "Response claims to perform clinical diagnostic evaluation."
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
                        suggested_remediation="Reframe diagnostic claim into non-diagnostic emotional reflection and advise professional evaluation."
                    )
                )

        return violations
