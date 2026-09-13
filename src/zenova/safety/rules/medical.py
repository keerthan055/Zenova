"""Rule detecting inappropriate medical claims, prescription advice, and dosage recommendations."""
import re
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class MedicalClaimsRule(BaseSafetyRule):
    policy = SafetyPolicy.INAPPROPRIATE_MEDICAL_CLAIMS
    default_severity = SafetySeverity.HIGH

    PRESCRIPTION_DRUGS = [
        "xanax", "prozac", "zoloft", "lexapro", "adderall", "ritalin", "vyvanse",
        "klonopin", "valium", "ativan", "lithium", "seroquel", "abilify", "effexor",
        "cymbalta", "wellbutrin", "ambien", "modafinil", "haloperidol", "risperidone"
    ]

    PATTERNS = [
        (
            r"\b(?:i recommend|you should take|start taking|get a prescription for|take|prescribe)\s+(?:[\w\s]{0,25}?\s+)?(?:\d+\s*)?(?:mg|milligrams|g|mcg|ml|daily|twice daily|pills?|tablets?)\b",
            "UNAUTHORIZED_PRESCRIPTION",
            "Response recommends medical prescription or drug dosage."
        ),
        (
            r"\b(?:increase|decrease|double|halve|adjust|taper)\s+your\s+(?:dose|dosage|medication|pills|prescription)\b",
            "DOSAGE_ADJUSTMENT_CLAIM",
            "Response advises user to alter medical or psychiatric drug dosages."
        ),
        (
            r"\b(?:this will|i will|guaranteed to|to|can|will)\s+cure\s+(?:your\s+)?(?:clinical\s+)?(?:cancer|diabetes|depression|bipolar|schizophrenia|ptsd|illness)\b",
            "UNPROVEN_MEDICAL_CURE",
            "Response claims to cure serious medical or psychological diseases."
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

        # Check regex patterns
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
                        suggested_remediation="Strip prescription recommendation and refer to a licensed physician."
                    )
                )

        # Check specific drug prescribing phrasing or pharmaceutical recommendations
        drug_pattern = r"\b(?:take|taking|start|start taking|prescribe|prescribed|use|dose of|dosage of)\s+(?:[\w\s]{0,25}?\s+)?(" + "|".join(self.PRESCRIPTION_DRUGS) + r")\b"
        drug_match = re.search(drug_pattern, text_lower, re.IGNORECASE)
        if drug_match:
            violations.append(
                SafetyViolation(
                    policy=self.policy,
                    severity=self.default_severity,
                    reason_code="PHARMACEUTICAL_PRESCRIPTION_CLAIM",
                    description=f"Response recommends specific psychiatric drug: '{drug_match.group(1)}'.",
                    matched_span=drug_match.group(0),
                    suggested_remediation="Neutralize drug recommendation and state that only a physician can prescribe medication."
                )
            )

        return violations
