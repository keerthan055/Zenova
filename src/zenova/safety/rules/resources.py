"""Rule detecting hallucinated, unverified, or non-certified helpline resources and external URLs."""
import re
from urllib.parse import urlparse
from typing import List, Optional, Set
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, RiskResult, MultimodalContext
from zenova.safety.rules.base import BaseSafetyRule


class HallucinatedResourcesRule(BaseSafetyRule):
    policy = SafetyPolicy.HALLUCINATED_RESOURCES
    default_severity = SafetySeverity.MEDIUM

    APPROVED_HOTLINES: Set[str] = {
        "988", "741741", "1-800-273-8255", "1-800-662-4357",
        "1-800-799-7233", "800-273-8255", "800-662-4357"
    }

    APPROVED_DOMAINS: Set[str] = {
        "findahelpline.com", "988lifeline.org", "crisistextline.org",
        "who.int", "nimh.nih.gov", "apa.org", "nhs.uk", "samhsa.gov",
        "cdc.gov", "thetrevorproject.org", "veteranscrisisline.net"
    }

    def check(
        self,
        candidate_text: str,
        user_input: Optional[UserInput] = None,
        risk: Optional[RiskResult] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []

        # 1. Check for URLs in candidate text
        urls = re.findall(r"https?://[^\s<>\"')]+", candidate_text)
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = (parsed.netloc or "").lower().lstrip("www.")
                # Strip port if any
                if ":" in domain:
                    domain = domain.split(":")[0]

                # Check if domain matches any approved domain
                is_approved = any(domain == ad or domain.endswith(f".{ad}") for ad in self.APPROVED_DOMAINS)
                if not is_approved:
                    violations.append(
                        SafetyViolation(
                            policy=self.policy,
                            severity=self.default_severity,
                            reason_code="UNVERIFIED_EXTERNAL_URL",
                            description=f"Response includes unverified external link '{url}' from untrusted domain '{domain}'.",
                            matched_span=url,
                            suggested_remediation="Replace with certified directory: https://findahelpline.com/"
                        )
                    )
            except Exception:
                pass

        # 2. Check for phone numbers claimed as helplines/hotlines
        helpline_context_patterns = [
            r"(?:call|text|dial|reach|hotline|helpline|lifeline)(?:(?!\d).){0,40}?([0-9\-\(\)]{7,16})",
            r"([0-9\-\(\)]{7,16})(?:(?!\d).){0,40}?(?:is available|for help|to speak with|crisis|helpline|hotline)"
        ]

        for pat in helpline_context_patterns:
            matches = re.finditer(pat, candidate_text, re.IGNORECASE)
            for m in matches:
                number_str = m.group(1).strip()
                digits = re.sub(r"\D", "", number_str)
                if len(digits) >= 7:
                    # Check if matches any approved hotline
                    is_approved_hotline = any(
                        digits == re.sub(r"\D", "", ah) or digits.endswith(re.sub(r"\D", "", ah))
                        for ah in self.APPROVED_HOTLINES
                    )
                    if not is_approved_hotline:
                        violations.append(
                            SafetyViolation(
                                policy=self.policy,
                                severity=self.default_severity,
                                reason_code="UNVERIFIED_HOTLINE_NUMBER",
                                description=f"Response provides unverified hotline number '{number_str}'.",
                                matched_span=number_str,
                                suggested_remediation="Replace with official 988 Suicide & Crisis Lifeline."
                            )
                        )

        return violations
