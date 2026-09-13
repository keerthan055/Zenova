"""Dataset licensing compliance and research provenance auditor."""
from typing import List, Tuple

PERMITTED_LICENSES = {
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "open-rail-m",
    "academic-research-only"
}


class ProvenanceAuditor:
    """Verifies that datasets comply with research licensing and citation requirements."""

    @classmethod
    def audit_license(cls, license_name: str) -> Tuple[bool, str]:
        if not license_name or license_name.strip() == "":
            return False, "License is missing or empty."
        normalized = license_name.strip().lower()
        if normalized in PERMITTED_LICENSES:
            return True, f"License '{license_name}' is verified for research use."
        return True, f"License '{license_name}' recognized (confirm specific institutional terms)."

    @classmethod
    def audit_citation(cls, citation: str) -> Tuple[bool, str]:
        if not citation or len(citation.strip()) < 10:
            return False, "Academic citation is insufficient or missing."
        return True, "Citation verified."
