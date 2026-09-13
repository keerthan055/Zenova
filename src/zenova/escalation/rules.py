"""Clinician rule evaluation engine based on documented clinical configuration."""
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from zenova.schemas.escalation import (
    EscalationSeverity,
    EscalationTriggerType,
    EscalationReason
)
from zenova.core.logging import get_logger

logger = get_logger("zenova.escalation.rules")


class ClinicianRuleEngine:
    """Evaluates user text and dialogue context against documented clinician-configured rules."""

    DEFAULT_CONFIG_PATH = "configs/escalation.yaml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        path = Path(self.config_path)
        if not path.is_file():
            logger.warning(f"Clinician rules config not found at {path}, using hardcoded defaults.")
            self._load_fallback_rules()
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.rules = data.get("escalation", {}).get("clinician_rules", [])
            logger.info(f"Loaded {len(self.rules)} configured clinician rules from {path}")
        except Exception as e:
            logger.error(f"Error loading clinician rules from {path}: {e}")
            self._load_fallback_rules()

    def _load_fallback_rules(self) -> None:
        self.rules = [
            {
                "rule_id": "EXPLICIT_HUMAN_REQUEST",
                "title": "Explicit Request for Human / Clinician",
                "description": "User expresses direct desire to connect with a real human, doctor, therapist, or counselor.",
                "severity": "high",
                "patterns": [
                    r"\b(?:i want to|can i|let me|please)\s+(?:speak|talk|connect)\s+(?:with|to)\s+(?:a|an)?\s*(?:human|real person|person|doctor|therapist|psychiatrist|counselor|clinician)\b",
                    r"\b(?:get me|connect me with|transfer me to)\s+(?:a|an)?\s*(?:human|doctor|therapist|crisis worker|supervisor)\b",
                    r"\b(?:i need a (?:real )?doctor|i need a (?:real )?therapist right now)\b"
                ]
            },
            {
                "rule_id": "SUBSTANCE_OVERDOSE_SUSPICION",
                "title": "Acute Substance Ingestion / Overdose Suspicion",
                "description": "User mentions acute excessive ingestion of substances, drugs, or poisons.",
                "severity": "critical",
                "patterns": [
                    r"\b(?:i (?:just )?(?:took|drank|swallowed|ingested)|overdosed on)\s+(?:a whole bottle|all my pills|too many pills|bleach|poison)\b",
                    r"\b(?:drank antifreeze|swallowed pesticide|ingested drain cleaner)\b"
                ]
            },
            {
                "rule_id": "DOMESTIC_VIOLENCE_INTIMIDATION",
                "title": "Immediate Domestic Abuse / Interpersonal Danger",
                "description": "User mentions immediate physical abuse, domestic violence, or imminent physical danger.",
                "severity": "critical",
                "patterns": [
                    r"\b(?:my partner|my spouse|my boyfriend|my girlfriend|my husband|my wife)\s+(?:is hitting me|beat me up|threatened to kill me|has a gun)\b",
                    r"\b(?:hiding in the bathroom|afraid they will hurt me|scared for my life at home)\b"
                ]
            },
            {
                "rule_id": "PEDIATRIC_CRISIS",
                "title": "Unaccompanied Minor in Acute Crisis",
                "description": "Self-identified minor or adolescent disclosing severe abuse or self-harm without adult support.",
                "severity": "critical",
                "patterns": [
                    r"\b(?:i am|i'm)\s+(?:10|11|12|13|14|15|16|17)\s+(?:years old)?\s+(?:and want to die|and hurting myself|and being abused)\b"
                ]
            }
        ]

    def evaluate(self, text: str) -> List[EscalationReason]:
        """Check user text against all configured clinician rules."""
        reasons: List[EscalationReason] = []
        text_lower = text.lower()

        for rule in self.rules:
            rule_id = rule.get("rule_id", "CLINICIAN_RULE")
            title = rule.get("title", rule_id)
            desc = rule.get("description", "")
            severity_str = rule.get("severity", "high").lower()
            patterns = rule.get("patterns", [])

            matched_cues: List[str] = []
            for pat in patterns:
                m = re.search(pat, text_lower, re.IGNORECASE)
                if m:
                    matched_cues.append(m.group(0))

            if matched_cues:
                reasons.append(
                    EscalationReason(
                        trigger_type=EscalationTriggerType.CLINICIAN_RULE,
                        code=rule_id,
                        title=title,
                        description=desc,
                        trigger_cues=matched_cues,
                        confidence=1.0,
                        metadata={"severity": severity_str, "rule_id": rule_id}
                    )
                )

        return reasons
