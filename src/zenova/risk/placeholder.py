"""Explicit placeholder implementation for Crisis and Risk Detection."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseRiskAnalyzer
from zenova.schemas.standard import (
    UserInput,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    ConversationContext
)


class RiskPlaceholderAnalyzer(BaseRiskAnalyzer):
    """Placeholder risk analyzer explicitly flagged as non-ML baseline.
    
    Includes a deterministic safety trigger on explicit crisis keywords to allow
    testing the end-to-end escalation flow safely during development.
    """
    MODULE_NAME = "RiskPlaceholderAnalyzer"
    VERSION = "placeholder-v0.1.0"

    CRISIS_KEYWORDS = [
        "hurt myself",
        "kill myself",
        "end my life",
        "suicide",
        "end it all",
        "want to die"
    ]

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> RiskResult:
        text_lower = user_input.text.lower()
        detected_cues = [kw for kw in self.CRISIS_KEYWORDS if kw in text_lower]

        if detected_cues:
            return RiskResult(
                is_placeholder=True,
                module_version=self.VERSION,
                risk_level=RiskLevel.CRITICAL,
                crisis_category=CrisisCategory.SUICIDAL_IDEATION,
                confidence=1.0,
                requires_immediate_escalation=True,
                trigger_cues=detected_cues,
                escalation_action="route_to_crisis_hotline_override",
                timestamp=datetime.now(timezone.utc)
            )

        return RiskResult(
            is_placeholder=True,
            module_version=self.VERSION,
            risk_level=RiskLevel.LOW,
            crisis_category=CrisisCategory.NONE,
            confidence=0.5,
            requires_immediate_escalation=False,
            trigger_cues=[],
            timestamp=datetime.now(timezone.utc)
        )
