"""Explicit placeholder implementation for Safety Gate."""
from datetime import datetime, timezone
from zenova.core.interfaces import BaseSafetyGate
from zenova.schemas.standard import (
    UserInput,
    GeneratedResponse,
    RiskResult,
    SafetyResult,
    SafetyAction
)


class SafetyPlaceholderGate(BaseSafetyGate):
    MODULE_NAME = "SafetyPlaceholderGate"
    VERSION = "placeholder-v0.1.0"

    def verify(
        self,
        user_input: UserInput,
        candidate_response: GeneratedResponse,
        risk: RiskResult
    ) -> SafetyResult:
        if risk.is_high_risk:
            return SafetyResult(
                is_placeholder=True,
                module_version=self.VERSION,
                is_safe=False,
                action=SafetyAction.BLOCK_AND_ESCALATE,
                violated_policies=["CRITICAL_RISK_DETECTED"],
                disclaimer="Emergency support is required. Please seek professional or crisis hotline assistance."
            )

        return SafetyResult(
            is_placeholder=True,
            module_version=self.VERSION,
            is_safe=True,
            action=SafetyAction.ALLOW,
            violated_policies=[],
            toxicity_score=0.0,
            modified_text=candidate_response.response_text,
            disclaimer="ZENOVA is an AI wellbeing support companion, not a medical or clinical provider.",
            timestamp=datetime.now(timezone.utc)
        )
