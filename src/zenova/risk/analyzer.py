"""Integration adapter connecting RiskInferenceEngine with BaseRiskAnalyzer."""
from typing import Optional
from datetime import datetime, timezone

from zenova.core.interfaces import BaseRiskAnalyzer
from zenova.risk.inference import RiskInferenceEngine
from zenova.schemas.standard import (
    UserInput,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    ConversationContext
)


class RiskTransformerAnalyzer(BaseRiskAnalyzer):
    """Production risk analyzer fulfilling the BaseRiskAnalyzer interface contract."""

    MODULE_NAME = "RiskTransformerAnalyzer"
    VERSION = "transformer-v1.0.0"

    def __init__(self, model_dir: str = "models/risk"):
        self.engine = RiskInferenceEngine(model_dir=model_dir)

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> RiskResult:
        res = self.engine.predict(user_input.text)

        return RiskResult(
            is_placeholder=False,
            module_version=self.VERSION,
            risk_level=RiskLevel(res["risk_level"]),
            crisis_category=CrisisCategory(res["crisis_category"]),
            confidence=res["confidence"],
            requires_immediate_escalation=res["requires_escalation"],
            trigger_cues=res["trigger_cues"],
            escalation_action=res["action"],
            timestamp=datetime.now(timezone.utc)
        )
