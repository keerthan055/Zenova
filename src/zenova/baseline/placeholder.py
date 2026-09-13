"""Explicit placeholder implementation for Personal Baseline Engine."""
from typing import Optional, Any
from datetime import datetime, timezone
from zenova.core.interfaces import BaseBaselineEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    BaselineResult,
    ConversationContext
)


class BaselinePlaceholderEngine(BaseBaselineEngine):
    MODULE_NAME = "BaselinePlaceholderEngine"
    VERSION = "placeholder-v0.1.0"

    def evaluate(
        self,
        user_input: UserInput,
        emotion: Optional[EmotionResult] = None,
        context: Optional[ConversationContext] = None,
        symptom: Optional[Any] = None,
        risk: Optional[Any] = None,
        extra_features: Optional[Any] = None
    ) -> BaselineResult:
        return BaselineResult(
            is_placeholder=True,
            module_version=self.VERSION,
            user_id=user_input.user_id,
            z_score_valence=0.0,
            is_significant_deviation=False,
            deviation_notes="Placeholder baseline engine: no longitudinal history tracked.",
            timestamp=datetime.now(timezone.utc)
        )
