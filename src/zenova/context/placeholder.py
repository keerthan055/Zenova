"""Placeholder Multimodal Context Engine for testing and fallback."""
from typing import Optional, List, Dict, Any

from zenova.core.interfaces import BaseMultimodalContextEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    ConversationTurn,
)
from zenova.schemas.context import (
    MultimodalContext,
    ConversationContextBlock,
    EmotionContextBlock,
    SymptomsContextBlock,
    RiskContextBlock,
    BaselineContextBlock,
    BehaviorContextBlock,
    VoiceContextBlock,
    HistoryContextBlock,
    MetadataContextBlock,
    DialogStage,
    PrivacyLevel
)
from zenova.context.engine import MultimodalContextEngine


class ContextPlaceholderEngine(BaseMultimodalContextEngine):
    """Fallback placeholder context engine."""

    MODULE_NAME = "ContextPlaceholderEngine"
    VERSION = "placeholder-v0.1.0"

    def __init__(self):
        self._delegate = MultimodalContextEngine()

    def build_context(
        self,
        user_input: UserInput,
        history: Optional[List[ConversationTurn]] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        baseline: Optional[BaselineResult] = None,
        behavior: Optional[BehavioralResult] = None,
        voice: Optional[VoiceResult] = None,
        previous_strategies: Optional[List[StrategyResult]] = None,
        previous_outcomes: Optional[List[Dict[str, Any]]] = None,
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dialog_stage: DialogStage = DialogStage.EXPLORATION,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
        is_crisis_bypass: bool = False
    ) -> MultimodalContext:
        ctx = self._delegate.build_context(
            user_input=user_input,
            history=history,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            baseline=baseline,
            behavior=behavior,
            voice=voice,
            previous_strategies=previous_strategies,
            previous_outcomes=previous_outcomes,
            user_feedback=user_feedback,
            metadata=metadata,
            dialog_stage=dialog_stage,
            privacy_level=privacy_level,
            is_crisis_bypass=is_crisis_bypass
        )
        ctx.metadata.engine_version = self.VERSION
        return ctx
