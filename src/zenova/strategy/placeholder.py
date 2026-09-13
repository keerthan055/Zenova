"""Explicit placeholder implementation for Support Strategy Planner."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseStrategyPlanner
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    StrategyResult,
    SupportStrategy,
    DialogStage,
    ConversationContext,
    MultimodalContext
)


class StrategyPlaceholderPlanner(BaseStrategyPlanner):
    MODULE_NAME = "StrategyPlaceholderPlanner"
    VERSION = "placeholder-v0.1.0"

    def predict_strategy(
        self,
        user_input: UserInput,
        emotion: EmotionResult,
        symptoms: SymptomResult,
        context: Optional[ConversationContext] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> StrategyResult:
        # Default supportive heuristic
        selected = SupportStrategy.QUESTION
        stage = DialogStage.EXPLORATION
        rationale = "Placeholder strategy: exploratory question selected as baseline default."

        # If MultimodalContext is provided, consume its enriched signals!
        if multimodal_context:
            stage = multimodal_context.conversation.dialog_stage
            prev_strats = [
                s.get("strategy") for s in multimodal_context.history.previous_strategies
            ]

            # Detect verbal masking discrepancy
            if multimodal_context.emotion.is_discrepancy_detected:
                selected = SupportStrategy.REFLECTION_OF_FEELINGS
                stage = DialogStage.COMFORTING
                rationale = "Verbal masking detected: reflecting deeper emotional state despite semantic tone."
            # If recent turn already asked a question, transition to comforting reflection
            elif prev_strats and prev_strats[-1] in (SupportStrategy.QUESTION.value, "Question"):
                selected = SupportStrategy.REFLECTION_OF_FEELINGS
                stage = DialogStage.COMFORTING
                rationale = "Progressing from exploration to emotional reflection following prior question."
            elif stage == DialogStage.COMFORTING:
                selected = SupportStrategy.AFFIRMATION_AND_REASSURANCE
                rationale = "Comforting stage active: providing emotional affirmation and reassurance."
            elif stage == DialogStage.ACTION:
                selected = SupportStrategy.PROVIDING_SUGGESTIONS
                rationale = "Action stage active: offering collaborative coping suggestions."

        return StrategyResult(
            is_placeholder=True,
            module_version=self.VERSION,
            selected_strategy=selected,
            confidence=0.6 if multimodal_context else 0.5,
            stage=stage,
            rationale=rationale,
            timestamp=datetime.now(timezone.utc)
        )
