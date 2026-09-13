"""Production Support Strategy Planner using trained models and MultimodalContext."""
from typing import Optional, Dict, Any, List
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
from zenova.strategy.inference import StrategyInferenceEngine
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.core.logging import get_logger

logger = get_logger("zenova.strategy.planner")


class ESConvStrategyPlanner(BaseStrategyPlanner):
    """Production support strategy planner consuming enriched MultimodalContext and recommending calibrated emotional support strategies."""
    MODULE_NAME = "ESConvStrategyPlanner"
    VERSION = "strategy-v1.0.0"

    def __init__(
        self,
        model_dir: str = "models/strategy",
        use_transformer: bool = True,
        device: str = "cpu"
    ):
        self.model_dir = model_dir
        self.use_transformer = use_transformer
        self.device = device
        self.engine = StrategyInferenceEngine(
            model_dir=model_dir,
            use_transformer=use_transformer,
            device=device
        )

    def predict_strategy(
        self,
        user_input: UserInput,
        emotion: EmotionResult,
        symptoms: SymptomResult,
        context: Optional[ConversationContext] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> StrategyResult:
        """Recommend emotional support strategy from multimodal and conversational context."""
        text = user_input.text

        # Extract multimodal signals
        emotion_label = None
        situation = None
        problem_type = None
        dialog_stage = None

        if multimodal_context:
            if multimodal_context.emotion and multimodal_context.emotion.primary_emotion:
                emotion_label = multimodal_context.emotion.primary_emotion
            dialog_stage = multimodal_context.conversation.dialog_stage
        elif emotion and emotion.primary_emotion:
            emotion_label = emotion.primary_emotion.value

        if user_input.metadata:
            situation = user_input.metadata.get("situation")
            problem_type = user_input.metadata.get("problem_type")

        # Call inference engine
        pred = self.engine.predict_strategy(
            text=text,
            emotion=emotion_label,
            situation=situation,
            problem_type=problem_type,
            stage=dialog_stage
        )

        # Context-aware clinical nuance adjustments
        rationale = pred["rationale"]
        if multimodal_context and multimodal_context.emotion and multimodal_context.emotion.is_discrepancy_detected:
            rationale = f"[Acoustic/Verbal Discrepancy Active] {rationale}"

        return StrategyResult(
            is_placeholder=False,
            module_version=self.VERSION,
            selected_strategy=pred["selected_strategy"],
            confidence=pred["confidence"],
            stage=pred["stage"],
            rationale=rationale,
            alternatives=pred.get("alternatives", []),
            probabilities=pred.get("probabilities", {}),
            timestamp=datetime.now(timezone.utc)
        )
