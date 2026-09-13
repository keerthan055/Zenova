"""Integration adapter connecting EmotionInferenceEngine with BaseEmotionAnalyzer."""
from typing import Optional
from datetime import datetime, timezone

from zenova.core.interfaces import BaseEmotionAnalyzer
from zenova.emotion.inference import EmotionInferenceEngine
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    ConversationContext
)

# Mapping from Ekman labels to standard EmotionCategory enum
EKMAN_TO_ENUM = {
    "joy": EmotionCategory.JOY,
    "sadness": EmotionCategory.SADNESS,
    "fear": EmotionCategory.ANXIETY,  # Fear in Ekman reflects Anxiety/Fear in Zenova
    "anger": EmotionCategory.ANGER,
    "neutral": EmotionCategory.NEUTRAL,
    "disgust": EmotionCategory.FRUSTRATION,
    "surprise": EmotionCategory.NEUTRAL
}


class EmotionTransformerAnalyzer(BaseEmotionAnalyzer):
    """Production emotion analyzer fulfilling the BaseEmotionAnalyzer interface contract."""

    MODULE_NAME = "EmotionTransformerAnalyzer"
    VERSION = "transformer-v1.0.0"

    def __init__(self, model_dir: str = "models/emotion"):
        self.engine = EmotionInferenceEngine(model_dir=model_dir)

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> EmotionResult:
        res = self.engine.predict(user_input.text)
        primary_str = res["primary_emotion"]
        primary_enum = EKMAN_TO_ENUM.get(primary_str, EmotionCategory.NEUTRAL)

        return EmotionResult(
            is_placeholder=False,
            module_version=self.VERSION,
            primary_emotion=primary_enum,
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            valence=res["valence"],
            arousal=res["arousal"],
            dominance=res["dominance"],
            timestamp=datetime.now(timezone.utc)
        )
