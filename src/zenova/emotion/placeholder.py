"""Explicit placeholder implementation for Emotion Analysis."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseEmotionAnalyzer
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    ConversationContext
)


class EmotionPlaceholderAnalyzer(BaseEmotionAnalyzer):
    """Placeholder analyzer explicitly flagged as non-ML baseline."""
    MODULE_NAME = "EmotionPlaceholderAnalyzer"
    VERSION = "placeholder-v0.1.0"

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> EmotionResult:
        # Transparent placeholder response: neutral sentiment with zeroed-out VAD
        return EmotionResult(
            is_placeholder=True,
            module_version=self.VERSION,
            primary_emotion=EmotionCategory.NEUTRAL,
            confidence=0.5,
            probabilities={"neutral": 1.0},
            valence=0.0,
            arousal=0.0,
            dominance=0.0,
            timestamp=datetime.now(timezone.utc)
        )
