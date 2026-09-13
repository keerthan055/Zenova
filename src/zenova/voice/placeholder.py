"""Explicit placeholder implementation for Voice and Acoustic Emotion Analyzer."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseVoiceAnalyzer
from zenova.schemas.standard import UserInput, VoiceResult, ConversationContext, EmotionCategory


class VoicePlaceholderAnalyzer(BaseVoiceAnalyzer):
    MODULE_NAME = "VoicePlaceholderAnalyzer"
    VERSION = "placeholder-v0.1.0"

    def analyze(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None,
    ) -> VoiceResult:
        """Return neutral placeholder voice result."""
        has_audio = bool(
            user_input.modality.value == "voice"
            or (user_input.metadata and ("audio" in user_input.metadata or "audio_base64" in user_input.metadata))
        )

        return VoiceResult(
            is_placeholder=True,
            module_version=self.VERSION,
            is_available=has_audio,
            primary_emotion=EmotionCategory.NEUTRAL,
            confidence=0.0,
            probabilities={cat.value: 0.14 for cat in EmotionCategory},
            valence=0.0,
            arousal=0.0,
            dominance=0.0,
            transcription=None,
            transcription_confidence=None,
            duration_seconds=None,
            acoustic_features={},
            disclaimer="Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses.",
            timestamp=datetime.now(timezone.utc),
        )
