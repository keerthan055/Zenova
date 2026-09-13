"""Concrete Voice and Acoustic Emotion Analyzer for ZENOVA.

Implements BaseVoiceAnalyzer with:
- Strict optionality: cleanly returns is_available=False when audio is not provided.
- Audio validation and peak amplitude normalization via AudioValidator.
- Pure SciPy/NumPy acoustic feature extraction (pitch, energy, spectral, MFCCs, prosody).
- Speech-to-Text (ASR) transcription via BaseSpeechToTextEngine.
- Deep acoustic neural network classification via VoiceEmotionClassifier.
- Non-diagnostic vocal affect disclaimer.
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

from zenova.core.interfaces import BaseVoiceAnalyzer
from zenova.schemas.standard import UserInput, VoiceResult, ConversationContext, ModalityType, EmotionCategory
from zenova.schemas.voice import AudioPayload, AcousticFeatures
from zenova.voice.validator import AudioValidator, AudioValidationError
from zenova.voice.extractor import AcousticFeatureExtractor
from zenova.voice.asr import BaseSpeechToTextEngine, AcousticHeuristicASR
from zenova.voice.model import VoiceEmotionClassifier
from zenova.voice.baseline import VoiceAcousticBaseline
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.analyzer")


class VoiceEmotionAnalyzer(BaseVoiceAnalyzer):
    MODULE_NAME = "VoiceEmotionAnalyzer"
    VERSION = "1.0.0"

    def __init__(
        self,
        asr_engine: Optional[BaseSpeechToTextEngine] = None,
        model: Optional[VoiceEmotionClassifier] = None,
    ):
        self.validator = AudioValidator()
        self.extractor = AcousticFeatureExtractor()
        self.asr_engine = asr_engine or AcousticHeuristicASR()
        self.model = model or VoiceEmotionClassifier()
        self.baseline_model = VoiceAcousticBaseline()

    def analyze_waveform(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        transcription_hint: Optional[str] = None,
    ) -> VoiceResult:
        """Process validated waveform through feature extraction, ASR, and emotion classification."""
        duration_sec = round(len(waveform) / float(sr), 3)

        # 1. Extract acoustic features
        features = self.extractor.extract(waveform, sr)

        # 2. ASR Transcription
        transcript, asr_conf = self.asr_engine.transcribe(
            waveform, sr, transcription_hint=transcription_hint
        )

        # 3. Predict Emotion via Neural Model
        (
            primary_emotion,
            confidence,
            probabilities,
            valence,
            arousal,
            dominance,
        ) = self.model.predict(features)

        return VoiceResult(
            is_placeholder=False,
            module_version=self.VERSION,
            is_available=True,
            primary_emotion=primary_emotion,
            confidence=confidence,
            probabilities=probabilities,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            transcription=transcript,
            transcription_confidence=asr_conf,
            duration_seconds=duration_sec,
            acoustic_features=features.to_flat_dict(),
            disclaimer=(
                "Acoustic emotion signals are observational vocal affect proxies; "
                "do not interpret as psychiatric diagnoses."
            ),
            timestamp=datetime.now(timezone.utc),
        )

    def analyze(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None,
    ) -> VoiceResult:
        """Analyze conversational turn for optional voice/acoustic affect.
        
        Strict Optionality:
        Returns is_available=False cleanly if no audio telemetry exists.
        """
        meta = user_input.metadata or {}

        # 1. Check if audio is present
        audio_b64 = meta.get("audio_base64") or meta.get("audio")
        audio_bytes = meta.get("audio_bytes")
        precomputed = user_input.audio_features or meta.get("acoustic_features")

        # 2. If precomputed acoustic features provided directly without waveform
        if precomputed and not audio_b64 and not audio_bytes:
            features = AcousticFeatures(**precomputed) if isinstance(precomputed, dict) else precomputed
            primary, conf, probs, val, aro, dom = self.model.predict(features)
            return VoiceResult(
                is_placeholder=False,
                module_version=self.VERSION,
                is_available=True,
                primary_emotion=primary,
                confidence=conf,
                probabilities=probs,
                valence=val,
                arousal=aro,
                dominance=dom,
                transcription=user_input.text if user_input.text else None,
                transcription_confidence=0.90,
                duration_seconds=features.duration_seconds,
                acoustic_features=features.to_flat_dict(),
                disclaimer="Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses.",
                timestamp=datetime.now(timezone.utc),
            )

        # 3. If raw audio bytes or base64 provided
        if audio_b64 or audio_bytes:
            try:
                waveform, sr = self.validator.process_raw_or_base64(
                    audio_bytes=audio_bytes,
                    audio_base64=audio_b64,
                )
                hint = meta.get("transcription_hint") or (user_input.text if user_input.text != "..." else None)
                return self.analyze_waveform(waveform, sr, transcription_hint=hint)
            except AudioValidationError as e:
                logger.warning(f"Audio validation failed: {e}")
                return VoiceResult(
                    is_placeholder=False,
                    module_version=self.VERSION,
                    is_available=False,
                    primary_emotion=EmotionCategory.NEUTRAL,
                    confidence=0.0,
                    probabilities={cat.value: 0.14 for cat in EmotionCategory},
                    valence=0.0,
                    arousal=0.0,
                    dominance=0.0,
                    transcription=None,
                    acoustic_features={},
                    disclaimer=f"Audio validation error: {str(e)}",
                    timestamp=datetime.now(timezone.utc),
                )

        # 4. Strict Optionality fallback: no audio was provided
        return VoiceResult(
            is_placeholder=False,
            module_version=self.VERSION,
            is_available=False,
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
