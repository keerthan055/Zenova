"""Voice and acoustic emotion analysis subpackage for ZENOVA."""

from zenova.schemas.voice import (
    AudioFormat,
    AudioPayload,
    AcousticFeatures,
    VoiceEmotionResult,
    VoiceResult,
)
from zenova.voice.validator import AudioValidator, AudioValidationError
from zenova.voice.extractor import AcousticFeatureExtractor
from zenova.voice.asr import (
    BaseSpeechToTextEngine,
    AcousticHeuristicASR,
    PluggableWhisperASR,
)
from zenova.voice.model import VoiceEmotionClassifier, RAVDESS_EMOTIONS, RAVDESS_TO_ZENOVA
from zenova.voice.baseline import VoiceAcousticBaseline
from zenova.voice.placeholder import VoicePlaceholderAnalyzer
from zenova.voice.analyzer import VoiceEmotionAnalyzer
from zenova.voice.fusion import MultimodalFusionEngine, MultimodalAffectiveResult

__all__ = [
    "AudioFormat",
    "AudioPayload",
    "AcousticFeatures",
    "VoiceEmotionResult",
    "VoiceResult",
    "AudioValidator",
    "AudioValidationError",
    "AcousticFeatureExtractor",
    "BaseSpeechToTextEngine",
    "AcousticHeuristicASR",
    "PluggableWhisperASR",
    "VoiceEmotionClassifier",
    "RAVDESS_EMOTIONS",
    "RAVDESS_TO_ZENOVA",
    "VoiceAcousticBaseline",
    "VoicePlaceholderAnalyzer",
    "VoiceEmotionAnalyzer",
    "MultimodalFusionEngine",
    "MultimodalAffectiveResult",
]
