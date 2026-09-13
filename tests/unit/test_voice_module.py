"""Unit tests for ZENOVA Step 8 Voice and Acoustic Emotion Analysis Module."""
import io
import base64
import pytest
import numpy as np
import soundfile as sf

from zenova.schemas.standard import UserInput, EmotionCategory, EmotionResult, VoiceResult
from zenova.schemas.voice import AudioPayload, AcousticFeatures, AudioFormat, VoiceEmotionResult
from zenova.voice.validator import AudioValidator, AudioValidationError
from zenova.voice.extractor import AcousticFeatureExtractor
from zenova.voice.asr import AcousticHeuristicASR
from zenova.voice.model import VoiceEmotionClassifier, RAVDESS_EMOTIONS, RAVDESS_TO_ZENOVA
from zenova.voice.baseline import VoiceAcousticBaseline
from zenova.voice.placeholder import VoicePlaceholderAnalyzer
from zenova.voice.analyzer import VoiceEmotionAnalyzer
from zenova.voice.fusion import MultimodalFusionEngine


def create_synthetic_wav(
    duration_sec: float = 1.0,
    freq_hz: float = 220.0,
    sr: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Generate in-memory WAV audio bytes with a pure sine tone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    waveform = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ==============================================================================
# 1. AUDIO VALIDATOR TESTS
# ==============================================================================

def test_audio_validator_clean_wav():
    validator = AudioValidator()
    wav_bytes = create_synthetic_wav(duration_sec=1.5, freq_hz=220.0)
    waveform, sr = validator.process_raw_or_base64(audio_bytes=wav_bytes)

    assert waveform.ndim == 1
    assert sr == 16000
    assert len(waveform) == int(1.5 * 16000)
    assert np.max(np.abs(waveform)) > 0.1


def test_audio_validator_base64_decoding():
    validator = AudioValidator()
    wav_bytes = create_synthetic_wav(duration_sec=1.0, freq_hz=150.0)
    b64_str = base64.b64encode(wav_bytes).decode("utf-8")
    data_uri = f"data:audio/wav;base64,{b64_str}"

    waveform, sr = validator.process_raw_or_base64(audio_base64=data_uri)
    assert waveform.ndim == 1
    assert sr == 16000


def test_audio_validator_duration_too_short():
    validator = AudioValidator(min_duration_sec=0.5)
    short_wav = create_synthetic_wav(duration_sec=0.1)
    with pytest.raises(AudioValidationError, match="shorter than minimum"):
        validator.process_raw_or_base64(audio_bytes=short_wav)


def test_audio_validator_silence_detection():
    validator = AudioValidator(min_amplitude=1e-3)
    silent_wav = create_synthetic_wav(duration_sec=1.0, amplitude=1e-6)
    with pytest.raises(AudioValidationError, match="near-silent"):
        validator.process_raw_or_base64(audio_bytes=silent_wav)


# ==============================================================================
# 2. ACOUSTIC FEATURE EXTRACTOR TESTS
# ==============================================================================

def test_acoustic_feature_extractor():
    extractor = AcousticFeatureExtractor()
    sr = 16000
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    # 200 Hz tone with harmonic at 400 Hz
    wave = (0.4 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    features = extractor.extract(wave, sr)

    assert features.duration_seconds == pytest.approx(1.2, abs=0.05)
    assert features.pitch_mean_hz is not None
    assert 180.0 <= features.pitch_mean_hz <= 220.0
    assert features.rms_energy_mean is not None and features.rms_energy_mean > 0.0
    assert features.spectral_centroid_mean is not None and features.spectral_centroid_mean > 0.0
    assert len(features.mfcc_means) == 13
    assert features.jitter_local is not None
    assert features.shimmer_local is not None

    flat = features.to_flat_dict()
    assert "mfcc_1" in flat
    assert "pitch_mean_hz" in flat


# ==============================================================================
# 3. SPEECH-TO-TEXT (ASR) TESTS
# ==============================================================================

def test_acoustic_heuristic_asr():
    asr = AcousticHeuristicASR()
    sr = 16000
    wave = np.zeros(sr, dtype=np.float32)

    # Empty/silent audio
    text_empty, conf_empty = asr.transcribe(wave, sr)
    assert text_empty == ""
    assert conf_empty == 0.0

    # With transcription hint
    wave_active = (0.1 * np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 32000))).astype(np.float32)
    text_hint, conf_hint = asr.transcribe(wave_active, sr, transcription_hint="I am feeling okay today.")
    assert text_hint == "I am feeling okay today."
    assert conf_hint == 0.95

    # Acoustic heuristic inference without hint
    text_auto, conf_auto = asr.transcribe(wave_active, sr)
    assert len(text_auto) > 0
    assert conf_auto >= 0.70


# ==============================================================================
# 4. VOICE EMOTION CLASSIFIER & BASELINE TESTS
# ==============================================================================

def test_voice_emotion_classifier():
    classifier = VoiceEmotionClassifier()
    features = AcousticFeatures(
        duration_seconds=2.5,
        pitch_mean_hz=210.0,
        pitch_std_hz=40.0,
        rms_energy_mean=0.05,
        intensity_db=-22.0,
        spectral_centroid_mean=2100.0,
        spectral_rolloff_mean=3600.0,
        zero_crossing_rate=0.08,
        mfcc_means=[-10.0, 5.0, -2.0, 3.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0],
        jitter_local=0.02,
        shimmer_local=0.04,
        hnr_db=15.0,
    )

    primary, conf, probs, val, aro, dom = classifier.predict(features)

    assert isinstance(primary, EmotionCategory)
    assert 0.0 <= conf <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert -1.0 <= val <= 1.0
    assert -1.0 <= aro <= 1.0
    assert -1.0 <= dom <= 1.0


def test_voice_acoustic_baseline():
    baseline = VoiceAcousticBaseline()
    calm_features = AcousticFeatures(
        duration_seconds=3.0,
        pitch_mean_hz=140.0,
        pitch_std_hz=15.0,
        intensity_db=-32.0,
        jitter_local=0.015,
    )
    primary, conf, probs = baseline.predict(calm_features)
    assert primary in [EmotionCategory.NEUTRAL, EmotionCategory.SADNESS]
    assert conf > 0.0


# ==============================================================================
# 5. VOICE ANALYZER & STRICT OPTIONALITY TESTS
# ==============================================================================

def test_voice_placeholder_analyzer():
    placeholder = VoicePlaceholderAnalyzer()
    inp = UserInput(session_id="s_ph", user_id="u_ph", text="Hello")
    res = placeholder.analyze(inp)
    assert res.is_placeholder is True
    assert res.is_available is False
    assert res.primary_emotion == EmotionCategory.NEUTRAL


def test_voice_analyzer_strict_optionality():
    """Verify that when no audio is provided, VoiceEmotionAnalyzer cleanly returns is_available=False."""
    analyzer = VoiceEmotionAnalyzer()
    inp = UserInput(session_id="s_opt", user_id="u_opt", text="Just regular text here.")
    res = analyzer.analyze(inp)

    assert isinstance(res, VoiceResult)
    assert res.is_available is False
    assert res.is_placeholder is False
    assert res.transcription is None
    assert "Acoustic emotion signals are observational vocal affect proxies" in res.disclaimer


def test_voice_analyzer_with_audio_base64():
    """Verify analyzer decodes base64 audio, extracts features, transcribes, and classifies affect."""
    analyzer = VoiceEmotionAnalyzer()
    wav_bytes = create_synthetic_wav(duration_sec=1.5, freq_hz=230.0)
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    inp = UserInput(
        session_id="s_voice_turn",
        user_id="u_voice_turn",
        text="...",
        metadata={
            "audio_base64": b64_audio,
            "transcription_hint": "I am feeling a little better today.",
        }
    )

    res = analyzer.analyze(inp)
    assert res.is_available is True
    assert res.transcription == "I am feeling a little better today."
    assert res.transcription_confidence == 0.95
    assert res.duration_seconds == pytest.approx(1.5, abs=0.1)
    assert len(res.acoustic_features) > 0
    assert "pitch_mean_hz" in res.acoustic_features


# ==============================================================================
# 6. MULTIMODAL AFFECTIVE FUSION TESTS
# ==============================================================================

def test_multimodal_fusion_congruent():
    fusion = MultimodalFusionEngine()

    text_res = EmotionResult(
        primary_emotion=EmotionCategory.JOY,
        confidence=0.85,
        probabilities={EmotionCategory.JOY.value: 0.85, EmotionCategory.NEUTRAL.value: 0.15},
        valence=0.75,
        arousal=0.50,
        dominance=0.40,
    )
    voice_res = VoiceResult(
        is_available=True,
        primary_emotion=EmotionCategory.JOY,
        confidence=0.75,
        probabilities={EmotionCategory.JOY.value: 0.75, EmotionCategory.NEUTRAL.value: 0.25},
        valence=0.65,
        arousal=0.45,
        dominance=0.35,
    )

    fused = fusion.fuse(text_res, voice_res)
    assert fused.primary_emotion == EmotionCategory.JOY
    assert fused.confidence >= 0.75
    assert fused.valence > 0.60
    assert fused.affective_discrepancy_detected is False


def test_multimodal_fusion_affective_discrepancy():
    """Verify detection of acoustic-semantic incongruence (verbal masking).
    User text is cheerful (+0.80), but vocal acoustic prosody is distressed (-0.75).
    """
    fusion = MultimodalFusionEngine(discrepancy_threshold=0.80)

    text_res = EmotionResult(
        primary_emotion=EmotionCategory.JOY,
        confidence=0.80,
        probabilities={EmotionCategory.JOY.value: 0.80, EmotionCategory.NEUTRAL.value: 0.20},
        valence=0.80,
        arousal=0.30,
        dominance=0.40,
    )
    voice_res = VoiceResult(
        is_available=True,
        primary_emotion=EmotionCategory.SADNESS,
        confidence=0.85,
        probabilities={EmotionCategory.SADNESS.value: 0.85, EmotionCategory.NEUTRAL.value: 0.15},
        valence=-0.75,
        arousal=-0.40,
        dominance=-0.50,
    )

    fused = fusion.fuse(text_res, voice_res)
    assert fused.affective_discrepancy_detected is True
    assert fused.discrepancy_notes is not None
    assert "incongruence" in fused.discrepancy_notes.lower()
