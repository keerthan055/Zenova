"""Feature vectorizer mapping heterogeneous multimodal inputs into numerical representations."""
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
    EmotionCategory,
    RiskLevel,
    SymptomSeverity
)
from zenova.schemas.fusion import ModalityMask


class MultimodalFeatureVectorizer:
    """Encodes available modalities into fixed-dimension numerical vectors with zero-imputation."""

    MODALITY_DIMS = {
        "text": 4,
        "emotion": 6,
        "symptoms": 6,
        "risk": 6,
        "voice": 8,
        "behavior": 4,
        "baseline": 4,
        "history": 4
    }
    TOTAL_DIM = sum(MODALITY_DIMS.values())  # 42
    FEATURE_DIM = TOTAL_DIM

    EMOTION_MAP = {
        EmotionCategory.NEUTRAL: 0,
        EmotionCategory.JOY: 1,
        EmotionCategory.SADNESS: 2,
        EmotionCategory.ANGER: 3,
        EmotionCategory.ANXIETY: 4,
        EmotionCategory.FEAR: 5,
        EmotionCategory.FRUSTRATION: 6,
        EmotionCategory.HOPE: 7,
        EmotionCategory.GRIEF: 8,
        EmotionCategory.GUILT: 9,
        EmotionCategory.SHAME: 10
    }

    RISK_MAP = {
        RiskLevel.LOW: 0.25,
        RiskLevel.MODERATE: 0.50,
        RiskLevel.HIGH: 0.75,
        RiskLevel.CRITICAL: 1.00
    }

    SEVERITY_MAP = {
        SymptomSeverity.NONE: 0.0,
        SymptomSeverity.SUBCLINICAL: 0.25,
        SymptomSeverity.MILD: 0.50,
        SymptomSeverity.MODERATE: 0.75,
        SymptomSeverity.SEVERE: 1.00
    }

    @classmethod
    def vectorize_text(cls, user_input: Optional[UserInput]) -> np.ndarray:
        if not user_input or not user_input.text:
            return np.zeros(cls.MODALITY_DIMS["text"], dtype=np.float32)
        txt = user_input.text
        length = min(len(txt) / 500.0, 1.0)
        word_count = min(len(txt.split()) / 100.0, 1.0)
        has_negation = 1.0 if any(w in txt.lower().split() for w in ["not", "never", "no", "cant", "can't"]) else 0.0
        has_crisis_keyword = 1.0 if any(w in txt.lower() for w in ["die", "kill", "suicide", "end it", "hopeless"]) else 0.0
        return np.array([length, word_count, has_negation, has_crisis_keyword], dtype=np.float32)

    @classmethod
    def vectorize_emotion(cls, emotion: Optional[EmotionResult]) -> np.ndarray:
        if not emotion or getattr(emotion, "confidence", 0.0) <= 0.0:
            return np.zeros(cls.MODALITY_DIMS["emotion"], dtype=np.float32)
        primary_idx = cls.EMOTION_MAP.get(emotion.primary_emotion, 0) / 8.0
        valence = float(getattr(emotion, "valence", 0.0))
        arousal = float(getattr(emotion, "arousal", 0.0))
        dominance = float(getattr(emotion, "dominance", 0.0))
        conf = float(getattr(emotion, "confidence", 1.0))
        is_negative = 1.0 if valence < -0.3 else 0.0
        return np.array([primary_idx, valence, arousal, dominance, conf, is_negative], dtype=np.float32)

    @classmethod
    def vectorize_symptoms(cls, symptoms: Optional[SymptomResult]) -> np.ndarray:
        sig_list = getattr(symptoms, "signals", []) if symptoms else []
        if not symptoms or (not sig_list and not getattr(symptoms, "is_available", False)):
            return np.zeros(cls.MODALITY_DIMS["symptoms"], dtype=np.float32)
        total_signals = min(len(sig_list) / 5.0, 1.0)
        sev_count = sum(1 for s in sig_list if s.severity == SymptomSeverity.SEVERE) / 3.0
        mod_count = sum(1 for s in sig_list if s.severity == SymptomSeverity.MODERATE) / 3.0
        mild_count = sum(1 for s in sig_list if s.severity == SymptomSeverity.MILD) / 3.0
        has_severe = 1.0 if sev_count > 0 else 0.0
        conf = float(np.mean([s.confidence for s in sig_list])) if sig_list else 0.80
        return np.array([total_signals, min(sev_count, 1.0), min(mod_count, 1.0), min(mild_count, 1.0), has_severe, conf], dtype=np.float32)

    @classmethod
    def vectorize_risk(cls, risk: Optional[RiskResult]) -> np.ndarray:
        if not risk or getattr(risk, "confidence", 0.0) <= 0.0:
            return np.zeros(cls.MODALITY_DIMS["risk"], dtype=np.float32)
        rl_val = cls.RISK_MAP.get(risk.risk_level, 0.0)
        is_high = 1.0 if getattr(risk, "is_high_risk", False) else 0.0
        conf = float(getattr(risk, "confidence", 1.0))
        suicide_prob = float(getattr(risk, "suicide_risk_probability", rl_val))
        cues_count = min(len(getattr(risk, "trigger_cues", [])) / 5.0, 1.0)
        urgency = min(rl_val * 0.7 + is_high * 0.3, 1.0)
        return np.array([rl_val, is_high, conf, suicide_prob, cues_count, urgency], dtype=np.float32)

    @classmethod
    def vectorize_voice(cls, voice: Optional[VoiceResult]) -> np.ndarray:
        if not voice or not getattr(voice, "is_available", False):
            return np.zeros(cls.MODALITY_DIMS["voice"], dtype=np.float32)
        v_dur_raw = getattr(voice, "duration_seconds", 0.0)
        v_dur = min(float(v_dur_raw if v_dur_raw is not None else 0.0) / 15.0, 1.0)
        v_conf_raw = getattr(voice, "confidence", 1.0)
        v_conf = float(v_conf_raw if v_conf_raw is not None else 1.0)
        em_idx = cls.EMOTION_MAP.get(voice.primary_emotion, 0) / 10.0 if voice.primary_emotion else 0.0

        ac = getattr(voice, "acoustic_features", {}) or {}
        pitch = min(float(ac.get("pitch_mean", 150.0)) / 400.0, 1.0)
        jitter = min(float(ac.get("jitter", 0.01)) * 50.0, 1.0)
        shimmer = min(float(ac.get("shimmer", 0.03)) * 20.0, 1.0)
        val = float(getattr(voice, "valence", 0.0) or 0.0)
        arousal = float(getattr(voice, "arousal", 0.5) or 0.5)
        return np.array([v_dur, v_conf, em_idx, pitch, jitter, shimmer, val, arousal], dtype=np.float32)

    @classmethod
    def vectorize_behavior(cls, behavior: Optional[BehavioralResult]) -> np.ndarray:
        if not behavior or not getattr(behavior, "is_available", False):
            return np.zeros(cls.MODALITY_DIMS["behavior"], dtype=np.float32)
        m = getattr(behavior, "metrics", {}) or {}
        steps_norm = min(float(m.get("step_count", 5000)) / 15000.0, 1.0)
        screen_norm = min(float(m.get("screen_time_hours", 4.0)) / 16.0, 1.0)
        sleep_norm = min(float(m.get("sleep_duration_hours", 7.0)) / 12.0, 1.0)
        anomaly_raw = getattr(behavior, "anomaly_score", None)
        if anomaly_raw is None:
            anomaly_raw = m.get("anomaly_score", 0.8 if getattr(behavior, "behavioral_anomaly_detected", False) else 0.0)
        anomaly = float(anomaly_raw)
        return np.array([steps_norm, screen_norm, sleep_norm, anomaly], dtype=np.float32)

    @classmethod
    def vectorize_baseline(cls, baseline: Optional[BaselineResult]) -> np.ndarray:
        if not baseline or getattr(baseline, "total_observations", 0) <= 0:
            return np.zeros(cls.MODALITY_DIMS["baseline"], dtype=np.float32)
        is_dev = 1.0 if getattr(baseline, "is_significant_deviation", False) else 0.0
        obs_norm = min(float(getattr(baseline, "total_observations", 0)) / 30.0, 1.0)
        conf = float(getattr(baseline, "confidence", 0.85))

        dev_dict = getattr(baseline, "metric_deviations", {}) or {}
        max_dev = 0.0
        if dev_dict:
            max_dev = min(max(abs(float(v)) for v in dev_dict.values()) / 4.0, 1.0)
        return np.array([is_dev, obs_norm, conf, max_dev], dtype=np.float32)

    @classmethod
    def vectorize_history(cls, history: Optional[List[Any]]) -> np.ndarray:
        if not history:
            return np.zeros(cls.MODALITY_DIMS["history"], dtype=np.float32)
        turns_count = min(len(history) / 20.0, 1.0)
        user_turns = [t for t in history if getattr(t, "speaker", "") in ("user", "SpeakerRole.USER")]
        user_turn_norm = min(len(user_turns) / 10.0, 1.0)

        # Valence trend
        valences = []
        for t in user_turns:
            em = getattr(t, "emotion", None)
            if em and hasattr(em, "valence"):
                valences.append(float(em.valence))
        avg_val = float(np.mean(valences)) if valences else 0.0
        val_delta = float(valences[-1] - valences[0]) if len(valences) >= 2 else 0.0

        return np.array([turns_count, user_turn_norm, avg_val, val_delta], dtype=np.float32)

    @classmethod
    def vectorize_all(
        cls,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Returns feature vectors by modality and a unified concatenated vector."""
        v_text = cls.vectorize_text(user_input)
        v_emo = cls.vectorize_emotion(emotion)
        v_sym = cls.vectorize_symptoms(symptoms)
        v_risk = cls.vectorize_risk(risk)
        v_voice = cls.vectorize_voice(voice)
        v_beh = cls.vectorize_behavior(behavior)
        v_base = cls.vectorize_baseline(baseline)
        v_hist = cls.vectorize_history(history)

        combined = np.concatenate([v_text, v_emo, v_sym, v_risk, v_voice, v_beh, v_base, v_hist])

        return {
            "text": v_text,
            "emotion": v_emo,
            "symptoms": v_sym,
            "risk": v_risk,
            "voice": v_voice,
            "behavior": v_beh,
            "baseline": v_base,
            "history": v_hist,
            "combined": combined
        }

    @classmethod
    def vectorize(
        cls,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from zenova.fusion.masking import ModalityMaskExtractor
        vecs = cls.vectorize_all(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline,
            history=history
        )
        mask = ModalityMaskExtractor.extract_mask(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline,
            history=history
        )
        mod_names = ["text", "emotion", "symptoms", "risk", "voice", "behavior", "baseline", "history"]
        mask_arr = np.array([1.0 if mask.mask.get(m, False) else 0.0 for m in mod_names], dtype=np.float32)
        conf_arr = np.array([mask.confidences.get(m, 0.0) for m in mod_names], dtype=np.float32)
        return vecs["combined"], mask_arr, conf_arr
