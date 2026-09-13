"""Multimodal feature extraction for personal baseline tracking.

Gracefully extracts standardized numerical features from text turns, emotion outputs,
symptom signals, risk triage, and optional voice / user-reported wellbeing inputs.
Missing modalities are omitted without raising errors.
"""
from typing import Dict, Any, Optional
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    ConversationContext,
    RiskLevel
)

# Valence mappings for Ekman emotion categories
EKMAN_VALENCE = {
    "joy": 0.8,
    "surprise": 0.2,
    "neutral": 0.0,
    "sadness": -0.8,
    "anger": -0.7,
    "fear": -0.7,
    "disgust": -0.6
}

EKMAN_AROUSAL = {
    "anger": 0.85,
    "fear": 0.80,
    "joy": 0.75,
    "surprise": 0.70,
    "disgust": 0.50,
    "sadness": 0.35,
    "neutral": 0.20
}

RISK_SEVERITY_MAP = {
    RiskLevel.LOW.value: 0.0,
    RiskLevel.MODERATE.value: 1.0,
    RiskLevel.HIGH.value: 2.0,
    RiskLevel.CRITICAL.value: 3.0,
    RiskLevel.LOW: 0.0,
    RiskLevel.MODERATE: 1.0,
    RiskLevel.HIGH: 2.0,
    RiskLevel.CRITICAL: 3.0
}


class FeatureExtractor:
    """Extracts a normalized numerical feature dictionary from available modalities."""

    @classmethod
    def extract_from_turn(
        cls,
        user_input: UserInput,
        emotion: Optional[EmotionResult] = None,
        symptom: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        context: Optional[ConversationContext] = None,
        extra_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Extract all available features for a single interaction turn."""
        features: Dict[str, float] = {}

        # 1. Behavioral Text Features
        text = user_input.text.strip()
        words = text.split()
        features["word_count"] = float(len(words))
        features["char_count"] = float(len(text))

        # 2. Emotion Features (Valence, Arousal, Dominant Confidences)
        if emotion and not emotion.is_placeholder:
            if getattr(emotion, "valence", None) is not None and emotion.valence != 0.0:
                features["valence_score"] = float(emotion.valence)
            if getattr(emotion, "arousal", None) is not None and emotion.arousal != 0.0:
                features["arousal_score"] = float(emotion.arousal)

            probs = getattr(emotion, "probabilities", {}) or {}
            if probs:
                for em_k, em_p in probs.items():
                    features[f"emotion_{em_k.lower()}"] = round(float(em_p), 4)
                if "valence_score" not in features:
                    val = sum(EKMAN_VALENCE.get(k.lower(), 0.0) * p for k, p in probs.items())
                    features["valence_score"] = round(val, 4)
                if "arousal_score" not in features:
                    aro = sum(EKMAN_AROUSAL.get(k.lower(), 0.3) * p for k, p in probs.items())
                    features["arousal_score"] = round(aro, 4)
            elif getattr(emotion, "primary_emotion", None):
                dom = str(getattr(emotion.primary_emotion, "value", emotion.primary_emotion)).lower()
                if "valence_score" not in features:
                    features["valence_score"] = EKMAN_VALENCE.get(dom, 0.0)
                if "arousal_score" not in features:
                    features["arousal_score"] = EKMAN_AROUSAL.get(dom, 0.3)

        # 3. Symptom / Signal Features
        if symptom and not symptom.is_placeholder:
            active_count = 0
            for sig in symptom.signals:
                active_count += 1
                lbl = getattr(sig, "marker_name", getattr(sig, "label", ""))
                norm_label = str(lbl).lower().replace(" ", "_").replace("/", "_")
                features[f"symptom_{norm_label}"] = round(sig.confidence, 4)
                # Specific alias for common symptoms
                if "anxiety" in norm_label or "panic" in norm_label:
                    features["anxiety_score"] = round(sig.confidence, 4)
                if "sleep" in norm_label:
                    features["sleep_disturbance_score"] = round(sig.confidence, 4)
                if "depressed" in norm_label or "loss_of_interest" in norm_label:
                    features["depressive_score"] = round(sig.confidence, 4)

            features["active_symptoms_count"] = float(active_count)

        # 4. Risk Features
        if risk and not risk.is_placeholder:
            features["risk_severity"] = float(RISK_SEVERITY_MAP.get(risk.risk_level, 0.0))
            features["risk_confidence"] = round(float(risk.confidence), 4)

        # 5. Extra / Out-of-band features (Voice, EMA, Questionnaires)
        if extra_features:
            for k, v in extra_features.items():
                if isinstance(v, (int, float)):
                    features[k] = round(float(v), 4)

        return features
