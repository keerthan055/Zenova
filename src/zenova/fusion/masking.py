"""Modality availability mask and confidence extraction for ZENOVA Multimodal Fusion."""
import numpy as np
from typing import Optional, List, Any
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
)
from zenova.schemas.fusion import ModalityMask, ModalityName


class ModalityMaskExtractor:
    """Evaluates availability and confidence across all potential modalities."""

    DEFAULT_CONFIDENCES = {
        ModalityName.TEXT.value: 1.0,
        ModalityName.EMOTION.value: 0.85,
        ModalityName.SYMPTOMS.value: 0.80,
        ModalityName.RISK.value: 0.90,
        ModalityName.VOICE.value: 0.80,
        ModalityName.BEHAVIOR.value: 0.75,
        ModalityName.BASELINE.value: 0.85,
        ModalityName.HISTORY.value: 0.90,
    }

    @classmethod
    def extract_mask(
        cls,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> ModalityMask:
        """Construct the availability mask and confidence scores for each modality."""
        mask = {}
        confidences = {}

        # 1. Text
        has_text = bool(user_input and user_input.text and user_input.text.strip())
        mask[ModalityName.TEXT.value] = has_text
        confidences[ModalityName.TEXT.value] = 1.0 if has_text else 0.0

        # 2. Text Emotion
        has_emotion = bool(emotion and getattr(emotion, "confidence", 0.0) > 0.0)
        mask[ModalityName.EMOTION.value] = has_emotion
        confidences[ModalityName.EMOTION.value] = float(getattr(emotion, "confidence", 0.0)) if has_emotion else 0.0

        # 3. Symptoms
        sig_list = getattr(symptoms, "signals", []) if symptoms else []
        has_symptoms = bool(symptoms and (len(sig_list) > 0 or getattr(symptoms, "is_available", False)))
        if has_symptoms:
            sym_conf = float(np.mean([s.confidence for s in sig_list])) if sig_list else 0.80
        else:
            sym_conf = 0.0
        mask[ModalityName.SYMPTOMS.value] = has_symptoms
        confidences[ModalityName.SYMPTOMS.value] = sym_conf

        # 4. Risk
        has_risk = bool(risk and getattr(risk, "confidence", 0.0) > 0.0)
        mask[ModalityName.RISK.value] = has_risk
        confidences[ModalityName.RISK.value] = float(getattr(risk, "confidence", 0.0)) if has_risk else 0.0

        # 5. Voice
        has_voice = bool(voice and getattr(voice, "is_available", False))
        mask[ModalityName.VOICE.value] = has_voice
        confidences[ModalityName.VOICE.value] = float(getattr(voice, "confidence", 0.0)) if has_voice else 0.0

        # 6. Behavioral
        has_behavior = bool(behavior and getattr(behavior, "is_available", False))
        mask[ModalityName.BEHAVIOR.value] = has_behavior
        confidences[ModalityName.BEHAVIOR.value] = float(getattr(behavior, "confidence", 0.75)) if has_behavior else 0.0

        # 7. Baseline
        has_baseline = bool(baseline and getattr(baseline, "total_observations", 0) > 0)
        mask[ModalityName.BASELINE.value] = has_baseline
        confidences[ModalityName.BASELINE.value] = float(getattr(baseline, "confidence", 0.85)) if has_baseline else 0.0

        # 8. History
        has_history = bool(history and len(history) > 0)
        mask[ModalityName.HISTORY.value] = has_history
        confidences[ModalityName.HISTORY.value] = 0.90 if has_history else 0.0

        active_mods = [k for k, v in mask.items() if v]

        return ModalityMask(
            mask=mask,
            confidences=confidences,
            available_count=len(active_mods),
            active_modalities=active_mods
        )
