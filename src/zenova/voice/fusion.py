"""Multimodal Affective Fusion Engine for ZENOVA.

Combines text semantic emotion analysis (GoEmotions) with acoustic speech emotion analysis (RAVDESS).
Performs confidence-weighted late fusion and detects acoustic-semantic discrepancies
(e.g., verbal masking where user says "I am fine" with flat or distressed vocal affect).
"""
from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field

from zenova.schemas.standard import EmotionResult, VoiceResult, EmotionCategory
from zenova.core.logging import get_logger

logger = get_logger("zenova.voice.fusion")


class MultimodalAffectiveResult(BaseModel):
    """Result of combining text semantic emotion and acoustic vocal affect."""
    primary_emotion: EmotionCategory
    confidence: float
    fused_probabilities: Dict[str, float]
    valence: float
    arousal: float
    dominance: float
    text_weight: float
    voice_weight: float
    affective_discrepancy_detected: bool = False
    discrepancy_notes: Optional[str] = None


class MultimodalFusionEngine:
    """Combines text and acoustic modalities using confidence-weighted late fusion."""

    def __init__(
        self,
        base_text_weight: float = 0.55,
        base_voice_weight: float = 0.45,
        discrepancy_threshold: float = 0.8,
    ):
        self.base_text_weight = base_text_weight
        self.base_voice_weight = base_voice_weight
        self.discrepancy_threshold = discrepancy_threshold

    def fuse(
        self,
        text_emotion: EmotionResult,
        voice_emotion: VoiceResult,
    ) -> MultimodalAffectiveResult:
        """Blend text and acoustic emotion predictions into a unified affective assessment."""
        # 1. If voice is unavailable, return text emotion directly
        if not voice_emotion.is_available:
            return MultimodalAffectiveResult(
                primary_emotion=text_emotion.primary_emotion,
                confidence=text_emotion.confidence,
                fused_probabilities=text_emotion.probabilities,
                valence=text_emotion.valence,
                arousal=text_emotion.arousal,
                dominance=text_emotion.dominance,
                text_weight=1.0,
                voice_weight=0.0,
                affective_discrepancy_detected=False,
            )

        # 2. Compute dynamic weights based on respective confidence scores
        raw_text_w = self.base_text_weight * max(0.1, text_emotion.confidence)
        raw_voice_w = self.base_voice_weight * max(0.1, voice_emotion.confidence)
        total_w = raw_text_w + raw_voice_w

        w_text = raw_text_w / total_w
        w_voice = raw_voice_w / total_w

        # 3. Fuse probability distributions over EmotionCategory
        fused_probs: Dict[str, float] = {}
        for cat in EmotionCategory:
            p_text = text_emotion.probabilities.get(cat.value, 0.0)
            p_voice = voice_emotion.probabilities.get(cat.value, 0.0)
            fused_p = (w_text * p_text) + (w_voice * p_voice)
            fused_probs[cat.value] = round(fused_p, 4)

        # Re-normalize
        p_tot = sum(fused_probs.values())
        if p_tot > 0:
            fused_probs = {k: round(v / p_tot, 4) for k, v in fused_probs.items()}

        best_cat_str = max(fused_probs, key=fused_probs.get)
        fused_primary = EmotionCategory(best_cat_str)
        fused_confidence = float(fused_probs[best_cat_str])

        # 4. Blend VAD coordinates
        fused_val = round(w_text * text_emotion.valence + w_voice * voice_emotion.valence, 3)
        fused_aro = round(w_text * text_emotion.arousal + w_voice * voice_emotion.arousal, 3)
        fused_dom = round(w_text * text_emotion.dominance + w_voice * voice_emotion.dominance, 3)

        # 5. Detect Acoustic-Semantic Discrepancy
        valence_delta = abs(text_emotion.valence - voice_emotion.valence)
        discrepancy_detected = valence_delta >= self.discrepancy_threshold
        discrepancy_notes = None

        if discrepancy_detected:
            discrepancy_notes = (
                f"Affective incongruence observed: text valence ({text_emotion.valence:.2f}, "
                f"{text_emotion.primary_emotion.value}) diverges from acoustic vocal affect "
                f"({voice_emotion.valence:.2f}, {voice_emotion.primary_emotion.value})."
            )
            logger.info(f"Multimodal discrepancy detected: {discrepancy_notes}")

        return MultimodalAffectiveResult(
            primary_emotion=fused_primary,
            confidence=fused_confidence,
            fused_probabilities=fused_probs,
            valence=fused_val,
            arousal=fused_aro,
            dominance=fused_dom,
            text_weight=round(w_text, 3),
            voice_weight=round(w_voice, 3),
            affective_discrepancy_detected=discrepancy_detected,
            discrepancy_notes=discrepancy_notes,
        )
