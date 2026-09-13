"""Confidence-calibrated Weighted Rule Fusion baseline with discrepancy detection."""
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

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
from zenova.schemas.fusion import (
    ModalityMask,
    ModalityName,
    CrossModalDiscrepancy,
    FusedMultimodalState
)
from zenova.fusion.masking import ModalityMaskExtractor
from zenova.core.logging import get_logger

logger = get_logger("zenova.fusion.baseline_rule")


class WeightedRuleFusion:
    """Confidence-calibrated, interpretable rule-based multimodal fusion engine."""

    DEFAULT_DOMAIN_PRIORS = {
        ModalityName.EMOTION.value: 0.25,
        ModalityName.RISK.value: 0.25,
        ModalityName.SYMPTOMS.value: 0.15,
        ModalityName.VOICE.value: 0.15,
        ModalityName.BEHAVIOR.value: 0.10,
        ModalityName.BASELINE.value: 0.05,
        ModalityName.HISTORY.value: 0.05,
    }

    def __init__(self, domain_priors: Optional[Dict[str, float]] = None):
        self.domain_priors = domain_priors or self.DEFAULT_DOMAIN_PRIORS

    def compute_modality_weights(self, mask: ModalityMask) -> Dict[str, float]:
        """Compute re-normalized, confidence-scaled modality contribution weights."""
        raw_weights = {}
        for mod_name, prior in self.domain_priors.items():
            if mask.mask.get(mod_name, False):
                conf = mask.confidences.get(mod_name, 0.8)
                raw_weights[mod_name] = prior * max(conf, 0.1)
            else:
                raw_weights[mod_name] = 0.0

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            # Fallback if only text or no analytical modules present
            return {k: (1.0 if k == ModalityName.EMOTION.value else 0.0) for k in self.domain_priors}

        return {k: round(v / total_weight, 4) for k, v in raw_weights.items()}

    def detect_discrepancies(
        self,
        user_input: Optional[UserInput],
        emotion: Optional[EmotionResult],
        symptoms: Optional[SymptomResult],
        risk: Optional[RiskResult],
        voice: Optional[VoiceResult],
        behavior: Optional[BehavioralResult],
        baseline: Optional[BaselineResult],
    ) -> CrossModalDiscrepancy:
        """Detect cross-modal incongruities, emotional masking, and behavioral contradictions."""
        types = []
        reasons = []

        # 1. Acoustic-Semantic Discrepancy (Verbal Masking)
        if emotion and voice and voice.is_available and emotion.confidence > 0.4 and voice.confidence > 0.4:
            txt_val = emotion.valence
            vox_val = voice.valence if hasattr(voice, "valence") else 0.0
            vox_arousal = voice.arousal if hasattr(voice, "arousal") else 0.5

            # Neutral/positive text with negative or highly agitated voice
            if txt_val >= 0.0 and (vox_val < -0.3 or (vox_arousal > 0.7 and voice.primary_emotion in (EmotionCategory.FEAR, EmotionCategory.ANGER, EmotionCategory.SADNESS))):
                types.append("acoustic_semantic_masking")
                reasons.append(f"Verbal text indicates neutral/positive affect ({emotion.primary_emotion.value}, val={txt_val:.2f}) while vocal prosody indicates agitation/distress ({voice.primary_emotion.value}, val={vox_val:.2f}, arousal={vox_arousal:.2f}).")

            # Flat/blunted voice during acute emotional text
            if txt_val < -0.5 and vox_arousal < 0.2:
                types.append("affective_blunting")
                reasons.append(f"Severe negative text ({emotion.primary_emotion.value}) paired with blunted/flat vocal affect (arousal={vox_arousal:.2f}).")

        # 2. Behavioral-Verbal Discrepancy
        if emotion and behavior and behavior.is_available:
            txt_val = emotion.valence
            metrics = getattr(behavior, "metrics", {}) or {}
            anomaly_raw = getattr(behavior, "anomaly_score", None)
            if anomaly_raw is None:
                anomaly_raw = metrics.get("anomaly_score", 0.8 if getattr(behavior, "behavioral_anomaly_detected", False) else 0.0)
            anomaly = float(anomaly_raw)
            sleep_h = metrics.get("sleep_duration_hours")

            # Calm/positive text but severe behavioral anomaly
            if txt_val >= 0.0 and (anomaly > 0.6 or getattr(behavior, "behavioral_anomaly_detected", False)):
                types.append("behavioral_verbal_masking")
                reasons.append(f"User verbally discloses neutral/positive state, but passive sensing indicates high behavioral anomaly ({anomaly:.2f}) with irregular sleep ({sleep_h}h).")

        # 3. Baseline Deviation Discrepancy
        if baseline and baseline.is_significant_deviation and emotion and emotion.valence > 0.2:
            types.append("baseline_affect_discrepancy")
            reasons.append(f"Significant longitudinal baseline deviation (z >= 2.0) despite positive short-term turn valence.")

        # 4. Symptom-Affect Incongruity
        if symptoms and symptoms.signals and emotion:
            has_severe_sym = any(s.severity == SymptomSeverity.SEVERE for s in symptoms.signals)
            if has_severe_sym and emotion.valence > 0.4:
                types.append("symptom_affect_incongruity")
                reasons.append("Severe psychological symptoms reported alongside high positive emotional expression.")

        return CrossModalDiscrepancy(
            detected=len(types) > 0,
            discrepancy_types=types,
            reasons=reasons,
            confidence=0.85 if len(types) > 0 else 0.0
        )

    def fuse(
        self,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> FusedMultimodalState:
        """Perform confidence-calibrated rule-based multimodal fusion."""
        # 1. Extract Mask & Normalized Weights
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
        weights = self.compute_modality_weights(mask)

        # 2. Fused Valence & Arousal
        val_components = []
        arousal_components = []

        # Text Emotion contribution
        if mask.mask[ModalityName.EMOTION.value] and emotion:
            w_e = weights[ModalityName.EMOTION.value]
            val_components.append((emotion.valence, w_e))
            arousal_components.append((emotion.arousal, w_e))

        # Voice contribution
        if mask.mask[ModalityName.VOICE.value] and voice:
            w_v = weights[ModalityName.VOICE.value]
            v_val = voice.valence if hasattr(voice, "valence") else 0.0
            v_arousal = voice.arousal if hasattr(voice, "arousal") else 0.5
            val_components.append((v_val, w_v))
            arousal_components.append((v_arousal, w_v))

        # Risk shift on affect
        if mask.mask[ModalityName.RISK.value] and risk:
            w_r = weights[ModalityName.RISK.value]
            r_val = -0.8 if risk.is_high_risk else (-0.4 if risk.risk_level == RiskLevel.MODERATE else 0.0)
            r_arousal = 0.8 if risk.is_high_risk else 0.5
            val_components.append((r_val, w_r))
            arousal_components.append((r_arousal, w_r))

        # Symptoms shift on affect
        if mask.mask[ModalityName.SYMPTOMS.value] and symptoms:
            w_s = weights[ModalityName.SYMPTOMS.value]
            has_sev = any(s.severity == SymptomSeverity.SEVERE for s in getattr(symptoms, "signals", []))
            s_val = -0.6 if has_sev else -0.3
            val_components.append((s_val, w_s))

        # Calculate weighted sums
        if val_components:
            total_val_w = sum(w for _, w in val_components)
            fused_valence = sum(v * w for v, w in val_components) / max(total_val_w, 1e-6)
        else:
            fused_valence = 0.0

        if arousal_components:
            total_ar_w = sum(w for _, w in arousal_components)
            fused_arousal = sum(a * w for a, w in arousal_components) / max(total_ar_w, 1e-6)
        else:
            fused_arousal = 0.5

        fused_valence = max(-1.0, min(1.0, float(fused_valence)))
        fused_arousal = max(0.0, min(1.0, float(fused_arousal)))

        # 3. Fused Distress Severity Score (0.0 to 1.0)
        distress_factors = []
        if emotion:
            distress_factors.append((max(0.0, -emotion.valence), weights.get("emotion", 0.25)))
        if symptoms and symptoms.signals:
            sev_score = max((1.0 if s.severity == SymptomSeverity.SEVERE else (0.6 if s.severity == SymptomSeverity.MODERATE else 0.3)) for s in symptoms.signals)
            distress_factors.append((sev_score, weights.get("symptoms", 0.15)))
        if risk:
            r_score = 1.0 if risk.is_high_risk else (0.6 if risk.risk_level == RiskLevel.MODERATE else 0.2)
            distress_factors.append((r_score, weights.get("risk", 0.25)))
        if voice and voice.is_available:
            v_score = max(0.0, -getattr(voice, "valence", 0.0)) * 0.7 + getattr(voice, "arousal", 0.5) * 0.3
            distress_factors.append((v_score, weights.get("voice", 0.15)))
        if behavior and behavior.is_available:
            distress_factors.append((getattr(behavior, "anomaly_score", 0.0), weights.get("behavior", 0.10)))
        if baseline and baseline.is_significant_deviation:
            distress_factors.append((0.85, weights.get("baseline", 0.05)))

        if distress_factors:
            total_d_w = sum(w for _, w in distress_factors)
            fused_distress = sum(d * w for d, w in distress_factors) / max(total_d_w, 1e-6)
        else:
            fused_distress = 0.0
        fused_distress = max(0.0, min(1.0, float(fused_distress)))

        # 4. Discrepancy Detection
        discrepancy = self.detect_discrepancies(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline
        )

        # 5. Fused Risk Level & Urgency
        fused_risk = RiskLevel.LOW
        if risk:
            fused_risk = risk.risk_level

        # Multi-modal risk escalation:
        # If masking discrepancy is detected, or distress is acute, escalate risk
        if discrepancy.detected and (fused_distress >= 0.35 or any("masking" in t for t in discrepancy.discrepancy_types)):
            if fused_risk in (RiskLevel.LOW, RiskLevel.MODERATE):
                fused_risk = RiskLevel.HIGH
                logger.info("Multimodal fusion escalated risk to HIGH due to detected cross-modal discrepancy and elevated distress.")
        elif fused_risk == RiskLevel.MODERATE:
            corroborating_signals = 0
            if symptoms and any(s.severity == SymptomSeverity.SEVERE for s in symptoms.signals):
                corroborating_signals += 1
            b_anom = getattr(behavior, "anomaly_score", None) or (getattr(behavior, "metrics", {}) or {}).get("anomaly_score", 0.0) or (0.8 if getattr(behavior, "behavioral_anomaly_detected", False) else 0.0)
            if behavior and float(b_anom) > 0.6:
                corroborating_signals += 1
            if baseline and baseline.is_significant_deviation:
                corroborating_signals += 1
            if corroborating_signals >= 2:
                fused_risk = RiskLevel.HIGH
                logger.info("Multimodal fusion promoted risk from MODERATE to HIGH based on corroborating modalities.")

        urgency_score = min(1.0, fused_distress * 0.6 + (1.0 if fused_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL) else (0.5 if fused_risk == RiskLevel.MODERATE else 0.1)) * 0.4)

        # 6. Primary Affect Category
        primary_affect = EmotionCategory.NEUTRAL
        if emotion and emotion.confidence > 0.0:
            primary_affect = emotion.primary_emotion
        elif voice and voice.is_available and voice.primary_emotion:
            primary_affect = voice.primary_emotion

        # Overall confidence
        active_confs = [mask.confidences[m] for m in mask.active_modalities if m in mask.confidences]
        overall_conf = float(np.mean(active_confs)) if active_confs else 0.8

        return FusedMultimodalState(
            primary_affect=primary_affect,
            fused_valence=round(fused_valence, 3),
            fused_arousal=round(fused_arousal, 3),
            fused_dominance=0.0,
            fused_distress_score=round(fused_distress, 3),
            fused_risk_level=fused_risk,
            urgency_score=round(urgency_score, 3),
            modality_weights=weights,
            modality_mask=mask.mask,
            discrepancy=discrepancy,
            fusion_method="weighted_rule",
            confidence=round(overall_conf, 3),
            timestamp=datetime.now(timezone.utc)
        )
