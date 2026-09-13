"""Context Normalizers for ZENOVA Step 9.

Transforms heterogeneous analytical outputs from individual modules into the
standardized, strictly-typed blocks of the MultimodalContext object.

Strict Non-Fabrication Rule:
Any omitted or unavailable modality is marked with is_available=False and
a precise, documented reason. No surrogate metrics are fabricated.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    ConversationTurn,
    DialogStage
)
from zenova.schemas.context import (
    ConversationContextBlock,
    EmotionContextBlock,
    SymptomsContextBlock,
    RiskContextBlock,
    BaselineContextBlock,
    BehaviorContextBlock,
    VoiceContextBlock,
    HistoryContextBlock,
    ValenceTrend
)
from zenova.context.trajectory import TrajectoryAnalyzer


class ContextNormalizers:
    """Normalizers translating module outputs into standardized context blocks."""

    @staticmethod
    def normalize_conversation(
        user_input: UserInput,
        turn_id: int = 1,
        turn_count: int = 1,
        dialog_stage: DialogStage = DialogStage.EXPLORATION,
        input_modalities: Optional[List[str]] = None,
        is_crisis_bypass: bool = False
    ) -> ConversationContextBlock:
        modalities = input_modalities or ["text"]
        return ConversationContextBlock(
            session_id=user_input.session_id,
            user_id=user_input.user_id,
            turn_id=turn_id,
            current_text=user_input.text or "",
            speaker="user",
            dialog_stage=dialog_stage,
            turn_count=turn_count,
            language="en",
            input_modalities=modalities,
            is_crisis_bypass=is_crisis_bypass,
            timestamp=datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_emotion(
        emotion: Optional[EmotionResult],
        voice: Optional[VoiceResult] = None
    ) -> EmotionContextBlock:
        if not emotion:
            return EmotionContextBlock(
                is_available=False,
                reason="emotion_analyzer_omitted",
                primary_emotion=None,
                confidence=0.0,
                source="unavailable"
            )

        # Check if discrepancy was detected between voice and text
        discrepancy = False
        if voice and voice.is_available and voice.valence is not None and emotion.valence is not None:
            # Opposite valence with notable acoustic arousal indicates potential verbal masking
            if (voice.valence * emotion.valence < 0) and (voice.arousal or 0.0) >= 0.4:
                discrepancy = True

        return EmotionContextBlock(
            is_available=True,
            reason=None,
            primary_emotion=emotion.primary_emotion.value if hasattr(emotion.primary_emotion, "value") else str(emotion.primary_emotion),
            confidence=round(emotion.confidence, 4),
            valence=round(emotion.valence or 0.0, 4),
            arousal=round(emotion.arousal or 0.0, 4),
            dominance=round(emotion.dominance or 0.0, 4),
            probabilities={k: round(v, 4) for k, v in (emotion.probabilities or {}).items()},
            source="multimodal_fusion" if (voice and voice.is_available) else "text_analyzer",
            is_discrepancy_detected=discrepancy,
            timestamp=emotion.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_symptoms(symptom: Optional[SymptomResult]) -> SymptomsContextBlock:
        if not symptom:
            return SymptomsContextBlock(
                is_available=False,
                reason="symptom_analyzer_omitted",
                signals=[],
                signal_count=0,
                primary_signals=[],
                max_confidence=0.0
            )

        signal_dicts = []
        primary_signals = []
        max_conf = 0.0

        for sig in symptom.signals:
            raw_lbl = getattr(sig, "marker_name", None) or getattr(sig, "label", None) or str(sig)
            lbl = raw_lbl.value if hasattr(raw_lbl, "value") else str(raw_lbl)
            conf = float(sig.confidence)
            sev = sig.severity.value if hasattr(sig.severity, "value") else str(sig.severity)
            if conf > max_conf:
                max_conf = conf
            if lbl not in primary_signals:
                primary_signals.append(lbl)
            signal_dicts.append({
                "label": lbl,
                "confidence": round(conf, 4),
                "severity": sev,
                "evidence_spans": sig.evidence_spans or []
            })

        return SymptomsContextBlock(
            is_available=True,
            reason=None,
            signals=signal_dicts,
            signal_count=len(signal_dicts),
            primary_signals=primary_signals,
            max_confidence=round(max_conf, 4),
            disclaimer=symptom.disclaimer or "Identified symptom signals are observational patterns and do NOT constitute clinical diagnosis.",
            model_version=getattr(symptom, "model_version", None) or getattr(symptom, "module_version", "1.0.0"),
            timestamp=symptom.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_risk(risk: Optional[RiskResult]) -> RiskContextBlock:
        if not risk:
            return RiskContextBlock(
                is_available=False,
                reason="risk_analyzer_omitted",
                risk_level="unknown",
                confidence=0.0,
                crisis_category="none",
                is_high_risk=False,
                requires_escalation=False,
                trigger_cues=[]
            )

        r_lvl = risk.risk_level.value if hasattr(risk.risk_level, "value") else str(risk.risk_level)
        c_cat = risk.crisis_category.value if hasattr(risk.crisis_category, "value") else str(risk.crisis_category)

        is_high = getattr(risk, "is_high_risk", False)
        req_esc = getattr(risk, "requires_escalation", None)
        if req_esc is None:
            req_esc = getattr(risk, "requires_immediate_escalation", False)

        return RiskContextBlock(
            is_available=True,
            reason=None,
            risk_level=r_lvl,
            confidence=round(risk.confidence, 4),
            crisis_category=c_cat,
            is_high_risk=bool(is_high),
            requires_escalation=bool(req_esc),
            trigger_cues=list(risk.trigger_cues or []),
            model_version=getattr(risk, "model_version", None) or getattr(risk, "module_version", "1.0.0"),
            timestamp=risk.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_baseline(baseline: Optional[BaselineResult]) -> BaselineContextBlock:
        if not baseline:
            return BaselineContextBlock(
                is_available=False,
                reason="baseline_engine_omitted",
                status="insufficient_data",
                confidence=0.0,
                total_observations=0,
                active_features={},
                anomalous_features=[]
            )

        stat = baseline.status.value if hasattr(baseline.status, "value") else str(baseline.status)
        feat_dicts = {}
        anomalous = []

        feats = getattr(baseline, "features", None) or getattr(baseline, "metric_deviations", None) or {}
        if isinstance(feats, dict):
            for k, v in feats.items():
                if hasattr(v, "model_dump"):
                    f_dict = v.model_dump()
                elif isinstance(v, dict):
                    f_dict = v
                else:
                    f_dict = {"value": v}
                feat_dicts[k] = f_dict
                dev = f_dict.get("deviation", 0.0)
                if abs(dev) >= 2.0:
                    anomalous.append(k)

        notes = getattr(baseline, "notes", None) or getattr(baseline, "deviation_notes", None)

        return BaselineContextBlock(
            is_available=True,
            reason=None,
            status=stat,
            confidence=round(baseline.confidence, 4),
            total_observations=int(baseline.total_observations),
            active_features=feat_dicts,
            anomalous_features=anomalous,
            interpretation_summary=notes,
            timestamp=baseline.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_behavior(behavior: Optional[BehavioralResult]) -> BehaviorContextBlock:
        if not behavior or not behavior.is_available:
            reason_str = "passive_sensing_omitted" if not behavior else (
                getattr(behavior, "notes", None) or getattr(behavior, "anomaly_notes", None) or "behavior_unavailable"
            )
            return BehaviorContextBlock(
                is_available=False,
                reason=reason_str,
                source_device=None,
                domains={},
                bdi=0.0,
                is_anomalous=False,
                affected_domains=[],
                confidence=0.0
            )

        # Categorize metrics into standard 5 domains
        metrics = behavior.metrics or {}
        domains = {
            "activity": {k: v for k, v in metrics.items() if "step" in k or "active" in k or "sedentary" in k},
            "sleep": {k: v for k, v in metrics.items() if "sleep" in k or "disturb" in k},
            "mobility": {k: v for k, v in metrics.items() if "radius" in k or "home" in k or "entropy" in k},
            "social": {k: v for k, v in metrics.items() if "conv" in k or "call" in k},
            "digital": {k: v for k, v in metrics.items() if "screen" in k or "unlock" in k or "companion" in k},
        }

        # Filter empty domains
        domains = {k: v for k, v in domains.items() if v}

        return BehaviorContextBlock(
            is_available=True,
            reason=None,
            source_device=behavior.source_device or "unknown_device",
            domains=domains,
            bdi=round(getattr(behavior, "bdi", 0.0) or 0.0, 3),
            is_anomalous=bool(behavior.is_anomalous),
            affected_domains=list(behavior.affected_domains or []),
            confidence=round(behavior.confidence or 0.8, 4),
            disclaimer=behavior.disclaimer or "Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses.",
            timestamp=behavior.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_voice(voice: Optional[VoiceResult]) -> VoiceContextBlock:
        if not voice or not voice.is_available:
            return VoiceContextBlock(
                is_available=False,
                reason="audio_omitted" if not voice else "voice_unavailable",
                transcription=None,
                transcription_confidence=None,
                duration_seconds=None,
                predicted_emotion=None,
                confidence=0.0,
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                acoustic_summary={}
            )

        emo_str = None
        if voice.primary_emotion:
            emo_str = voice.primary_emotion.value if hasattr(voice.primary_emotion, "value") else str(voice.primary_emotion)

        return VoiceContextBlock(
            is_available=True,
            reason=None,
            transcription=voice.transcription,
            transcription_confidence=round(voice.transcription_confidence, 4) if voice.transcription_confidence is not None else None,
            duration_seconds=round(voice.duration_seconds, 2) if voice.duration_seconds is not None else None,
            predicted_emotion=emo_str,
            confidence=round(voice.confidence or 0.0, 4),
            valence=round(voice.valence or 0.0, 4),
            arousal=round(voice.arousal or 0.0, 4),
            dominance=round(voice.dominance or 0.0, 4),
            acoustic_summary={k: round(v, 4) for k, v in (voice.acoustic_features or {}).items()},
            disclaimer=voice.disclaimer or "Acoustic emotion signals are observational vocal affect proxies; do not interpret as psychiatric diagnoses.",
            timestamp=voice.timestamp or datetime.now(timezone.utc)
        )

    @staticmethod
    def normalize_history(
        turns: Optional[List[Any]] = None,
        current_valence: Optional[float] = None,
        current_arousal: Optional[float] = None,
        current_emotion: Optional[str] = None,
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        escalation_events: Optional[List[Dict[str, Any]]] = None,
    ) -> HistoryContextBlock:
        if not turns:
            return HistoryContextBlock(
                turn_count=0,
                recent_turns=[],
                previous_strategies=[],
                previous_outcomes=[],
                user_feedback=user_feedback or [],
                affective_trajectory=[],
                valence_trend=ValenceTrend.INSUFFICIENT_DATA,
                dominant_themes=[],
                escalation_history=escalation_events or []
            )

        turn_dicts = []
        strategies = []

        for t in turns:
            if hasattr(t, "model_dump"):
                td = t.model_dump()
            elif isinstance(t, dict):
                td = t
            else:
                # SQLAlchemy model instance
                import json
                td = {
                    "turn_id": getattr(t, "turn_id", 0),
                    "speaker": getattr(t, "speaker", "user"),
                    "text": getattr(t, "content", ""),
                    "timestamp": getattr(t, "created_at", None),
                    "emotion": json.loads(getattr(t, "emotion_json", None) or "{}"),
                    "symptoms": json.loads(getattr(t, "symptom_json", None) or "{}"),
                    "risk": json.loads(getattr(t, "risk_json", None) or "{}"),
                    "strategy": json.loads(getattr(t, "strategy_json", None) or "{}"),
                    "safety": json.loads(getattr(t, "safety_json", None) or "{}")
                }
            turn_dicts.append(td)

            if td.get("strategy") and td.get("speaker") == "assistant":
                strat = td.get("strategy")
                strategies.append({
                    "turn_id": td.get("turn_id"),
                    "strategy": strat.get("selected_strategy") if isinstance(strat, dict) else str(strat),
                    "stage": strat.get("stage") if isinstance(strat, dict) else "Comforting",
                    "confidence": strat.get("confidence") if isinstance(strat, dict) else 0.5
                })

        # Calculate Trajectory and Outcomes
        traj_res = TrajectoryAnalyzer.analyze_trajectory(
            recent_turns=turn_dicts,
            current_valence=current_valence,
            current_arousal=current_arousal,
            current_emotion=current_emotion
        )

        return HistoryContextBlock(
            turn_count=len(turn_dicts),
            recent_turns=turn_dicts[-10:],  # keep last 10 turns for context efficiency
            previous_strategies=strategies[-5:],
            previous_outcomes=traj_res["previous_outcomes"][-5:],
            user_feedback=user_feedback or [],
            affective_trajectory=traj_res["affective_trajectory"],
            valence_trend=traj_res["valence_trend"],
            dominant_themes=traj_res["dominant_themes"],
            escalation_history=escalation_events or []
        )
