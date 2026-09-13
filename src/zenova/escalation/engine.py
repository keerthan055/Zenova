"""Escalation Decision Engine evaluating risk, safety, longitudinal, repeated, and clinician signals."""
from typing import Optional, List, Dict, Any

from zenova.schemas.standard import (
    UserInput,
    RiskResult,
    RiskLevel,
    SafetyResult,
    SafetyAction,
    BaselineResult,
    BehavioralResult,
    ConversationTurn,
    MultimodalContext,
    EmotionCategory,
    SymptomSeverity
)
from zenova.schemas.escalation import (
    EscalationSeverity,
    EscalationTriggerType,
    EscalationReason,
    EscalationContextSnapshot,
    EscalationDecision
)
from zenova.escalation.rules import ClinicianRuleEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.escalation.engine")


class EscalationDecisionEngine:
    """Core decision engine evaluating whether a dialogue turn requires clinician intervention."""

    SEVERITY_ORDER = {
        EscalationSeverity.CRITICAL: 4,
        EscalationSeverity.HIGH: 3,
        EscalationSeverity.MEDIUM: 2,
        EscalationSeverity.LOW: 1
    }

    def __init__(self, config_path: str = "configs/escalation.yaml"):
        self.rule_engine = ClinicianRuleEngine(config_path=config_path)
        logger.info("Initialized EscalationDecisionEngine.")

    def evaluate(
        self,
        user_input: UserInput,
        risk: Optional[RiskResult] = None,
        safety: Optional[SafetyResult] = None,
        baseline: Optional[BaselineResult] = None,
        behavior: Optional[BehavioralResult] = None,
        history: Optional[List[ConversationTurn]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> EscalationDecision:
        """Evaluates all 5 trigger categories to make an escalation decision."""
        potential_reasons: List[tuple[EscalationSeverity, EscalationReason]] = []

        # -------------------------------------------------------------
        # 1. High / Critical Risk Classification
        # -------------------------------------------------------------
        if risk:
            if risk.risk_level == RiskLevel.CRITICAL or risk.requires_immediate_escalation:
                potential_reasons.append((
                    EscalationSeverity.CRITICAL,
                    EscalationReason(
                        trigger_type=EscalationTriggerType.CRISIS_RISK,
                        code="CRITICAL_RISK_DETECTED",
                        title="Critical Crisis Risk Detected",
                        description=f"Model assessed critical risk: category='{risk.crisis_category.value}', confidence={risk.confidence:.2f}.",
                        trigger_cues=risk.trigger_cues,
                        confidence=risk.confidence,
                        metadata={"crisis_category": risk.crisis_category.value, "risk_level": "critical"}
                    )
                ))
            elif risk.risk_level == RiskLevel.HIGH:
                potential_reasons.append((
                    EscalationSeverity.HIGH,
                    EscalationReason(
                        trigger_type=EscalationTriggerType.CRISIS_RISK,
                        code="HIGH_RISK_DETECTED",
                        title="High Crisis Risk Detected",
                        description=f"Model assessed high risk: category='{risk.crisis_category.value}', confidence={risk.confidence:.2f}.",
                        trigger_cues=risk.trigger_cues,
                        confidence=risk.confidence,
                        metadata={"crisis_category": risk.crisis_category.value, "risk_level": "high"}
                    )
                ))

        # -------------------------------------------------------------
        # 2. Safety Gate Events
        # -------------------------------------------------------------
        if safety:
            if safety.action == SafetyAction.BLOCK_AND_ESCALATE:
                potential_reasons.append((
                    EscalationSeverity.CRITICAL,
                    EscalationReason(
                        trigger_type=EscalationTriggerType.SAFETY_GATE_EVENT,
                        code="SAFETY_GATE_BLOCK_AND_ESCALATE",
                        title="Safety Gate Intercepted Critical Threat",
                        description=f"Response Safety Gate issued BLOCK_AND_ESCALATE. Violated policies: {safety.violated_policies}.",
                        trigger_cues=safety.reason_codes,
                        confidence=1.0,
                        metadata={"violated_policies": safety.violated_policies}
                    )
                ))
            elif any(p in ("harmful_instructions", "crisis_mishandling") for p in safety.violated_policies):
                potential_reasons.append((
                    EscalationSeverity.CRITICAL,
                    EscalationReason(
                        trigger_type=EscalationTriggerType.SAFETY_GATE_EVENT,
                        code="SAFETY_CRITICAL_VIOLATION",
                        title="Critical Safety Policy Breach",
                        description=f"Candidate text breached critical safety policy: {safety.violated_policies}.",
                        trigger_cues=safety.reason_codes,
                        confidence=1.0,
                        metadata={"violated_policies": safety.violated_policies}
                    )
                ))

        # -------------------------------------------------------------
        # 3. Configured Clinician Rules
        # -------------------------------------------------------------
        clinician_reasons = self.rule_engine.evaluate(user_input.text)
        for cr in clinician_reasons:
            sev_str = cr.metadata.get("severity", "high").lower()
            sev = EscalationSeverity(sev_str) if sev_str in [s.value for s in EscalationSeverity] else EscalationSeverity.HIGH
            potential_reasons.append((sev, cr))

        # -------------------------------------------------------------
        # 4. Repeated Concerning Signals across recent history
        # -------------------------------------------------------------
        if history and len(history) >= 3:
            import json
            recent_turns = []
            for t in history:
                spk = getattr(t, "speaker", None)
                spk_val = getattr(spk, "value", spk) if spk else None
                if spk_val == "user":
                    recent_turns.append(t)
            recent_turns = recent_turns[-3:]

            if len(recent_turns) >= 3:
                acute_emotions = {"sadness", "fear", "grief", "shame"}
                consecutive_acute_turns = 0
                for t in recent_turns:
                    emotion_obj = getattr(t, "emotion", None)
                    primary_em = None
                    val = 0.0
                    if emotion_obj:
                        pe = getattr(emotion_obj, "primary_emotion", None)
                        primary_em = getattr(pe, "value", pe)
                        val = getattr(emotion_obj, "valence", 0.0)
                    elif hasattr(t, "emotion_json") and t.emotion_json:
                        try:
                            em_data = json.loads(t.emotion_json)
                            primary_em = em_data.get("primary_emotion")
                            val = em_data.get("valence", 0.0)
                        except Exception:
                            pass

                    if primary_em in acute_emotions and val <= -0.6:
                        consecutive_acute_turns += 1

                if consecutive_acute_turns >= 3:
                    potential_reasons.append((
                        EscalationSeverity.HIGH,
                        EscalationReason(
                            trigger_type=EscalationTriggerType.REPEATED_SIGNALS,
                            code="REPEATED_CONCERNING_AFFECT",
                            title="Persistent Acute Negative Affect",
                            description="User demonstrated severe negative affect and distress across 3+ consecutive turns.",
                            trigger_cues=["persistent_negative_valence"],
                            confidence=0.85
                        )
                    ))

                # Check consecutive severe symptoms across 3 turns
                consecutive_symptom_turns = 0
                for t in recent_turns:
                    has_sev = False
                    sym_obj = getattr(t, "symptoms", None)
                    if sym_obj and hasattr(sym_obj, "signals"):
                        has_sev = any(
                            getattr(s.severity, "value", s.severity) == "severe"
                            for s in sym_obj.signals
                        )
                    elif hasattr(t, "symptom_json") and t.symptom_json:
                        try:
                            sym_data = json.loads(t.symptom_json)
                            signals = sym_data.get("signals", [])
                            has_sev = any(s.get("severity") == "severe" for s in signals)
                        except Exception:
                            pass

                    if has_sev:
                        consecutive_symptom_turns += 1

                if consecutive_symptom_turns >= 3:
                    potential_reasons.append((
                        EscalationSeverity.HIGH,
                        EscalationReason(
                            trigger_type=EscalationTriggerType.REPEATED_SIGNALS,
                            code="PERSISTENT_ACUTE_SYMPTOMS",
                            title="Persistent Severe Clinical Symptoms",
                            description="User exhibited severe symptoms across 3 consecutive turns.",
                            trigger_cues=["consecutive_severe_symptoms"],
                            confidence=0.85
                        )
                    ))

        # -------------------------------------------------------------
        # 5. Significant Longitudinal Changes (Baseline / Behavior)
        # -------------------------------------------------------------
        if baseline:
            metric_devs = getattr(baseline, "metric_deviations", {}) or {}
            high_devs = {}
            for feat, z in metric_devs.items():
                val = z.get("z_score", z.get("deviation", 0.0)) if isinstance(z, dict) else (z if isinstance(z, (int, float)) else 0.0)
                try:
                    if abs(float(val)) >= 3.0:
                        high_devs[feat] = float(val)
                except (ValueError, TypeError):
                    pass

            raw_devs = getattr(baseline, "deviations", []) or []
            for d in raw_devs:
                feat = getattr(d, "feature", "metric")
                dev_val = getattr(d, "deviation", 0.0)
                try:
                    if abs(float(dev_val)) >= 3.0:
                        high_devs[feat] = float(dev_val)
                except (ValueError, TypeError):
                    pass

            if high_devs or getattr(baseline, "is_significant_deviation", False):
                cues = [f"{feat}:{z:+.2f}σ" for feat, z in high_devs.items()]
                if not cues and getattr(baseline, "deviating_features", None):
                    cues = baseline.deviating_features

                if high_devs or cues:
                    potential_reasons.append((
                        EscalationSeverity.HIGH,
                        EscalationReason(
                            trigger_type=EscalationTriggerType.LONGITUDINAL_CHANGE,
                            code="LONGITUDINAL_BASELINE_SPIKE",
                            title="Significant Personal Baseline Deviation",
                            description=f"User state deviated >= 3.0σ from established baseline: {', '.join(cues)}.",
                            trigger_cues=cues,
                            confidence=0.90,
                            metadata={"deviations": high_devs}
                        )
                    ))

        if behavior and behavior.behavioral_anomaly_detected:
            # Check affected metrics count
            if len(behavior.metrics) >= 3:
                potential_reasons.append((
                    EscalationSeverity.MEDIUM,
                    EscalationReason(
                        trigger_type=EscalationTriggerType.LONGITUDINAL_CHANGE,
                        code="BEHAVIORAL_ANOMALY_SPIKE",
                        title="Multimodal Passive Sensing Behavioral Anomaly",
                        description=f"Passive sensor data detected significant anomaly across {len(behavior.metrics)} domains.",
                        trigger_cues=[f"anomaly_notes: {behavior.anomaly_notes or 'detected'}"],
                        confidence=0.80
                    )
                ))

        if not potential_reasons:
            return EscalationDecision(should_escalate=False)

        # Select highest severity reason
        potential_reasons.sort(key=lambda item: self.SEVERITY_ORDER.get(item[0], 0), reverse=True)
        top_severity, top_reason = potential_reasons[0]

        # Assemble context snapshot
        snapshot = EscalationContextSnapshot(
            session_id=user_input.session_id,
            user_id=user_input.user_id,
            turn_id=None,
            last_user_message=user_input.text,
            emotion_summary=None,
            risk_summary={"risk_level": risk.risk_level.value, "category": risk.crisis_category.value} if risk else None,
            baseline_summary={"status": getattr(baseline, "status", None) or getattr(baseline, "baseline_status", None)} if baseline else None,
            safety_summary={"action": safety.action.value, "policies": safety.violated_policies} if safety else None
        )

        logger.warning(
            f"Escalation Decision triggered: severity={top_severity.value}, "
            f"type={top_reason.trigger_type.value}, code={top_reason.code}"
        )

        return EscalationDecision(
            should_escalate=True,
            severity=top_severity,
            trigger_type=top_reason.trigger_type,
            reason=top_reason,
            context_snapshot=snapshot
        )
