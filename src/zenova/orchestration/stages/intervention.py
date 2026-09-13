"""Conversational Intervention, Strategy Planning, Generation, and Safety Stage."""
import uuid
from typing import Optional, List, Dict, Any

from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.orchestration.fallbacks import get_safe_fallback_response
from zenova.schemas.standard import (
    SafetyAction,
    GeneratedResponse,
    StrategyResult,
    SupportStrategy,
    SafetyResult
)
from zenova.schemas.rag import RAGQueryResult
from zenova.schemas.orchestration import PipelineStatus
from zenova.escalation.engine import EscalationDecisionEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.intervention")

CRISIS_RESPONSE_TEXT = (
    "I hear how much pain you are experiencing right now, and your safety is the most important thing. "
    "I am an AI wellbeing companion and cannot provide emergency care. "
    "Please connect with trained professionals who can help you immediately:\n"
    "• Call or text 988 (USA & Canada) — Free, confidential, 24/7\n"
    "• Text HOME to 741741 to reach Crisis Text Line\n"
    "• International support: https://findahelpline.com/\n\n"
    "You do not have to carry this alone."
)


class InterventionStage(BasePipelineStage):
    """Executes clinical intervention: crisis bypass routing or strategy+RAG+generation+safety validation."""

    def __init__(self, escalation_engine: Optional[EscalationDecisionEngine] = None):
        self.escalation_engine = escalation_engine or EscalationDecisionEngine()

    @property
    def stage_name(self) -> str:
        return "intervention"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(span_name=self.stage_name) as span:
            strategy_planner = self._safe_get_module(ctx, "strategy")
            response_generator = self._safe_get_module(ctx, "generator")
            safety_gate = self._safe_get_module(ctx, "safety")
            rag_engine = self._safe_get_module(ctx, "rag")

            # --- BRANCH A: CRISIS INTERVENTION BYPASS ---
            if ctx.is_crisis:
                await self._handle_crisis_pathway(ctx, strategy_planner, safety_gate)
                span.output_summary = {
                    "pathway": "crisis_bypass",
                    "escalated_to_human": ctx.escalated_to_human,
                    "escalation_id": ctx.escalation_incident_id
                }
                return

            # --- BRANCH B: NORMAL SUPPORT PATHWAY ---
            await self._handle_normal_pathway(
                ctx,
                strategy_planner=strategy_planner,
                rag_engine=rag_engine,
                response_generator=response_generator,
                safety_gate=safety_gate
            )
            span.output_summary = {
                "pathway": "strategy_controlled_support",
                "strategy": ctx.strategy_res.selected_strategy.value if ctx.strategy_res else None,
                "safety_action": ctx.safety_res.action.value if ctx.safety_res else None,
                "escalated_to_human": ctx.escalated_to_human
            }

    async def _handle_crisis_pathway(
        self,
        ctx: PipelineStageContext,
        strategy_planner: Optional[Any],
        safety_gate: Optional[Any]
    ) -> None:
        """Immediate crisis safety protocol with clinician escalation."""
        ctx.escalated_to_human = True
        incident_id = f"esc_{uuid.uuid4().hex[:12]}"
        ctx.escalation_incident_id = incident_id
        sev_val = "critical" if (ctx.risk_res and ctx.risk_res.risk_level.value == "critical") else "high"

        # Record escalation alert if DB is available
        if ctx.escalation_repo:
            try:
                await ctx.escalation_repo.create_alert(
                    alert_id=incident_id,
                    session_id=ctx.user_input.session_id,
                    user_id=ctx.user_input.user_id,
                    severity=sev_val,
                    trigger_type="crisis_risk",
                    risk_level=ctx.risk_res.risk_level.value if ctx.risk_res else "high",
                    crisis_category=ctx.risk_res.crisis_category.value if ctx.risk_res else "self_harm",
                    trigger_cues=ctx.risk_res.trigger_cues if ctx.risk_res else [],
                    reason_dict={
                        "code": "CRITICAL_RISK_DETECTED" if sev_val == "critical" else "HIGH_RISK_DETECTED",
                        "title": "Crisis Risk Intervention",
                        "description": f"Acute crisis triggered: category={ctx.risk_res.crisis_category.value if ctx.risk_res else 'unknown'}",
                        "trigger_cues": ctx.risk_res.trigger_cues if ctx.risk_res else []
                    },
                    context_summary_dict={
                        "session_id": ctx.user_input.session_id,
                        "user_id": ctx.user_input.user_id,
                        "last_user_message": ctx.user_input.text
                    }
                )
                audit_id = f"aud_{uuid.uuid4().hex[:12]}"
                await ctx.escalation_repo.record_audit(
                    audit_id=audit_id,
                    alert_id=incident_id,
                    action="created",
                    actor_id="system_orchestrator",
                    actor_role="system_admin",
                    previous_status=None,
                    new_status="pending",
                    details_dict={"risk_level": ctx.risk_res.risk_level.value if ctx.risk_res else "high", "severity": sev_val}
                )
            except Exception as e:
                logger.warning(f"Failed to record crisis escalation in database: {e}")

        # Safety Gate Verification on crisis template
        strat_applied = SupportStrategy.OTHERS
        if strategy_planner:
            try:
                st_pred = strategy_planner.predict_strategy(
                    ctx.user_input, ctx.emotion_res, ctx.symptom_res, ctx.context, multimodal_context=ctx.multimodal_ctx
                )
                if st_pred and st_pred.selected_strategy:
                    strat_applied = st_pred.selected_strategy
            except Exception:
                pass

        candidate_obj = GeneratedResponse(
            is_placeholder=True,
            response_text=CRISIS_RESPONSE_TEXT,
            strategy_applied=strat_applied,
            model_name="crisis_intervention_bypass"
        )
        if safety_gate:
            try:
                ctx.safety_res = safety_gate.verify(
                    ctx.user_input,
                    candidate_obj,
                    ctx.risk_res,
                    multimodal_context=ctx.multimodal_ctx
                )
            except Exception as s_err:
                logger.warning(f"Safety verification warning on crisis response: {s_err}")
                ctx.safety_res = SafetyResult(
                    is_safe=True,
                    action=SafetyAction.ALLOW,
                    modified_text=CRISIS_RESPONSE_TEXT,
                    explanation="Default crisis bypass safety allowance"
                )
        else:
            ctx.safety_res = SafetyResult(
                is_safe=True,
                action=SafetyAction.ALLOW,
                modified_text=CRISIS_RESPONSE_TEXT,
                explanation="Default crisis bypass safety allowance"
            )

        # Audit safety log
        if ctx.safety_audit_repo and ctx.safety_res:
            try:
                await ctx.safety_audit_repo.record_audit(
                    audit_id=ctx.safety_res.audit_id or f"aud_{uuid.uuid4().hex[:12]}",
                    session_id=ctx.user_input.session_id,
                    turn_id=ctx.turn_idx,
                    model_version=ctx.safety_res.module_version,
                    action=ctx.safety_res.action.value if hasattr(ctx.safety_res.action, "value") else str(ctx.safety_res.action),
                    is_safe=ctx.safety_res.is_safe,
                    violated_policies=ctx.safety_res.violated_policies,
                    reason_codes=ctx.safety_res.reason_codes,
                    risk_level=ctx.risk_res.risk_level.value if ctx.risk_res else "critical",
                    latency_ms=0.0
                )
            except Exception as aud_err:
                logger.warning(f"Failed to record crisis safety audit: {aud_err}")

        ctx.final_response_text = CRISIS_RESPONSE_TEXT

    async def _handle_normal_pathway(
        self,
        ctx: PipelineStageContext,
        strategy_planner: Optional[Any],
        rag_engine: Optional[Any],
        response_generator: Optional[Any],
        safety_gate: Optional[Any]
    ) -> None:
        """Strategy planning -> Curated RAG -> LLM generation -> Safety verification -> Escalation check."""
        
        # 1. Strategy Planning
        async with ctx.tracer.record_span(
            span_name="strategy_planning",
            module_name="strategy",
            swallow_exception=True,
            is_degraded_on_error=True
        ) as st_span:
            if strategy_planner:
                try:
                    ctx.strategy_res = strategy_planner.predict_strategy(
                        ctx.user_input, ctx.emotion_res, ctx.symptom_res, ctx.context, multimodal_context=ctx.multimodal_ctx
                    )
                    st_span.output_summary = {
                        "strategy": ctx.strategy_res.selected_strategy.value,
                        "confidence": ctx.strategy_res.confidence
                    }
                except Exception as st_err:
                    logger.warning(f"Strategy planner exception: {st_err}")
                    ctx.strategy_res = StrategyResult(
                        selected_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
                        confidence=0.0,
                        probabilities={},
                        is_available=False,
                        model_version="fallback_reflection"
                    )
                    ctx.tracer.mark_degraded("strategy", reason=str(st_err))
            else:
                ctx.strategy_res = StrategyResult(
                    selected_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
                    confidence=0.0,
                    probabilities={},
                    is_available=False,
                    model_version="fallback_reflection"
                )
                ctx.tracer.mark_degraded("strategy", reason="Module not available")

        # 2. Curated RAG Grounding
        if rag_engine:
            async with ctx.tracer.record_span(
                span_name="rag_retrieval",
                module_name="rag",
                swallow_exception=True,
                is_degraded_on_error=True
            ) as rag_span:
                try:
                    ctx.rag_res = rag_engine.query(ctx.user_input.text)
                    if ctx.rag_res and ctx.rag_res.is_confident and ctx.rag_res.formatted_context:
                        ctx.rag_context = ctx.rag_res.formatted_context
                        rag_span.output_summary = {
                            "results_count": len(ctx.rag_res.results),
                            "confidence": ctx.rag_res.highest_score
                        }
                    else:
                        rag_span.output_summary = {"results_count": 0, "status": "insufficient_confidence_or_no_match"}
                except Exception as rag_err:
                    logger.warning(f"RAG query skipped or failed: {rag_err}")
                    ctx.tracer.mark_degraded("rag", reason=str(rag_err))
                    ctx.rag_res = None
                    ctx.rag_context = None

        # 3. Response Generation (LLM or Verified Clinical Template Fallback)
        async with ctx.tracer.record_span(
            span_name="response_generation",
            module_name="generator",
            swallow_exception=True,
            is_degraded_on_error=True
        ) as gen_span:
            if response_generator:
                try:
                    ctx.candidate_gen = response_generator.generate(
                        ctx.user_input,
                        ctx.strategy_res,
                        ctx.context,
                        rag_context=ctx.rag_context,
                        multimodal_context=ctx.multimodal_ctx
                    )
                    gen_span.output_summary = {
                        "model_name": ctx.candidate_gen.model_name,
                        "strategy_applied": ctx.candidate_gen.strategy_applied,
                        "response_length": len(ctx.candidate_gen.response_text)
                    }
                except Exception as gen_err:
                    logger.warning(f"Response generator exception: {gen_err}")
                    strat = ctx.strategy_res.selected_strategy if ctx.strategy_res else SupportStrategy.REFLECTION_OF_FEELINGS
                    ctx.candidate_gen = get_safe_fallback_response(strat, ctx.user_input)
                    ctx.tracer.mark_degraded("generator", reason=str(gen_err))
                    ctx.tracer.set_status(PipelineStatus.SAFE_FALLBACK)
            else:
                strat = ctx.strategy_res.selected_strategy if ctx.strategy_res else SupportStrategy.REFLECTION_OF_FEELINGS
                ctx.candidate_gen = get_safe_fallback_response(strat, ctx.user_input)
                ctx.tracer.mark_degraded("generator", reason="Module not registered")
                ctx.tracer.set_status(PipelineStatus.SAFE_FALLBACK)

        # 4. Independent Safety Gate Verification
        async with ctx.tracer.record_span(
            span_name="safety_verification",
            module_name="safety",
            swallow_exception=True,
            is_degraded_on_error=True
        ) as saf_span:
            if safety_gate:
                try:
                    ctx.safety_res = safety_gate.verify(
                        ctx.user_input,
                        ctx.candidate_gen,
                        ctx.risk_res,
                        multimodal_context=ctx.multimodal_ctx
                    )
                    saf_span.output_summary = {
                        "action": ctx.safety_res.action.value,
                        "is_safe": ctx.safety_res.is_safe,
                        "violated_policies": ctx.safety_res.violated_policies
                    }
                except Exception as s_err:
                    logger.warning(f"Safety gate exception: {s_err}")
                    ctx.safety_res = SafetyResult(
                        is_safe=True,
                        action=SafetyAction.ALLOW,
                        modified_text=ctx.candidate_gen.response_text,
                        explanation="Emergency fallback safety pass"
                    )
                    ctx.tracer.mark_degraded("safety", reason=str(s_err))
            else:
                ctx.safety_res = SafetyResult(
                    is_safe=True,
                    action=SafetyAction.ALLOW,
                    modified_text=ctx.candidate_gen.response_text,
                    explanation="Safety module not registered"
                )
                ctx.tracer.mark_degraded("safety", reason="Module not registered")

        # 5. Record Safety Audit Log
        if ctx.safety_audit_repo and ctx.safety_res:
            try:
                await ctx.safety_audit_repo.record_audit(
                    audit_id=ctx.safety_res.audit_id or f"aud_{uuid.uuid4().hex[:12]}",
                    session_id=ctx.user_input.session_id,
                    turn_id=ctx.turn_idx,
                    model_version=ctx.safety_res.module_version,
                    action=ctx.safety_res.action.value if hasattr(ctx.safety_res.action, "value") else str(ctx.safety_res.action),
                    is_safe=ctx.safety_res.is_safe,
                    violated_policies=ctx.safety_res.violated_policies,
                    reason_codes=ctx.safety_res.reason_codes,
                    risk_level=ctx.risk_res.risk_level.value if ctx.risk_res else "low",
                    latency_ms=0.0
                )
            except Exception as aud_err:
                logger.warning(f"Failed to record safety audit: {aud_err}")

        # 6. Secondary Clinician Escalation Decision
        try:
            ctx.escalation_decision = self.escalation_engine.evaluate(
                user_input=ctx.user_input,
                risk=ctx.risk_res,
                safety=ctx.safety_res,
                baseline=ctx.baseline_res,
                behavior=ctx.behavior_res,
                history=ctx.history,
                multimodal_context=ctx.multimodal_ctx
            )
        except Exception as e:
            logger.warning(f"Escalation evaluation exception: {e}")

        # Check if escalation is required
        should_escalate = False
        if ctx.escalation_decision and ctx.escalation_decision.should_escalate:
            should_escalate = True
        if ctx.safety_res and ctx.safety_res.action == SafetyAction.BLOCK_AND_ESCALATE:
            should_escalate = True
        if ctx.risk_res and ctx.risk_res.is_high_risk:
            should_escalate = True

        if should_escalate:
            ctx.escalated_to_human = True
            ctx.escalation_incident_id = f"esc_{uuid.uuid4().hex[:12]}"
            sev_val = (
                ctx.escalation_decision.severity.value
                if ctx.escalation_decision and ctx.escalation_decision.severity
                else ("critical" if (ctx.risk_res and ctx.risk_res.is_high_risk) else "high")
            )
            trig_type = (
                ctx.escalation_decision.trigger_type.value
                if ctx.escalation_decision and ctx.escalation_decision.trigger_type
                else "safety_gate"
            )

            if ctx.escalation_repo:
                try:
                    reason_dict = ctx.escalation_decision.reason.model_dump() if (ctx.escalation_decision and ctx.escalation_decision.reason) else {
                        "code": "SAFETY_OR_RISK_ESCALATION",
                        "title": "Safety Gate Escalation",
                        "description": f"Safety Action: {ctx.safety_res.action.value if ctx.safety_res else 'unknown'}",
                        "trigger_cues": ctx.safety_res.reason_codes if ctx.safety_res else []
                    }
                    context_dict = ctx.escalation_decision.context_snapshot.model_dump() if (ctx.escalation_decision and ctx.escalation_decision.context_snapshot) else {
                        "session_id": ctx.user_input.session_id,
                        "user_id": ctx.user_input.user_id,
                        "last_user_message": ctx.user_input.text
                    }
                    combined_cues = list(set(
                        (ctx.risk_res.trigger_cues if ctx.risk_res else []) +
                        (ctx.safety_res.reason_codes if ctx.safety_res else [])
                    ))

                    await ctx.escalation_repo.create_alert(
                        alert_id=ctx.escalation_incident_id,
                        session_id=ctx.user_input.session_id,
                        user_id=ctx.user_input.user_id,
                        severity=sev_val,
                        trigger_type=trig_type,
                        risk_level=ctx.risk_res.risk_level.value if ctx.risk_res else "moderate",
                        crisis_category=ctx.risk_res.crisis_category.value if ctx.risk_res else "none",
                        trigger_cues=combined_cues,
                        reason_dict=reason_dict,
                        context_summary_dict=context_dict
                    )
                    audit_id = f"aud_{uuid.uuid4().hex[:12]}"
                    await ctx.escalation_repo.record_audit(
                        audit_id=audit_id,
                        alert_id=ctx.escalation_incident_id,
                        action="created",
                        actor_id="system_orchestrator",
                        actor_role="system_admin",
                        previous_status=None,
                        new_status="pending",
                        details_dict={"severity": sev_val, "trigger_type": trig_type}
                    )
                except Exception as e:
                    logger.warning(f"Failed to record escalation alert: {e}")

        # 7. Select Final Response Text
        if ctx.safety_res and ctx.safety_res.action == SafetyAction.BLOCK_AND_ESCALATE:
            ctx.final_response_text = ctx.safety_res.modified_text or CRISIS_RESPONSE_TEXT
        elif ctx.risk_res and ctx.risk_res.is_high_risk:
            ctx.final_response_text = ctx.safety_res.modified_text if (ctx.safety_res and ctx.safety_res.override_applied) else CRISIS_RESPONSE_TEXT
        elif ctx.safety_res and ctx.safety_res.action == SafetyAction.REVISE:
            logger.info(f"Safety Gate revised candidate response for session {ctx.user_input.session_id}")
            ctx.final_response_text = ctx.safety_res.modified_text or (ctx.candidate_gen.response_text if ctx.candidate_gen else "")
        else:
            ctx.final_response_text = ctx.candidate_gen.response_text if ctx.candidate_gen else ""

    def _safe_get_module(self, ctx: PipelineStageContext, module_name: str) -> Optional[Any]:
        try:
            return ctx.registry.get_module_instance(module_name)
        except Exception:
            return None
