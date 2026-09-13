"""Clinical Triage & Risk Decision Stage."""
from typing import Optional
from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.schemas.orchestration import PipelineStatus
from zenova.escalation.engine import EscalationDecisionEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.triage_decision")


class TriageDecisionStage(BasePipelineStage):
    """Evaluates acute risk, personal baseline deviation, and clinician escalation rules."""

    def __init__(self, escalation_engine: Optional[EscalationDecisionEngine] = None):
        self.escalation_engine = escalation_engine or EscalationDecisionEngine()

    @property
    def stage_name(self) -> str:
        return "triage_decision"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(span_name=self.stage_name) as span:
            is_high_risk = ctx.risk_res.is_high_risk if ctx.risk_res else False
            ctx.is_crisis = is_high_risk

            # Evaluate preliminary escalation rules
            try:
                ctx.escalation_decision = self.escalation_engine.evaluate(
                    user_input=ctx.user_input,
                    risk=ctx.risk_res,
                    safety=None,  # Not generated yet
                    baseline=ctx.baseline_res,
                    behavior=ctx.behavior_res,
                    history=ctx.history,
                    multimodal_context=ctx.multimodal_ctx
                )
            except Exception as e:
                logger.warning(f"Preliminary escalation evaluation exception: {e}")
                ctx.tracer.mark_degraded("escalation", reason=str(e))

            if ctx.is_crisis:
                ctx.tracer.set_status(PipelineStatus.CRISIS_BYPASS)
                logger.warning(
                    f"CRISIS TRIAGE ACTIVATED: session={ctx.user_input.session_id}, "
                    f"risk_level={ctx.risk_res.risk_level.value if ctx.risk_res else 'unknown'}"
                )

            span.output_summary = {
                "is_crisis": ctx.is_crisis,
                "risk_level": ctx.risk_res.risk_level.value if ctx.risk_res else None,
                "should_escalate_preliminary": ctx.escalation_decision.should_escalate if ctx.escalation_decision else False
            }
