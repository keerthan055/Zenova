"""Multimodal Context Aggregation Stage."""
from typing import Optional

from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.schemas.standard import DialogStage, MultimodalContext
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.context_engine")


class ContextAggregationStage(BasePipelineStage):
    """Aggregates analytical and acoustic signals into a normalized, immutable MultimodalContext."""

    @property
    def stage_name(self) -> str:
        return "context_aggregation"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(
            span_name=self.stage_name,
            module_name="context",
            swallow_exception=True,
            is_degraded_on_error=True
        ) as span:
            context_engine = None
            try:
                context_engine = ctx.registry.get_module_instance("context")
            except Exception as ce_err:
                logger.warning(f"Context engine not registered: {ce_err}")

            if context_engine:
                try:
                    fused_dict = (
                        {"fused_multimodal_state": ctx.fused_state.model_dump()}
                        if ctx.fused_state else None
                    )
                    ctx.multimodal_ctx = context_engine.build_context(
                        user_input=ctx.user_input,
                        history=ctx.history,
                        emotion=ctx.emotion_res,
                        symptoms=ctx.symptom_res,
                        risk=ctx.risk_res,
                        baseline=ctx.baseline_res,
                        behavior=ctx.behavior_res,
                        voice=ctx.voice_res,
                        dialog_stage=ctx.context.active_stage if ctx.context else DialogStage.EXPLORATION,
                        is_crisis_bypass=ctx.risk_res.is_high_risk if ctx.risk_res else False,
                        metadata=fused_dict
                    )
                    span.output_summary = {
                        "context_id": ctx.multimodal_ctx.metadata.context_id,
                        "context_hash": ctx.multimodal_ctx.metadata.context_hash
                    }

                    # Persist snapshot if DB is available
                    if ctx.context_repo and ctx.multimodal_ctx:
                        try:
                            await ctx.context_repo.record_snapshot(
                                context_id=ctx.multimodal_ctx.metadata.context_id,
                                session_id=ctx.user_input.session_id,
                                turn_id=ctx.turn_idx,
                                context_hash=ctx.multimodal_ctx.metadata.context_hash,
                                context_json=ctx.multimodal_ctx.model_dump_json(),
                                privacy_level=ctx.multimodal_ctx.metadata.privacy.get("privacy_level", "STANDARD")
                            )
                        except Exception as snap_err:
                            logger.warning(f"Context snapshot recording warning: {snap_err}")
                except Exception as ctx_err:
                    logger.warning(f"Context engine aggregation warning: {ctx_err}")
                    ctx.tracer.mark_degraded("context", reason=str(ctx_err))
            else:
                ctx.tracer.mark_degraded("context", reason="Module not available")
