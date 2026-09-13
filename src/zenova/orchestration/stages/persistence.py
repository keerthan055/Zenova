"""Longitudinal State & Execution Trace Persistence Stage."""
from typing import Optional

from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.orchestration.tracing.buffer import get_trace_buffer
from zenova.schemas.standard import SpeakerRole
from zenova.core.logging import get_logger

logger = get_logger("zenova.orchestration.stages.persistence")


class PersistenceStage(BasePipelineStage):
    """Persists conversation turns, multimodal snapshots, safety audits, and execution traces.
    
    Resilience Invariant: If database operations fail or lock out, the generated therapeutic
    response is safely preserved and delivered to the user, with the trace saved to the in-memory buffer.
    """

    @property
    def stage_name(self) -> str:
        return "persistence"

    async def execute(self, ctx: PipelineStageContext) -> None:
        async with ctx.tracer.record_span(span_name=self.stage_name) as span:
            # 1. Record turns to database
            if ctx.turn_repo:
                try:
                    # User turn
                    await ctx.turn_repo.record_turn(
                        session_id=ctx.user_input.session_id,
                        turn_id=ctx.turn_idx,
                        speaker=SpeakerRole.USER.value,
                        content=ctx.user_input.text,
                        emotion_json=ctx.emotion_res.model_dump_json() if ctx.emotion_res else None,
                        symptom_json=ctx.symptom_res.model_dump_json() if ctx.symptom_res else None,
                        risk_json=ctx.risk_res.model_dump_json() if ctx.risk_res else None,
                        behavior_json=ctx.behavior_res.model_dump_json() if ctx.behavior_res else None,
                        voice_json=ctx.voice_res.model_dump_json() if ctx.voice_res else None,
                        context_json=ctx.multimodal_ctx.model_dump_json() if ctx.multimodal_ctx else None
                    )

                    # Assistant response turn
                    await ctx.turn_repo.record_turn(
                        session_id=ctx.user_input.session_id,
                        turn_id=ctx.turn_idx + 1,
                        speaker=SpeakerRole.ASSISTANT.value,
                        content=ctx.final_response_text,
                        strategy_json=ctx.strategy_res.model_dump_json() if ctx.strategy_res else None,
                        safety_json=ctx.safety_res.model_dump_json() if ctx.safety_res else None
                    )
                    span.metadata["turns_persisted"] = True
                except Exception as db_turn_err:
                    logger.warning(f"Database turn persistence failed (non-blocking): {db_turn_err}")
                    ctx.db_persisted = False
                    ctx.tracer.mark_degraded("database", reason=str(db_turn_err))

            span.output_summary = {
                "db_persisted": ctx.db_persisted,
                "turns_persisted": span.metadata.get("turns_persisted", False)
            }

        # 2. Finalize trace after persistence span has completed
        final_trace = ctx.tracer.finalize()
        ctx.metadata["final_trace"] = final_trace

        # 3. Always buffer in-memory for zero-data-loss trace inspection
        get_trace_buffer().add_trace(final_trace)

        # 4. Persist Trace to Database if available
        if ctx.trace_repo and ctx.db_persisted:
            try:
                await ctx.trace_repo.record_trace(final_trace)
            except Exception as db_trace_err:
                logger.warning(f"Database trace persistence failed (non-blocking): {db_trace_err}")
                ctx.db_persisted = False
                ctx.tracer.mark_degraded("database", reason=str(db_trace_err))
