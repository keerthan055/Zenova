"""Central ZENOVA End-to-End Orchestration Engine."""
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from zenova.core.logging import get_logger
from zenova.core.config import get_system_config
from zenova.models.registry import ModelRegistry
from zenova.schemas.standard import (
    UserInput,
    ConversationContext
)
from zenova.schemas.orchestration import (
    OrchestrationResult,
    PipelineTrace,
    PipelineStatus
)
from zenova.orchestration.stages.base import BasePipelineStage, PipelineStageContext
from zenova.orchestration.stages.input_processing import InputProcessingStage
from zenova.orchestration.stages.analytical_layer import AnalyticalLayerStage
from zenova.orchestration.stages.context_engine import ContextAggregationStage
from zenova.orchestration.stages.triage_decision import TriageDecisionStage
from zenova.orchestration.stages.intervention import InterventionStage
from zenova.orchestration.stages.persistence import PersistenceStage
from zenova.orchestration.tracing.tracer import PipelineTracer
from zenova.orchestration.tracing.buffer import get_trace_buffer

from zenova.db.session import get_db_session
from zenova.db.repositories import (
    SessionRepository,
    TurnRepository,
    EscalationRepository,
    ContextRepository,
    SafetyAuditRepository,
    TraceRepository
)

logger = get_logger("zenova.orchestration.engine")


class ZenovaOrchestrationEngine:
    """Central End-to-End Orchestrator coordinating modular pipeline stages.
    
    Architecture:
        User Input
            ↓
        Input Processing
            ↓
        Analytical Layer (Emotion, Symptoms, Risk, Voice, Behavior, Baseline, Fusion)
            ↓
        Context Aggregation
            ↓
        Triage & Risk Decision
            ├─ IF High / Critical: Crisis Safety Bypass + Human Escalation
            └─ ELSE: Strategy Planning -> RAG -> LLM Generation -> Safety Gate
            ↓
        Persistence & Execution Tracing
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        stages: Optional[List[BasePipelineStage]] = None
    ):
        self.config = get_system_config()
        self.registry = model_registry or ModelRegistry()
        self.stages = stages if stages is not None else [
            InputProcessingStage(),
            AnalyticalLayerStage(),
            ContextAggregationStage(),
            TriageDecisionStage(),
            InterventionStage(),
            PersistenceStage()
        ]

    def get_module(self, module_name: str) -> Any:
        """Preserve individual module API queryability."""
        return self.registry.get_module_instance(module_name)

    async def process_turn(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """Process an inbound conversational turn end-to-end across modular stages."""
        start_mono = time.time()
        tracer = PipelineTracer(
            session_id=user_input.session_id,
            user_id=user_input.user_id
        )

        logger.info(
            f"Orchestration engine processing turn session={user_input.session_id}, "
            f"user={user_input.user_id}, trace={tracer.trace_id}"
        )

        db_available = True
        try:
            async with get_db_session() as db:
                session_repo = SessionRepository(db)
                turn_repo = TurnRepository(db)
                escalation_repo = EscalationRepository(db)
                context_repo = ContextRepository(db)
                safety_audit_repo = SafetyAuditRepository(db)
                trace_repo = TraceRepository(db)

                # Ensure session exists
                try:
                    await session_repo.get_or_create(user_input.session_id, user_input.user_id)
                except Exception as s_err:
                    logger.warning(f"Database session get_or_create failed: {s_err}")
                    db_available = False

                stage_ctx = PipelineStageContext(
                    user_input=user_input,
                    context=context,
                    tracer=tracer,
                    registry=self.registry,
                    session_repo=session_repo if db_available else None,
                    turn_repo=turn_repo if db_available else None,
                    escalation_repo=escalation_repo if db_available else None,
                    context_repo=context_repo if db_available else None,
                    safety_audit_repo=safety_audit_repo if db_available else None,
                    trace_repo=trace_repo if db_available else None,
                    db_persisted=db_available
                )

                # Execute modular pipeline stages sequentially
                for stage in self.stages:
                    await stage.execute(stage_ctx)

        except Exception as db_err:
            logger.warning(f"Database session unavailable or crashed: {db_err}. Executing with in-memory persistence fallback.")
            tracer.mark_degraded("database", reason=str(db_err))
            stage_ctx = PipelineStageContext(
                user_input=user_input,
                context=context,
                tracer=tracer,
                registry=self.registry,
                session_repo=None,
                turn_repo=None,
                escalation_repo=None,
                context_repo=None,
                safety_audit_repo=None,
                trace_repo=None,
                db_persisted=False
            )
            for stage in self.stages:
                await stage.execute(stage_ctx)

        total_latency_ms = (time.time() - start_mono) * 1000.0
        final_trace: PipelineTrace = stage_ctx.metadata.get("final_trace") or tracer.finalize()

        # Format backward-compatible response dictionary
        return {
            "session_id": stage_ctx.user_input.session_id,
            "turn_id": stage_ctx.turn_idx,
            "user_input": stage_ctx.user_input.text,
            "response": stage_ctx.final_response_text,
            "escalated_to_human": stage_ctx.escalated_to_human,
            "escalation_id": stage_ctx.escalation_incident_id,
            "escalation_event_id": stage_ctx.escalation_incident_id,
            "emotion": stage_ctx.emotion_res.model_dump() if stage_ctx.emotion_res else None,
            "symptoms": stage_ctx.symptom_res.model_dump() if stage_ctx.symptom_res else None,
            "risk": stage_ctx.risk_res.model_dump() if stage_ctx.risk_res else None,
            "baseline": stage_ctx.baseline_res.model_dump() if stage_ctx.baseline_res else None,
            "behavior": stage_ctx.behavior_res.model_dump() if stage_ctx.behavior_res else None,
            "voice": stage_ctx.voice_res.model_dump() if stage_ctx.voice_res else None,
            "fused_state": stage_ctx.fused_state.model_dump() if stage_ctx.fused_state else None,
            "context": stage_ctx.multimodal_ctx.model_dump() if stage_ctx.multimodal_ctx else None,
            "strategy": stage_ctx.strategy_res.model_dump() if stage_ctx.strategy_res else None,
            "rag": stage_ctx.rag_res.model_dump() if stage_ctx.rag_res else None,
            "safety": stage_ctx.safety_res.model_dump() if stage_ctx.safety_res else None,
            "latency_ms": total_latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Step 17 Execution Tracing fields
            "trace_id": final_trace.trace_id,
            "pipeline_status": final_trace.status.value if hasattr(final_trace.status, "value") else str(final_trace.status),
            "degraded_modules": list(final_trace.degraded_modules),
            "db_persisted": stage_ctx.db_persisted,
            "trace": final_trace.model_dump()
        }

    async def process_turn_typed(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None
    ) -> OrchestrationResult:
        """Process turn returning a strongly typed OrchestrationResult schema."""
        raw_dict = await self.process_turn(user_input, context)
        trace_dict = raw_dict.get("trace")
        trace_obj = PipelineTrace(**trace_dict) if trace_dict else None
        
        return OrchestrationResult(
            session_id=raw_dict["session_id"],
            turn_id=raw_dict["turn_id"],
            user_input=raw_dict["user_input"],
            response=raw_dict["response"],
            escalated_to_human=raw_dict["escalated_to_human"],
            escalation_id=raw_dict["escalation_id"],
            escalation_event_id=raw_dict["escalation_event_id"],
            emotion=raw_dict["emotion"],
            symptoms=raw_dict["symptoms"],
            risk=raw_dict["risk"],
            baseline=raw_dict["baseline"],
            behavior=raw_dict["behavior"],
            voice=raw_dict["voice"],
            fused_state=raw_dict["fused_state"],
            context=raw_dict["context"],
            strategy=raw_dict["strategy"],
            rag=raw_dict["rag"],
            safety=raw_dict["safety"],
            latency_ms=raw_dict["latency_ms"],
            timestamp=raw_dict["timestamp"],
            trace_id=raw_dict.get("trace_id"),
            pipeline_status=PipelineStatus(raw_dict.get("pipeline_status", "nominal")),
            degraded_modules=raw_dict.get("degraded_modules", []),
            trace=trace_obj,
            db_persisted=raw_dict.get("db_persisted", True)
        )
