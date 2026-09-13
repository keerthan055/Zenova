"""Base abstractions and shared context for ZENOVA pipeline stages."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from zenova.schemas.standard import (
    UserInput,
    ConversationTurn,
    ConversationContext,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    GeneratedResponse,
    SafetyResult,
    MultimodalContext
)
from zenova.schemas.rag import RAGQueryResult
from zenova.schemas.fusion import FusedMultimodalState
from zenova.orchestration.tracing.tracer import PipelineTracer
from zenova.models.registry import ModelRegistry


@dataclass
class PipelineStageContext:
    """Shared execution context passed across all modular pipeline stages."""
    user_input: UserInput
    context: Optional[ConversationContext]
    tracer: PipelineTracer
    registry: ModelRegistry
    
    # DB repositories (may be None if running in pure in-memory or detached mode)
    session_repo: Optional[Any] = None
    turn_repo: Optional[Any] = None
    escalation_repo: Optional[Any] = None
    context_repo: Optional[Any] = None
    safety_audit_repo: Optional[Any] = None
    trace_repo: Optional[Any] = None

    # Analytical outputs
    voice_res: Optional[VoiceResult] = None
    behavior_res: Optional[BehavioralResult] = None
    emotion_res: Optional[EmotionResult] = None
    symptom_res: Optional[SymptomResult] = None
    risk_res: Optional[RiskResult] = None
    baseline_res: Optional[BaselineResult] = None
    fused_state: Optional[FusedMultimodalState] = None
    multimodal_ctx: Optional[MultimodalContext] = None

    # Conversation history & turn index
    history: List[ConversationTurn] = field(default_factory=list)
    turn_idx: int = 1

    # Triage & Intervention outputs
    is_crisis: bool = False
    strategy_res: Optional[StrategyResult] = None
    rag_res: Optional[RAGQueryResult] = None
    rag_context: Optional[List[str]] = None
    candidate_gen: Optional[GeneratedResponse] = None
    safety_res: Optional[SafetyResult] = None
    escalation_decision: Optional[Any] = None
    escalated_to_human: bool = False
    escalation_incident_id: Optional[str] = None
    final_response_text: str = ""

    # Metadata & flags
    db_persisted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePipelineStage(ABC):
    """Abstract contract for an isolated, instrumented pipeline execution stage."""

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Name of the pipeline stage for tracing and logging."""
        pass

    @abstractmethod
    async def execute(self, ctx: PipelineStageContext) -> None:
        """Execute the stage logic, mutating the shared PipelineStageContext."""
        pass
