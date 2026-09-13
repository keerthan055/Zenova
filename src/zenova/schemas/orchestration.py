"""ZENOVA End-to-End Orchestration & Execution Tracing Schemas."""
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    """Execution status of an individual pipeline span."""
    SUCCESS = "success"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    ERROR = "error"


class PipelineStatus(str, Enum):
    """Overall status of the conversational turn execution."""
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRISIS_BYPASS = "crisis_bypass"
    SAFE_FALLBACK = "safe_fallback"
    ERROR = "error"


class TraceSpan(BaseModel):
    """Detailed telemetry record for an individual stage or component in the pipeline."""
    span_id: str
    span_name: str
    status: SpanStatus = SpanStatus.SUCCESS
    start_time: str
    duration_ms: float = 0.0
    module_name: Optional[str] = None
    module_version: Optional[str] = None
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    """Immutable execution trace of a conversational turn across all pipeline stages."""
    trace_id: str
    session_id: str
    turn_id: int
    user_id: str
    status: PipelineStatus = PipelineStatus.NOMINAL
    total_duration_ms: float = 0.0
    spans: List[TraceSpan] = Field(default_factory=list)
    degraded_modules: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestrationResult(BaseModel):
    """Comprehensive typed result from end-to-end turn processing."""
    session_id: str
    turn_id: int
    user_input: str
    response: str
    escalated_to_human: bool = False
    escalation_id: Optional[str] = None
    escalation_event_id: Optional[str] = None
    emotion: Optional[Dict[str, Any]] = None
    symptoms: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    baseline: Optional[Dict[str, Any]] = None
    behavior: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None
    fused_state: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    strategy: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace_id: Optional[str] = None
    pipeline_status: PipelineStatus = PipelineStatus.NOMINAL
    degraded_modules: List[str] = Field(default_factory=list)
    trace: Optional[PipelineTrace] = None
    db_persisted: bool = True


class OrchestratorTurnRequest(BaseModel):
    """Request payload for the high-level orchestrator process API."""
    session_id: str
    user_id: str
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    passive_behavior_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TraceListResponse(BaseModel):
    """Response payload for listing execution traces."""
    traces: List[PipelineTrace]
    total: int


class ComponentHealth(BaseModel):
    """Health check details for an individual pipeline component."""
    name: str
    available: bool
    version: Optional[str] = None
    degraded: bool = False
    error: Optional[str] = None


class OrchestratorHealthResponse(BaseModel):
    """Health status across all orchestrator modules and stages."""
    status: str
    timestamp: str
    components: List[ComponentHealth]
    total_components: int
    available_components: int
    degraded_components: int
