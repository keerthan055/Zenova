"""Central Orchestration Engine REST API Routes & Trace Inspection."""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Header, Query

from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType
from zenova.schemas.escalation import UserRole
from zenova.schemas.orchestration import (
    OrchestrationResult,
    OrchestratorTurnRequest,
    PipelineTrace,
    TraceListResponse,
    ComponentHealth,
    OrchestratorHealthResponse
)
from zenova.orchestration.tracing.buffer import get_trace_buffer
from zenova.db.session import get_db_session
from zenova.db.repositories import TraceRepository
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.orchestrator")

router = APIRouter(prefix="/api/v1/orchestrator", tags=["Orchestrator"])
orchestrator = ZenovaOrchestrator()

AUTHORIZED_TRACE_ROLES = {
    UserRole.CLINICIAN,
    UserRole.TRIAGE_SUPERVISOR,
    UserRole.SYSTEM_ADMIN,
    UserRole.AUDITOR
}


def get_authorized_user(
    x_user_role: str = Header(default="clinician", alias="X-User-Role"),
    x_user_id: str = Header(default="clinician_admin", alias="X-User-ID")
) -> tuple[UserRole, str]:
    """Validate caller RBAC role and enforce trace inspection authorization."""
    try:
        role = UserRole(x_user_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail=f"Invalid or unrecognized user role: '{x_user_role}'")

    if role not in AUTHORIZED_TRACE_ROLES:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role.value}' is not authorized to inspect internal execution traces."
        )
    return role, x_user_id


@router.post("/process", response_model=OrchestrationResult)
async def process_turn(request: OrchestratorTurnRequest) -> OrchestrationResult:
    """High-level unified API for end-to-end turn processing across all stages."""
    try:
        # Construct UserInput
        meta = dict(request.metadata or {})
        if request.audio_base64:
            meta["audio_base64"] = request.audio_base64
            modality = ModalityType.VOICE
        else:
            modality = ModalityType.TEXT

        user_input = UserInput(
            session_id=request.session_id,
            user_id=request.user_id,
            text=request.text or "...",
            modality=modality,
            audio_features=None,
            metadata=meta
        )

        return await orchestrator.engine.process_turn_typed(user_input)
    except Exception as e:
        logger.error(f"Orchestrator process failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orchestration execution failed: {str(e)}")


@router.get("/traces/{trace_id}", response_model=PipelineTrace)
async def get_execution_trace(
    trace_id: str,
    user_context: tuple[UserRole, str] = Depends(get_authorized_user)
) -> PipelineTrace:
    """Inspect full internal execution trace for authorized clinicians or developers."""
    # Try DB repository first
    try:
        async with get_db_session() as db:
            repo = TraceRepository(db)
            trace = await repo.get_trace(trace_id)
            if trace:
                return trace
    except Exception as db_err:
        logger.warning(f"Database query for trace {trace_id} failed: {db_err}")

    # Fall back to in-memory trace buffer
    trace = get_trace_buffer().get_trace(trace_id)
    if trace:
        return trace

    raise HTTPException(status_code=404, detail=f"Execution trace '{trace_id}' not found.")


@router.get("/traces", response_model=TraceListResponse)
async def list_execution_traces(
    session_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    user_context: tuple[UserRole, str] = Depends(get_authorized_user)
) -> TraceListResponse:
    """List execution traces filtered by session, user, or status for authorized clinicians."""
    traces: List[PipelineTrace] = []

    try:
        async with get_db_session() as db:
            repo = TraceRepository(db)
            traces = await repo.list_traces(session_id=session_id, user_id=user_id, status=status, limit=limit)
    except Exception as db_err:
        logger.warning(f"Database query for traces failed: {db_err}")

    # If DB returned nothing or failed, check in-memory buffer
    if not traces:
        traces = get_trace_buffer().list_traces(session_id=session_id, user_id=user_id, status=status, limit=limit)

    return TraceListResponse(
        traces=traces,
        total=len(traces)
    )


@router.get("/health", response_model=OrchestratorHealthResponse)
async def get_orchestrator_health() -> OrchestratorHealthResponse:
    """Component-level health check across all analytical models, planners, and safety stages."""
    components_to_check = [
        "emotion", "symptom", "risk", "baseline", "behavior",
        "voice", "context", "strategy", "generator", "safety", "rag", "fusion"
    ]
    comp_list: List[ComponentHealth] = []
    avail_count = 0
    degraded_count = 0

    for name in components_to_check:
        try:
            inst = orchestrator.get_module(name)
            is_avail = inst is not None
            if is_avail:
                avail_count += 1
                comp_list.append(ComponentHealth(
                    name=name,
                    available=True,
                    version=getattr(inst, "version", "1.0.0"),
                    degraded=False
                ))
            else:
                degraded_count += 1
                comp_list.append(ComponentHealth(
                    name=name,
                    available=False,
                    degraded=True,
                    error="Module returned None"
                ))
        except Exception as err:
            degraded_count += 1
            comp_list.append(ComponentHealth(
                name=name,
                available=False,
                degraded=True,
                error=str(err)
            ))

    status = "nominal" if degraded_count == 0 else ("degraded" if avail_count > 0 else "unhealthy")

    return OrchestratorHealthResponse(
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=comp_list,
        total_components=len(components_to_check),
        available_components=avail_count,
        degraded_components=degraded_count
    )
