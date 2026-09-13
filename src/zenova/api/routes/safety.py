"""API routes for Response Safety Gate verification, policies, and privacy-sanitized audit logs."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zenova.schemas.safety import SafetyGateResult, SafetyPolicy, SafetyAction
from zenova.safety.gate import ResponseSafetyGate
from zenova.db.session import get_db_session
from zenova.db.repositories import SafetyAuditRepository
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.safety")

router = APIRouter(prefix="/api/v1/safety", tags=["Response Safety Gate"])

# Shared gate instance
_gate: Optional[ResponseSafetyGate] = None


def get_safety_gate() -> ResponseSafetyGate:
    global _gate
    if _gate is None:
        _gate = ResponseSafetyGate()
    return _gate


class VerifyResponseRequest(BaseModel):
    candidate_text: str = Field(..., min_length=1, description="LLM candidate response text to evaluate")
    user_text: str = Field(default="", description="User utterance context")
    risk_level: str = Field(default="low", description="Assessed risk level (low, moderate, high, critical)")


@router.post("/verify", response_model=SafetyGateResult)
async def verify_candidate_response(req: VerifyResponseRequest) -> SafetyGateResult:
    """Verify an LLM candidate response against all 12 clinical, ethical, and legal safety policies."""
    try:
        gate = get_safety_gate()
        result = gate.verify_raw(
            candidate_text=req.candidate_text,
            user_text=req.user_text,
            risk_level=req.risk_level
        )
        return result
    except Exception as e:
        logger.error(f"Safety verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Safety verification failed: {str(e)}")


@router.get("/policies")
async def list_policies() -> Dict[str, Any]:
    """List all 12 configurable safety policies and their enforcement status."""
    gate = get_safety_gate()
    policies_meta = [
        {
            "id": p.value,
            "name": p.name,
            "enabled": gate.evaluator.enabled_policies.get(p.value, True),
            "description": f"Safety boundary check for {p.value.replace('_', ' ')}."
        }
        for p in SafetyPolicy
    ]
    return {
        "count": len(policies_meta),
        "policies": policies_meta
    }


@router.get("/audit")
async def get_safety_audit_logs(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    """Retrieve privacy-preserving safety audit log records (raw user conversation content omitted)."""
    # Try fetching from DB if available, fallback to in-memory audit buffer
    try:
        async with get_db_session() as db:
            repo = SafetyAuditRepository(db)
            db_entries = await repo.get_recent_audits(limit=limit)
            if db_entries:
                return {
                    "count": len(db_entries),
                    "source": "database",
                    "audits": [
                        {
                            "audit_id": e.audit_id,
                            "session_id": e.session_id,
                            "turn_id": e.turn_id,
                            "model_version": e.model_version,
                            "action": e.action,
                            "is_safe": bool(e.is_safe),
                            "violated_policies": eval(e.violated_policies_json) if e.violated_policies_json else [],
                            "reason_codes": eval(e.reason_codes_json) if e.reason_codes_json else [],
                            "risk_level": e.risk_level,
                            "latency_ms": e.latency_ms,
                            "created_at": e.created_at.isoformat() if e.created_at else None
                        }
                        for e in db_entries
                    ]
                }
    except Exception as db_err:
        logger.warning(f"Could not read audit logs from DB: {db_err}; falling back to memory buffer.")

    # In-memory buffer fallback
    mem_audits = ResponseSafetyGate.get_recent_audits(limit=limit)
    return {
        "count": len(mem_audits),
        "source": "memory_buffer",
        "audits": mem_audits
    }
