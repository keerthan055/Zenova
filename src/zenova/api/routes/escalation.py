"""Clinician crisis escalation and alert lifecycle endpoints with Role-Based Access Control."""
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Header, Query, Depends

from zenova.schemas.escalation import (
    EscalationSeverity,
    EscalationStatus,
    EscalationTriggerType,
    EscalationActionType,
    UserRole,
    CreateAlertRequest,
    AcknowledgeAlertRequest,
    UpdateStatusRequest,
    ResolveAlertRequest,
    AlertListResponse
)
from zenova.escalation.rbac import AccessControlManager
from zenova.escalation.rules import ClinicianRuleEngine
from zenova.db.session import get_db_session
from zenova.db.repositories import EscalationRepository
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.escalation")

router = APIRouter(prefix="/api/v1/escalations", tags=["Escalations & Clinician Dashboard"])

_rule_engine: Optional[ClinicianRuleEngine] = None


def get_rule_engine() -> ClinicianRuleEngine:
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = ClinicianRuleEngine()
    return _rule_engine


def get_current_user_context(
    x_user_role: str = Header(default="clinician", alias="X-User-Role"),
    x_user_id: str = Header(default="clinician-admin-1", alias="X-User-ID")
) -> tuple[UserRole, str]:
    """Extract and validate caller RBAC role and ID."""
    try:
        role = UserRole(x_user_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail=f"Invalid or unrecognized user role: '{x_user_role}'")
    return role, x_user_id


@router.get("/rules")
async def list_configured_clinician_rules(
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
) -> Dict[str, Any]:
    """Inspect documented clinician rules and trigger configurations."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_view_queue"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to inspect escalation rules.")

    engine = get_rule_engine()
    return {
        "count": len(engine.rules),
        "rules": engine.rules
    }


@router.get("", response_model=AlertListResponse)
async def list_escalations(
    status: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    clinician_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
):
    """List crisis escalation alert tickets with role-based filtering and data masking."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_view_queue"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to view the clinician escalation queue.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        events = await repo.list_alerts(
            status=status,
            severity=severity,
            clinician_id=clinician_id,
            limit=limit,
            offset=offset
        )

        formatted = []
        for e in events:
            raw_data = {
                "alert_id": e.event_id,
                "event_id": e.event_id,  # Backwards compatibility
                "session_id": e.session_id,
                "user_id": e.user_id,
                "severity": e.severity or "critical",
                "trigger_type": e.trigger_type or "crisis_risk",
                "risk_level": e.risk_level,
                "crisis_category": e.crisis_category,
                "trigger_cues": json.loads(e.trigger_cues_json) if e.trigger_cues_json else [],
                "reason": json.loads(e.reason_json) if e.reason_json else None,
                "context_snapshot": json.loads(e.context_summary_json) if e.context_summary_json else None,
                "status": e.status,
                "assigned_clinician_id": e.assigned_clinician_id,
                "acknowledged_by": e.acknowledged_by,
                "acknowledged_at": e.acknowledged_at.isoformat() if e.acknowledged_at else None,
                "action_taken": e.action_taken,
                "resolution_notes": e.resolution_notes,
                "resolved_by": e.resolved_by,
                "resolved_at": e.resolved_at.isoformat() if e.resolved_at else None,
                "notes": e.notes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "updated_at": e.updated_at.isoformat() if e.updated_at else None
            }
            # Apply RBAC data sanitization
            sanitized = AccessControlManager.sanitize_alert_data(raw_data, role=role)
            formatted.append(sanitized)

        return AlertListResponse(count=len(formatted), alerts=formatted)


@router.get("/{alert_id}")
async def get_escalation_alert(
    alert_id: str,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
) -> Dict[str, Any]:
    """Retrieve detailed clinical alert with context snapshot (masked for non-clinicians)."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_view_queue"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to view escalation alerts.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        e = await repo.get_alert(alert_id)
        if not e:
            raise HTTPException(status_code=404, detail=f"Escalation alert '{alert_id}' not found.")

        raw_data = {
            "alert_id": e.event_id,
            "event_id": e.event_id,
            "session_id": e.session_id,
            "user_id": e.user_id,
            "severity": e.severity or "critical",
            "trigger_type": e.trigger_type or "crisis_risk",
            "risk_level": e.risk_level,
            "crisis_category": e.crisis_category,
            "trigger_cues": json.loads(e.trigger_cues_json) if e.trigger_cues_json else [],
            "reason": json.loads(e.reason_json) if e.reason_json else None,
            "context_snapshot": json.loads(e.context_summary_json) if e.context_summary_json else None,
            "status": e.status,
            "assigned_clinician_id": e.assigned_clinician_id,
            "acknowledged_by": e.acknowledged_by,
            "acknowledged_at": e.acknowledged_at.isoformat() if e.acknowledged_at else None,
            "action_taken": e.action_taken,
            "resolution_notes": e.resolution_notes,
            "resolved_by": e.resolved_by,
            "resolved_at": e.resolved_at.isoformat() if e.resolved_at else None,
            "notes": e.notes,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "updated_at": e.updated_at.isoformat() if e.updated_at else None
        }
        return AccessControlManager.sanitize_alert_data(raw_data, role=role)


@router.post("/trigger")
async def trigger_escalation_alert(
    req: CreateAlertRequest,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
) -> Dict[str, Any]:
    """Programmatically trigger and record a new clinical alert."""
    role, user_id = user_ctx
    alert_id = f"alt_{uuid.uuid4().hex[:12]}"
    audit_id = f"aud_{uuid.uuid4().hex[:12]}"

    reason_dict = {
        "code": req.reason_code,
        "title": req.reason_title,
        "description": req.reason_description,
        "trigger_cues": req.trigger_cues,
        "metadata": req.metadata
    }
    context_dict = {
        "session_id": req.session_id,
        "user_id": req.user_id,
        "last_user_message": req.user_message or ""
    }

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        alert = await repo.create_alert(
            alert_id=alert_id,
            session_id=req.session_id,
            user_id=req.user_id,
            severity=req.severity.value,
            trigger_type=req.trigger_type.value,
            risk_level="high" if req.severity in (EscalationSeverity.HIGH, EscalationSeverity.CRITICAL) else "moderate",
            crisis_category=req.reason_code.lower(),
            trigger_cues=req.trigger_cues,
            reason_dict=reason_dict,
            context_summary_dict=context_dict
        )

        # Record creation audit log
        await repo.record_audit(
            audit_id=audit_id,
            alert_id=alert_id,
            action="created",
            actor_id=user_id,
            actor_role=role.value,
            previous_status=None,
            new_status="pending",
            details_dict={"severity": req.severity.value, "reason_code": req.reason_code}
        )

        logger.info(f"Escalation alert triggered via API: {alert_id} (severity={req.severity.value})")
        return {
            "alert_id": alert.event_id,
            "status": alert.status,
            "severity": alert.severity,
            "created_at": alert.created_at.isoformat()
        }


@router.post("/{alert_id}/acknowledge")
async def acknowledge_escalation(
    alert_id: str,
    req: AcknowledgeAlertRequest,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
):
    """Mark an escalation alert as claimed and acknowledged by a clinician."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_acknowledge"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to acknowledge alerts.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        alert = await repo.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail=f"Escalation alert '{alert_id}' not found.")

        prev_status = alert.status
        updated = await repo.acknowledge_alert(alert_id, clinician_id=req.clinician_id, notes=req.notes)

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        await repo.record_audit(
            audit_id=audit_id,
            alert_id=alert_id,
            action="acknowledged",
            actor_id=req.clinician_id,
            actor_role=role.value,
            previous_status=prev_status,
            new_status="acknowledged",
            details_dict={"notes": req.notes}
        )

        return {
            "alert_id": updated.event_id,
            "event_id": updated.event_id,
            "status": updated.status,
            "assigned_clinician_id": updated.assigned_clinician_id,
            "acknowledged_by": updated.acknowledged_by,
            "acknowledged_at": updated.acknowledged_at.isoformat() if updated.acknowledged_at else None,
            "notes": updated.notes
        }


@router.post("/{alert_id}/status")
async def update_alert_status(
    alert_id: str,
    req: UpdateStatusRequest,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
):
    """Transition alert lifecycle state (e.g. in_review or dismissed)."""
    role, user_id = user_ctx
    if not (AccessControlManager.check_permission(role, "can_acknowledge") or AccessControlManager.check_permission(role, "can_resolve")):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to transition alert status.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        alert = await repo.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail=f"Escalation alert '{alert_id}' not found.")

        prev_status = alert.status
        updated = await repo.update_status(alert_id, new_status=req.status.value, clinician_id=req.clinician_id, notes=req.notes)

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        await repo.record_audit(
            audit_id=audit_id,
            alert_id=alert_id,
            action="status_changed",
            actor_id=req.clinician_id,
            actor_role=role.value,
            previous_status=prev_status,
            new_status=req.status.value,
            details_dict={"notes": req.notes}
        )

        return {
            "alert_id": updated.event_id,
            "status": updated.status,
            "updated_at": updated.updated_at.isoformat() if updated.updated_at else None
        }


@router.post("/{alert_id}/resolve")
async def resolve_escalation(
    alert_id: str,
    req: ResolveAlertRequest,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
):
    """Resolve an alert with mandatory action taken and clinical justification notes."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_resolve"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to resolve escalation alerts.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        alert = await repo.get_alert(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail=f"Escalation alert '{alert_id}' not found.")

        prev_status = alert.status
        updated = await repo.resolve_alert(
            alert_id=alert_id,
            clinician_id=req.clinician_id,
            action_taken=req.action_taken.value,
            resolution_notes=req.resolution_notes
        )

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        await repo.record_audit(
            audit_id=audit_id,
            alert_id=alert_id,
            action="resolved",
            actor_id=req.clinician_id,
            actor_role=role.value,
            previous_status=prev_status,
            new_status="resolved",
            details_dict={
                "action_taken": req.action_taken.value,
                "resolution_notes": req.resolution_notes
            }
        )

        return {
            "alert_id": updated.event_id,
            "status": updated.status,
            "action_taken": updated.action_taken,
            "resolution_notes": updated.resolution_notes,
            "resolved_by": updated.resolved_by,
            "resolved_at": updated.resolved_at.isoformat() if updated.resolved_at else None
        }


@router.get("/{alert_id}/audit")
async def get_alert_audit_history(
    alert_id: str,
    user_ctx: tuple[UserRole, str] = Depends(get_current_user_context)
) -> Dict[str, Any]:
    """Retrieve immutable forensic audit trail for an escalation ticket."""
    role, user_id = user_ctx
    if not AccessControlManager.check_permission(role, "can_view_audit"):
        raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized to inspect audit logs.")

    async with get_db_session() as db:
        repo = EscalationRepository(db)
        audits = await repo.get_audit_trail(alert_id)
        return {
            "alert_id": alert_id,
            "count": len(audits),
            "audit_trail": [
                {
                    "audit_id": a.audit_id,
                    "action": a.action,
                    "actor_id": a.actor_id,
                    "actor_role": a.actor_role,
                    "previous_status": a.previous_status,
                    "new_status": a.new_status,
                    "details": json.loads(a.details_json) if a.details_json else {},
                    "timestamp": a.timestamp.isoformat() if a.timestamp else None
                }
                for a in audits
            ]
        }
