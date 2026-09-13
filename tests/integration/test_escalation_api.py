"""Integration tests for Human/Clinician Escalation API, RBAC, and Orchestrator interception."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.db.session import init_db
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType


@pytest.mark.asyncio
async def test_escalation_rules_endpoint():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/escalations/rules", headers={"X-User-Role": "clinician"})
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 4
        rule_ids = [r["rule_id"] for r in data["rules"]]
        assert "EXPLICIT_HUMAN_REQUEST" in rule_ids
        assert "SUBSTANCE_OVERDOSE_SUSPICION" in rule_ids
        assert "DOMESTIC_VIOLENCE_INTIMIDATION" in rule_ids
        assert "PEDIATRIC_CRISIS" in rule_ids


@pytest.mark.asyncio
async def test_alert_lifecycle_api_workflow():
    """Verify full alert lifecycle: trigger -> list -> acknowledge -> status -> resolve -> audit."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-User-Role": "clinician", "X-User-ID": "dr_smith"}

        # 1. Trigger an alert
        trigger_payload = {
            "session_id": "sess-lifecycle-test",
            "user_id": "user-lifecycle",
            "severity": "critical",
            "trigger_type": "crisis_risk",
            "reason_code": "CRITICAL_RISK_DETECTED",
            "reason_title": "Severe Suicidal Ideation",
            "reason_description": "User reported active intent with means.",
            "trigger_cues": ["suicide plan", "nowhere else to turn"],
            "user_message": "I don't think I can make it through the night."
        }
        res_create = await client.post("/api/v1/escalations/trigger", json=trigger_payload, headers=headers)
        assert res_create.status_code == 200
        alert_data = res_create.json()
        alert_id = alert_data["alert_id"]
        assert alert_id.startswith("alt_")
        assert alert_data["status"] == "pending"

        # 2. List escalations and verify alert is present
        res_list = await client.get("/api/v1/escalations?status=pending", headers=headers)
        assert res_list.status_code == 200
        alerts = res_list.json()["alerts"]
        matching = [a for a in alerts if a["alert_id"] == alert_id]
        assert len(matching) == 1
        assert matching[0]["severity"] == "critical"

        # 3. Get single alert
        res_get = await client.get(f"/api/v1/escalations/{alert_id}", headers=headers)
        assert res_get.status_code == 200
        fetched = res_get.json()
        assert fetched["user_id"] == "user-lifecycle"
        assert "suicide plan" in fetched["trigger_cues"]

        # 4. Acknowledge alert
        ack_payload = {
            "clinician_id": "dr_smith",
            "notes": "Claimed for immediate risk review."
        }
        res_ack = await client.post(f"/api/v1/escalations/{alert_id}/acknowledge", json=ack_payload, headers=headers)
        assert res_ack.status_code == 200
        assert res_ack.json()["status"] == "acknowledged"
        assert res_ack.json()["assigned_clinician_id"] == "dr_smith"

        # 5. Update status to in_review
        status_payload = {
            "status": "in_review",
            "clinician_id": "dr_smith",
            "notes": "Reviewing past conversation history and safety plan."
        }
        res_status = await client.post(f"/api/v1/escalations/{alert_id}/status", json=status_payload, headers=headers)
        assert res_status.status_code == 200
        assert res_status.json()["status"] == "in_review"

        # 6. Resolve alert
        resolve_payload = {
            "clinician_id": "dr_smith",
            "action_taken": "emergency_services_contacted",
            "resolution_notes": "Dispatched emergency crisis unit to patient residence; patient in care."
        }
        res_resolve = await client.post(f"/api/v1/escalations/{alert_id}/resolve", json=resolve_payload, headers=headers)
        assert res_resolve.status_code == 200
        resolved_data = res_resolve.json()
        assert resolved_data["status"] == "resolved"
        assert resolved_data["action_taken"] == "emergency_services_contacted"
        assert resolved_data["resolved_by"] == "dr_smith"

        # 7. Audit trail verification
        res_audit = await client.get(f"/api/v1/escalations/{alert_id}/audit", headers=headers)
        assert res_audit.status_code == 200
        audit_records = res_audit.json()["audit_trail"]
        assert len(audit_records) >= 4  # created -> acknowledged -> status_changed -> resolved
        actions = [a["action"] for a in audit_records]
        assert "created" in actions
        assert "acknowledged" in actions
        assert "status_changed" in actions
        assert "resolved" in actions


@pytest.mark.asyncio
async def test_rbac_unauthorized_access_denied():
    """Verify patients/unauthorized roles receive 403 when accessing clinical escalation routes."""
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Patient cannot list queue
        res = await client.get("/api/v1/escalations", headers={"X-User-Role": "patient"})
        assert res.status_code == 403

        # Patient cannot acknowledge
        ack_payload = {"clinician_id": "imposter"}
        res_ack = await client.post("/api/v1/escalations/alt_fake/acknowledge", json=ack_payload, headers={"X-User-Role": "patient"})
        assert res_ack.status_code == 403


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_escalation():
    """Verify orchestrator automatically triggers escalation on acute self-harm input."""
    await init_db()
    orchestrator = ZenovaOrchestrator()

    user_in = UserInput(
        session_id="orch-escalation-session",
        user_id="crisis-user-1",
        text="I want to end my life right now, please don't stop me.",
        modality=ModalityType.TEXT
    )

    result = await orchestrator.process_turn(user_in)

    # Verify escalation was raised
    assert result["escalated_to_human"] is True
    assert result["escalation_id"] is not None
    assert "988" in str(result["response"])

    # Verify alert exists in API
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-User-Role": "clinician"}
        res = await client.get(f"/api/v1/escalations/{result['escalation_id']}", headers=headers)
        assert res.status_code == 200
        alert_json = res.json()
        assert alert_json["alert_id"] == result["escalation_id"]
        assert alert_json["status"] == "pending"
        assert alert_json["severity"] in ("critical", "high")
