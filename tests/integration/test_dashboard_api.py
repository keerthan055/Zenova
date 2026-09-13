"""Integration tests for Clinician Dashboard API endpoints, RBAC, and HTML Web Portal."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.db.session import init_db, get_db_session
from zenova.db.repositories import SessionRepository, TurnRepository, EscalationRepository


@pytest.mark.asyncio
async def test_dashboard_modules_endpoint():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/dashboard/modules", headers={"X-User-Role": "clinician"})
        assert res.status_code == 200
        data = res.json()
        assert data["count"] >= 10
        mod_ids = [m["module_id"] for m in data["modules"]]
        assert "overview_alerts" in mod_ids
        assert "explainability_inspector" in mod_ids


@pytest.mark.asyncio
async def test_dashboard_overview_endpoint():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/dashboard/overview", headers={"X-User-Role": "clinician", "X-User-ID": "dr_smith"})
        assert res.status_code == 200
        data = res.json()
        assert "active_alerts_count" in data
        assert "system_status" in data
        assert data["system_status"]["status"] == "operational"
        assert "DO NOT constitute psychiatric diagnoses" in data["clinical_notice"]


@pytest.mark.asyncio
async def test_dashboard_patient_endpoints():
    await init_db()
    # Populate a test session and turn
    async with get_db_session() as session:
        sess_repo = SessionRepository(session)
        await sess_repo.get_or_create(session_id="dash-int-sess-1", user_id="patient-int-1")

        turn_repo = TurnRepository(session)
        await turn_repo.record_turn(
            session_id="dash-int-sess-1",
            turn_id=1,
            speaker="user",
            content="I am having trouble sleeping and feeling hopeless.",
            emotion_json='{"primary_emotion": "sadness", "valence": -0.65, "arousal": 0.3, "confidence": 0.89}',
            symptom_json='{"signals": [{"category": "depressive_symptoms", "severity": "moderate"}, {"category": "sleep_disturbances", "severity": "mild"}]}',
            risk_json='{"risk_level": "moderate", "crisis_category": "none", "confidence": 0.82}'
        )
        await turn_repo.record_turn(
            session_id="dash-int-sess-1",
            turn_id=2,
            speaker="assistant",
            content="I hear how exhausting that must feel. Would you like to share more?",
            strategy_json='{"selected_strategy": "Reflection of feelings", "confidence": 0.91}'
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-User-Role": "clinician", "X-User-ID": "dr_smith"}

        # 1. Users list
        res_users = await client.get("/api/v1/dashboard/users", headers=headers)
        assert res_users.status_code == 200
        users_data = res_users.json()
        assert users_data["count"] > 0

        # 2. Patient summary
        res_summary = await client.get("/api/v1/dashboard/users/patient-int-1/summary", headers=headers)
        assert res_summary.status_code == 200
        summary_data = res_summary.json()
        assert summary_data["user_id"] == "patient-int-1"
        assert summary_data["current_risk_level"] in ("moderate", "low", "no_risk")

        # 3. Patient trends
        res_trends = await client.get("/api/v1/dashboard/users/patient-int-1/trends", headers=headers)
        assert res_trends.status_code == 200
        trends_data = res_trends.json()
        assert len(trends_data["emotion_trends"]) >= 1
        assert trends_data["emotion_trends"][0]["primary_emotion"] == "sadness"
        assert len(trends_data["strategy_trends"]) >= 1
        assert trends_data["strategy_trends"][0]["strategy"] == "Reflection of feelings"

        # 4. Patient timeline
        res_timeline = await client.get("/api/v1/dashboard/users/patient-int-1/timeline", headers=headers)
        assert res_timeline.status_code == 200
        timeline_data = res_timeline.json()
        assert timeline_data["events_count"] >= 2

        # 5. Patient conversations
        res_convs = await client.get("/api/v1/dashboard/users/patient-int-1/conversations", headers=headers)
        assert res_convs.status_code == 200
        convs_data = res_convs.json()
        assert convs_data["turns_count"] >= 2
        turn1 = convs_data["turns"][0]
        assert turn1["speaker"] == "user"
        assert "hopeless" in turn1["content"]
        assert len(turn1["explainability"]) >= 2

        # 6. Patient explainability
        res_exp = await client.get("/api/v1/dashboard/users/patient-int-1/explainability", headers=headers)
        assert res_exp.status_code == 200
        exp_data = res_exp.json()
        assert len(exp_data["model_cards"]) >= 7


@pytest.mark.asyncio
async def test_dashboard_system_status_endpoint():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/dashboard/system-status", headers={"X-User-Role": "clinician"})
        assert res.status_code == 200
        data = res.json()
        assert data["system_name"] == "ZENOVA"
        assert "active_models" in data
        assert "emotion" in data["active_models"]
        assert "risk" in data["active_models"]


@pytest.mark.asyncio
async def test_dashboard_rbac_audit_logs():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Clinician cannot view audit logs
        res_clinician = await client.get("/api/v1/dashboard/audit-logs", headers={"X-User-Role": "clinician"})
        assert res_clinician.status_code == 403

        # Auditor can view audit logs
        res_auditor = await client.get("/api/v1/dashboard/audit-logs", headers={"X-User-Role": "auditor"})
        assert res_auditor.status_code == 200
        data = res_auditor.json()
        assert "logs" in data


@pytest.mark.asyncio
async def test_dashboard_patient_role_forbidden():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for route in [
            "/api/v1/dashboard/overview",
            "/api/v1/dashboard/users",
            "/api/v1/dashboard/modules",
            "/api/v1/dashboard/system-status"
        ]:
            res = await client.get(route, headers={"X-User-Role": "patient"})
            assert res.status_code == 403
            assert "Patients are strictly prohibited" in res.json()["detail"] or "prohibited" in res.json()["detail"]


@pytest.mark.asyncio
async def test_dashboard_html_web_app_served():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/dashboard")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
        html = res.text
        assert "ZENOVA Clinician Decision-Support Portal" in html
        assert "role-select" in html
        assert "tab-overview" in html
        assert "tab-patients" in html
        assert "tab-timeline" in html
        assert "tab-explainability" in html
        assert "tab-audit" in html
        assert "CLINICAL NOTICE:" in html
