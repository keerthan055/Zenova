"""Integration tests for ZENOVA Orchestrator REST API endpoints and RBAC trace inspection."""
import uuid
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_orchestrator_api_process_nominal(client):
    """Verify high-level /orchestrator/process endpoint with nominal turn."""
    sess_id = f"orch_sess_nom_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": sess_id,
        "user_id": "orch_user_1",
        "text": "I had a productive morning finishing my project on time."
    }

    response = client.post("/api/v1/orchestrator/process", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == sess_id
    assert data["turn_id"] >= 1
    assert data["escalated_to_human"] is False
    assert len(data["response"]) > 0
    assert data["pipeline_status"] in ["nominal", "degraded"]

    # Verify execution trace is returned in response
    assert "trace" in data
    trace = data["trace"]
    assert trace["trace_id"] == data["trace_id"]
    assert len(trace["spans"]) >= 4
    span_names = [s["span_name"] for s in trace["spans"]]
    assert "input_processing" in span_names
    assert "analytical_layer" in span_names
    assert "intervention" in span_names
    assert "persistence" in span_names


def test_orchestrator_api_process_crisis(client):
    """Verify high-level /orchestrator/process endpoint with acute crisis turn."""
    sess_id = f"orch_sess_crisis_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": sess_id,
        "user_id": "orch_user_crisis",
        "text": "I can't go on anymore, I am going to kill myself tonight."
    }

    response = client.post("/api/v1/orchestrator/process", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == sess_id
    assert data["escalated_to_human"] is True
    assert data["escalation_id"] is not None
    assert "988" in data["response"]
    assert "Crisis Text Line" in data["response"]
    assert data["pipeline_status"] == "crisis_bypass"


def test_orchestrator_api_get_trace_authorized(client):
    """Verify authorized clinician can inspect internal execution traces."""
    sess_id = f"orch_sess_trace_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": sess_id,
        "user_id": "orch_user_trace",
        "text": "I feel a bit overwhelmed but managing."
    }
    proc_res = client.post("/api/v1/orchestrator/process", json=payload)
    assert proc_res.status_code == 200
    trace_id = proc_res.json()["trace_id"]

    headers = {
        "X-User-Role": "clinician",
        "X-User-ID": "dr_smith"
    }
    res = client.get(f"/api/v1/orchestrator/traces/{trace_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["trace_id"] == trace_id
    assert data["session_id"] == sess_id
    assert len(data["spans"]) >= 4


def test_orchestrator_api_get_trace_unauthorized_patient(client):
    """Verify patients/unauthorized roles receive 403 Forbidden on internal trace inspection."""
    headers = {
        "X-User-Role": "patient",
        "X-User-ID": "patient_123"
    }
    res = client.get("/api/v1/orchestrator/traces/some_trace_id", headers=headers)
    assert res.status_code == 403
    assert "not authorized" in res.json()["detail"].lower()


def test_orchestrator_api_get_trace_not_found(client):
    """Verify 404 response when requesting non-existent trace."""
    headers = {
        "X-User-Role": "clinician",
        "X-User-ID": "dr_smith"
    }
    res = client.get("/api/v1/orchestrator/traces/non_existent_trace_9999", headers=headers)
    assert res.status_code == 404


def test_orchestrator_api_list_traces(client):
    """Verify listing and filtering execution traces for authorized clinicians."""
    headers = {
        "X-User-Role": "clinician",
        "X-User-ID": "dr_smith"
    }
    res = client.get("/api/v1/orchestrator/traces?limit=10", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "traces" in data
    assert "total" in data
    assert data["total"] >= 1
    assert len(data["traces"]) >= 1


def test_orchestrator_api_health(client):
    """Verify component-level health check endpoint."""
    res = client.get("/api/v1/orchestrator/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["nominal", "degraded"]
    assert data["total_components"] == 12
    assert data["available_components"] >= 10
    comp_names = [c["name"] for c in data["components"]]
    assert "emotion" in comp_names
    assert "symptom" in comp_names
    assert "risk" in comp_names
    assert "strategy" in comp_names
    assert "generator" in comp_names
    assert "safety" in comp_names
    assert "fusion" in comp_names
