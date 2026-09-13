import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app
from zenova.db.session import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    with TestClient(app) as client:
        yield client


def test_health_endpoint(setup_api):
    client = setup_api
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ZENOVA"


def test_status_endpoint(setup_api):
    client = setup_api
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_placeholder_mode"] is True
    assert "emotion" in data["active_providers"]


def test_models_endpoint(setup_api):
    client = setup_api
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_providers" in data


def test_conversation_standard_turn(setup_api):
    client = setup_api
    payload = {
        "session_id": "api_test_sess_1",
        "user_id": "api_test_user_1",
        "text": "I had a busy week at work."
    }
    response = client.post("/api/v1/conversation/turn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "api_test_sess_1"
    assert data["escalated_to_human"] is False
    assert len(data["response"]) > 0


def test_conversation_crisis_turn_and_escalation(setup_api):
    client = setup_api
    payload = {
        "session_id": "api_crisis_sess_1",
        "user_id": "api_crisis_user_1",
        "text": "I can't go on, I want to hurt myself tonight."
    }
    response = client.post("/api/v1/conversation/turn", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["escalated_to_human"] is True
    assert data["escalation_event_id"] is not None
    assert "988" in data["response"]

    # Verify escalation is listed
    esc_resp = client.get("/api/v1/escalations")
    assert esc_resp.status_code == 200
    esc_data = esc_resp.json()
    assert esc_data["count"] >= 1

    event_id = data["escalation_event_id"]
    ack_resp = client.post(
        f"/api/v1/escalations/{event_id}/acknowledge",
        json={"clinician_id": "dr_alice", "notes": "Contacted emergency protocol"}
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["status"] == "acknowledged"
