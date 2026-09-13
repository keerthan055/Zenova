"""Integration tests for ZENOVA User-Facing Application routes and Single Page Web Interface."""
import uuid
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app
from zenova.db.session import init_db


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_user_app_spa_html_served(client):
    """Verify GET /app serves the responsive single-page web app with disclaimers and ARIA roles."""
    response = client.get("/app")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text

    # Verify accessibility and critical clinical disclaimers
    assert "ZENOVA" in html
    assert "Non-Medical" in html or "not a licensed medical professional" in html
    assert "988" in html
    assert 'role="log"' in html
    assert 'aria-live="polite"' in html
    assert 'role="tablist"' in html
    assert "Right to be Forgotten" in html


def test_user_support_resources_endpoint(client):
    """Verify GET /api/v1/user/resources returns verified crisis lines and grounding exercises."""
    response = client.get("/api/v1/user/resources")
    assert response.status_code == 200
    data = response.json()
    assert "hotlines" in data
    assert len(data["hotlines"]) >= 5
    assert "grounding_techniques" in data
    assert len(data["grounding_techniques"]) >= 2
    assert "disclaimer" in data

    # Verify 988 hotline is present
    h_names = [h["name"] for h in data["hotlines"]]
    assert any("988" in name for name in h_names)


def test_user_preferences_api_crud(client):
    """Verify GET and PUT for user settings and privacy controls."""
    uid = f"api_user_prefs_{uuid.uuid4().hex[:8]}"

    # Get default preferences
    r_get = client.get(f"/api/v1/user/preferences/{uid}")
    assert r_get.status_code == 200
    prefs = r_get.json()
    assert prefs["user_id"] == uid
    assert prefs["save_history"] is True
    assert prefs["privacy_level"] == "standard"

    # Update preferences
    update_payload = {
        "save_history": False,
        "enable_voice": False,
        "privacy_level": "anonymized",
        "communication_style": "solution_focused"
    }
    r_put = client.put(f"/api/v1/user/preferences/{uid}", json=update_payload)
    assert r_put.status_code == 200
    updated = r_put.json()
    assert updated["save_history"] is False
    assert updated["enable_voice"] is False
    assert updated["privacy_level"] == "anonymized"
    assert updated["communication_style"] == "solution_focused"


def test_user_checkin_api_flow(client):
    """Verify submitting daily check-in and querying longitudinal history and summary stats."""
    uid = f"api_user_checkin_{uuid.uuid4().hex[:8]}"

    checkin_payload = {
        "user_id": uid,
        "mood_score": 8,
        "valence": 0.6,
        "sleep_hours": 8.0,
        "stress_level": 2,
        "energy_level": 4,
        "notes": "Had a restful morning and a walk outside."
    }
    r_post = client.post("/api/v1/user/checkin", json=checkin_payload)
    assert r_post.status_code == 200
    res = r_post.json()
    assert res["user_id"] == uid
    assert res["mood_score"] == 8
    assert "feedback_message" in res

    # Retrieve history
    r_hist = client.get(f"/api/v1/user/checkins/{uid}")
    assert r_hist.status_code == 200
    hist = r_hist.json()
    assert hist["total_checkins"] >= 1
    assert hist["average_mood"] == 8.0
    assert hist["average_sleep_hours"] == 8.0
    assert hist["checkins"][0]["mood_score"] == 8


def test_user_conversation_and_sessions_flow(client):
    """Verify conversation turn through orchestrator is tracked in user sessions."""
    uid = f"api_user_conv_{uuid.uuid4().hex[:8]}"
    sid = f"api_sess_conv_{uuid.uuid4().hex[:8]}"

    # Send a turn through the high-level orchestrator process endpoint
    turn_payload = {
        "session_id": sid,
        "user_id": uid,
        "text": "I would like to explore some mindfulness techniques for stress.",
        "metadata": {"client": "integration_test"}
    }
    r_turn = client.post("/api/v1/orchestrator/process", json=turn_payload)
    assert r_turn.status_code == 200
    t_res = r_turn.json()
    assert t_res["session_id"] == sid
    assert len(t_res["response"]) > 0

    # Check user sessions list
    r_sess = client.get(f"/api/v1/user/sessions/{uid}")
    assert r_sess.status_code == 200
    s_data = r_sess.json()
    assert s_data["total_sessions"] >= 1
    session_ids = [s["session_id"] for s in s_data["sessions"]]
    assert sid in session_ids


def test_user_data_export_and_purge_api(client):
    """Verify GDPR data export and complete user data purge via REST API."""
    uid = f"api_user_purge_{uuid.uuid4().hex[:8]}"
    sid = f"api_sess_purge_{uuid.uuid4().hex[:8]}"

    # Ingest turn and check-in
    client.post(
        "/api/v1/orchestrator/process",
        json={"session_id": sid, "user_id": uid, "text": "This data will be purged."}
    )
    client.post(
        "/api/v1/user/checkin",
        json={"user_id": uid, "mood_score": 7, "sleep_hours": 7.5, "stress_level": 3, "energy_level": 3}
    )

    # 1. Export Data
    r_exp = client.post(f"/api/v1/user/export/{uid}")
    assert r_exp.status_code == 200
    exp = r_exp.json()
    assert exp["user_id"] == uid
    assert "preferences" in exp
    assert exp["sessions_count"] >= 1
    assert len(exp["checkins"]) >= 1

    # 2. Purge Data
    r_del = client.delete(f"/api/v1/user/data/{uid}")
    assert r_del.status_code == 200
    del_res = r_del.json()
    assert del_res["status"] == "PURGED_SUCCESSFULLY"
    assert del_res["deleted_sessions"] >= 1
    assert del_res["deleted_checkins"] >= 1

    # 3. Verify Sessions are gone
    r_after = client.get(f"/api/v1/user/sessions/{uid}")
    assert r_after.status_code == 200
    assert r_after.json()["total_sessions"] == 0
