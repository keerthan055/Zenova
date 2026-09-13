"""Integration tests for Behavioral and Passive-Sensing API and Orchestrator."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app


@pytest.mark.asyncio
async def test_list_supported_devices():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/behavior/devices")
        assert resp.status_code == 200
        devices = resp.json()
        assert "smartphone_passive" in devices
        assert "apple_watch" in devices
        assert "fitbit" in devices


@pytest.mark.asyncio
async def test_behavior_profile_initial_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/behavior/usr_brand_new_passive/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "usr_brand_new_passive"
        assert data["is_established"] is False
        assert data["observation_count"] == 0


@pytest.mark.asyncio
async def test_behavior_observation_ingest_and_summary_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "usr_behavior_flow_test"

        # Ingest 3 observations
        for i in range(3):
            obs_payload = {
                "user_id": user_id,
                "step_count": 7500 + i * 200,
                "walking_minutes": 35.0,
                "sleep_duration_hours": 7.5,
                "conversation_duration_minutes": 60.0,
                "screen_unlock_count": 50,
            }
            resp = await ac.post("/api/v1/behavior/observation", json=obs_payload)
            assert resp.status_code == 200
            report = resp.json()
            assert report["user_id"] == user_id
            assert "disclaimer" in report

        # Summary check
        sum_resp = await ac.get(f"/api/v1/behavior/{user_id}/summary")
        assert sum_resp.status_code == 200
        summary = sum_resp.json()
        assert summary["observation_count"] == 3
        assert summary["is_established"] is True
        assert "mean_daily_steps" in summary["tracked_features"]


@pytest.mark.asyncio
async def test_ingest_raw_wearable_telemetry():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        req = {
            "user_id": "usr_apple_api",
            "device_type": "apple_watch",
            "payload": {
                "healthKitData": {
                    "HKQuantityTypeIdentifierStepCount": 11200,
                    "HKQuantityTypeIdentifierAppleExerciseTime": 55.0,
                    "sleepDurationHours": 8.1,
                }
            }
        }
        resp = await ac.post("/api/v1/behavior/ingest-raw", json=req)
        assert resp.status_code == 200
        report = resp.json()
        assert report["user_id"] == "usr_apple_api"


@pytest.mark.asyncio
async def test_end_to_end_conversation_strict_optionality():
    """Verify that when no behavioral telemetry is provided in a conversation turn,
    the system proceeds with 100% functionality and returns behavior.is_available=False.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        turn_payload = {
            "session_id": "sess_strict_opt_01",
            "user_id": "usr_strict_opt_01",
            "text": "I had a productive day at work.",
            "modality": "text",
            "metadata": {}
        }
        resp = await ac.post("/api/v1/conversation/turn", json=turn_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] is not None
        assert "behavior" in data
        assert data["behavior"] is not None
        assert data["behavior"]["is_available"] is False
        assert data["behavior"]["is_placeholder"] is False


@pytest.mark.asyncio
async def test_end_to_end_conversation_with_behavioral_telemetry():
    """Verify that when behavioral telemetry is provided in metadata,
    the pipeline ingests it, populates behavior, and reflects it in the turn.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        turn_payload = {
            "session_id": "sess_behavior_turn_01",
            "user_id": "usr_behavior_turn_01",
            "text": "I feel exhausted, haven't been sleeping well.",
            "modality": "text",
            "metadata": {
                "behavior": {
                    "step_count": 2100,
                    "walking_minutes": 10.0,
                    "sleep_duration_hours": 4.0,
                    "conversation_duration_minutes": 15.0,
                }
            }
        }
        resp = await ac.post("/api/v1/conversation/turn", json=turn_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["behavior"] is not None
        assert data["behavior"]["is_available"] is True
        assert data["behavior"]["sleep_duration_hours"] == 4.0
        assert data["behavior"]["activity_level"] == "sedentary"
        assert "Passive behavioral signals are observational proxies" in data["behavior"]["disclaimer"]


@pytest.mark.asyncio
async def test_clear_user_behavioral_data():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "usr_to_clear"
        obs_payload = {
            "user_id": user_id,
            "step_count": 5000,
        }
        await ac.post("/api/v1/behavior/observation", json=obs_payload)

        # Clear data
        clear_resp = await ac.post(f"/api/v1/behavior/{user_id}/clear")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["status"] == "success"

        # Check summary is cleared
        sum_resp = await ac.get(f"/api/v1/behavior/{user_id}/summary")
        assert sum_resp.status_code == 200
        assert sum_resp.json()["observation_count"] == 0
