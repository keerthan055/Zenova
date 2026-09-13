"""Integration tests for Personal Baseline Engine API and Orchestrator Integration."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app


@pytest.mark.asyncio
async def test_baseline_profile_initial_empty():
    """Test GET /api/v1/baseline/{user_id} for a brand new user."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/baseline/user_brand_new_999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user_brand_new_999"
        assert data["status"] == "insufficient_data"
        assert data["total_observations"] == 0
        assert data["confidence"] == 0.0


@pytest.mark.asyncio
async def test_baseline_observation_and_evaluation_flow():
    """Test ingesting observations and evaluating deviations via API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "user_api_baseline_test_01"

        # 1. Post 5 baseline observations
        for i in range(5):
            obs_payload = {
                "user_id": user_id,
                "valence": 0.20,
                "arousal": 0.30,
                "word_count": 25,
                "symptom_scores": {"anxiety": 0.15},
                "risk_severity_score": 0.0
            }
            obs_resp = await ac.post("/api/v1/baseline/observation", json=obs_payload)
            assert obs_resp.status_code == 200
            res_data = obs_resp.json()
            assert res_data["status"] == "recorded"

        # 2. Check profile status is now provisional
        prof_resp = await ac.get(f"/api/v1/baseline/{user_id}")
        assert prof_resp.status_code == 200
        prof_data = prof_resp.json()
        assert prof_data["total_observations"] == 5
        assert prof_data["status"] == "provisional_baseline"
        assert prof_data["confidence"] >= 0.40

        # 3. Evaluate an anomalous observation (high anxiety score: 0.88)
        eval_payload = {
            "user_id": user_id,
            "valence": -0.65,
            "arousal": 0.85,
            "word_count": 10,
            "symptom_scores": {"anxiety": 0.88},
            "risk_severity_score": 1.0
        }
        eval_resp = await ac.post("/api/v1/baseline/evaluate", json=eval_payload)
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()

        assert eval_data["user_id"] == user_id
        assert eval_data["is_significant_deviation"] is True
        assert len(eval_data["deviating_features"]) > 0
        assert "non-diagnostic" in eval_data["disclaimer"].lower()

        # Check anxiety deviation interpretation
        anx_dev = next(d for d in eval_data["deviations"] if "anxiety" in d["feature"])
        assert anx_dev["deviation"] > 2.0
        assert "above_personal_baseline" in anx_dev["interpretation"]


@pytest.mark.asyncio
async def test_baseline_reset_flow():
    """Test POST /api/v1/baseline/reset clears user baseline."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "user_api_reset_test"
        # Ingest one observation
        await ac.post("/api/v1/baseline/observation", json={"user_id": user_id, "valence": 0.5})

        # Reset
        reset_resp = await ac.post("/api/v1/baseline/reset", json={"user_id": user_id})
        assert reset_resp.status_code == 200
        assert reset_resp.json()["status"] == "success"

        # Verify profile is back to zero
        prof_resp = await ac.get(f"/api/v1/baseline/{user_id}")
        assert prof_resp.json()["total_observations"] == 0


@pytest.mark.asyncio
async def test_orchestrator_turn_with_personal_baseline():
    """Test end-to-end conversation turn invokes the active PersonalBaselineEngine."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "session_id": "sess_personal_baseline_turn",
            "user_id": "user_p_baseline_e2e",
            "text": "I had a wonderful day walking my dog in the sunshine."
        }
        resp = await ac.post("/api/v1/conversation/turn", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Verify baseline module output is active and not a placeholder
        baseline_res = data["baseline"]
        assert baseline_res["is_placeholder"] is False
        assert baseline_res["user_id"] == "user_p_baseline_e2e"
        assert baseline_res["status"] in ("insufficient_data", "provisional_baseline", "established_baseline")
        assert "disclaimer" in baseline_res
        assert "non-diagnostic" in baseline_res["disclaimer"].lower()
