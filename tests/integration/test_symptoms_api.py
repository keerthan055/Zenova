"""Integration tests for Symptom Identification API endpoints and Orchestrator."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app


@pytest.mark.asyncio
async def test_symptom_identify_endpoint_canonical():
    """Test POST /api/v1/symptoms/identify with canonical benchmark utterance."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/symptoms/identify",
            json={"text": "I haven't been sleeping properly and I don't enjoy things anymore."}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "signals" in data
        assert len(data["signals"]) >= 2

        signal_labels = [s["label"] for s in data["signals"]]
        assert "sleep disturbance" in signal_labels
        assert "loss of interest" in signal_labels

        # Verify confidences and evidence spans
        for sig in data["signals"]:
            assert 0.0 <= sig["confidence"] <= 1.0
            if sig["label"] in ("sleep disturbance", "loss of interest"):
                assert len(sig["evidence_spans"]) > 0

        # Verify disclaimer
        assert "not constitute" in data["disclaimer"].lower() or "informational" in data["disclaimer"].lower()


@pytest.mark.asyncio
async def test_symptom_identify_endpoint_benign():
    """Test POST /api/v1/symptoms/identify with non-symptom benign query."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/symptoms/identify",
            json={"text": "Can you give me some tips on how to prepare for an exam?"}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["signals"] == []
        assert data["aggregate_severity"] == "none"
        assert data["is_crisis_flagged"] is False


@pytest.mark.asyncio
async def test_conversation_turn_with_active_symptom_module():
    """Test end-to-end turn processing in orchestrator has is_placeholder: false for symptoms."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "session_id": "sess_integration_symptom_01",
            "user_id": "user_integration_symptom_01",
            "text": "I haven't been sleeping properly and I don't enjoy things anymore."
        }
        resp = await ac.post("/api/v1/conversation/turn", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Both emotion and symptoms should now be active real models!
        assert data["emotion"]["is_placeholder"] is False
        assert data["symptoms"]["is_placeholder"] is False

        symptom_signals = [s["marker_name"] for s in data["symptoms"]["signals"]]
        assert "sleep disturbance" in symptom_signals
        assert "loss of interest" in symptom_signals
