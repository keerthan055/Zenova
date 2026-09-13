"""Integration tests for Risk Detection API and Orchestrator Crisis Bypass."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app


@pytest.mark.asyncio
async def test_risk_detect_endpoint_high_risk():
    """Test POST /api/v1/risk/detect with active crisis text."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/risk/detect",
            json={"text": "I want to end my life, I'm planning my suicide tonight."}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["risk_level"] in ("high", "critical")
        assert data["requires_escalation"] is True
        assert len(data["trigger_cues"]) > 0
        assert "not clinical certainty" in data["disclaimer"].lower()


@pytest.mark.asyncio
async def test_risk_detect_endpoint_adversarial_idiom():
    """Test POST /api/v1/risk/detect with colloquial metaphor does not escalate."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/risk/detect",
            json={"text": "I'm dying of laughter, that meme is hilarious!"}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["risk_level"] == "low"
        assert data["requires_escalation"] is False


@pytest.mark.asyncio
async def test_orchestrator_crisis_bypass_end_to_end():
    """Test that HIGH/CRITICAL risk interrupts normal generation and provides 988 lifeline."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "session_id": "sess_crisis_bypass_test_01",
            "user_id": "user_crisis_bypass_test_01",
            "text": "I want to end my life right now, please forgive me."
        }
        resp = await ac.post("/api/v1/conversation/turn", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Verify risk module is active and real
        assert data["risk"]["is_placeholder"] is False
        assert data["risk"]["risk_level"] in ("high", "critical")
        assert data["risk"]["requires_immediate_escalation"] is True

        # Verify assistant response turn was overridden with emergency hotline
        assistant_content = data["response"]
        assert "988" in assistant_content
        assert "741741" in assistant_content
        assert "crisis" in assistant_content.lower() or "safety" in assistant_content.lower()


@pytest.mark.asyncio
async def test_orchestrator_benign_turn_proceeds():
    """Test that LOW risk turn does not trigger crisis bypass."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "session_id": "sess_benign_turn_test_02",
            "user_id": "user_benign_turn_test_02",
            "text": "I had a productive study session today and learned about machine learning."
        }
        resp = await ac.post("/api/v1/conversation/turn", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["risk"]["is_placeholder"] is False
        assert data["risk"]["risk_level"] == "low"
        assert data["risk"]["requires_immediate_escalation"] is False
        assert "988" not in data["response"]
