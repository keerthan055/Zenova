"""Integration tests for ZENOVA Multimodal Context Engine API and Orchestration."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.db.session import init_db
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, EmotionCategory, RiskLevel


@pytest.mark.asyncio
async def test_context_build_endpoint():
    """Test POST /api/v1/context/build returns fully normalized MultimodalContext."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": {
                "session_id": "sess_ctx_api_1",
                "user_id": "user_api_1",
                "text": "I feel stressed by all these deadlines."
            },
            "emotion": {
                "primary_emotion": "anxiety",
                "confidence": 0.88,
                "valence": -0.4,
                "arousal": 0.6,
                "dominance": -0.3,
                "probabilities": {"anxiety": 0.88, "neutral": 0.12}
            },
            "risk": {
                "risk_level": "low",
                "confidence": 0.95,
                "crisis_category": "none",
                "is_high_risk": False,
                "requires_escalation": False
            }
        }
        resp = await ac.post("/api/v1/context/build", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Validate 9 keys
        for key in ["conversation", "emotion", "symptoms", "risk", "baseline", "behavior", "voice", "history", "metadata"]:
            assert key in data

        assert data["conversation"]["session_id"] == "sess_ctx_api_1"
        assert data["emotion"]["primary_emotion"] == "anxiety"
        assert data["behavior"]["is_available"] is False
        assert data["voice"]["is_available"] is False
        assert len(data["metadata"]["context_hash"]) == 64


@pytest.mark.asyncio
async def test_context_anonymize_endpoint():
    """Test POST /api/v1/context/anonymize scrubs PII and pseudonymizes identifiers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        sample_context = {
            "conversation": {
                "session_id": "sess_real_123",
                "user_id": "john_doe_456",
                "current_text": "Please reach me at john@doe.org or call 555-987-6543."
            },
            "history": {
                "recent_turns": [
                    {"turn_id": 1, "speaker": "user", "text": "My phone is 555-000-1111."}
                ],
                "user_feedback": []
            },
            "metadata": {}
        }
        resp = await ac.post(
            "/api/v1/context/anonymize",
            json={"context": sample_context, "privacy_level": "ANONYMIZED"}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["privacy_level"] == "ANONYMIZED"
        assert data["redacted_tokens_count"] >= 3
        anon_conv = data["anonymized_context"]["conversation"]
        assert anon_conv["user_id"].startswith("anon_")
        assert anon_conv["session_id"].startswith("anon_")
        assert "[REDACTED_EMAIL]" in anon_conv["current_text"]
        assert "[REDACTED_PHONE]" in anon_conv["current_text"]


@pytest.mark.asyncio
async def test_orchestrator_context_generation_and_persistence():
    """Verify that processing a turn with ZenovaOrchestrator generates and stores MultimodalContext."""
    await init_db()
    orchestrator = ZenovaOrchestrator()

    u_in = UserInput(
        session_id="sess_orch_context_test",
        user_id="user_orch_context",
        text="I am feeling overwhelmed with school work."
    )

    result = await orchestrator.process_turn(u_in)

    assert "context" in result
    ctx_data = result["context"]
    assert ctx_data is not None
    assert ctx_data["conversation"]["session_id"] == "sess_orch_context_test"
    assert "metadata" in ctx_data
    assert "context_hash" in ctx_data["metadata"]

    # Verify latest context can be retrieved via API
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/api/v1/context/{u_in.session_id}/latest")
        assert resp.status_code == 200
        snap = resp.json()
        assert snap["session_id"] == "sess_orch_context_test"
        assert snap["context_hash"] == ctx_data["metadata"]["context_hash"]
        assert "context" in snap


@pytest.mark.asyncio
async def test_turn_feedback_endpoint():
    """Verify recording turn feedback via /api/v1/context/{session_id}/feedback."""
    await init_db()
    orchestrator = ZenovaOrchestrator()

    u_in = UserInput(
        session_id="sess_feedback_test",
        user_id="user_feedback",
        text="I appreciate your help earlier."
    )
    turn_res = await orchestrator.process_turn(u_in)
    user_turn_id = turn_res["turn_id"]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        feedback_payload = {
            "turn_id": user_turn_id,
            "rating": 5,
            "is_helpful": True,
            "feedback_text": "This response was really validating."
        }
        resp = await ac.post(
            f"/api/v1/context/{u_in.session_id}/feedback",
            json=feedback_payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "feedback_recorded"
        assert data["turn_id"] == user_turn_id
        assert data["rating"] == 5
