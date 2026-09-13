"""Integration tests for Response Safety Gate API endpoints and Orchestrator interception."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType, GeneratedResponse, SupportStrategy, RiskResult, RiskLevel


@pytest.mark.asyncio
async def test_safety_policies_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/safety/policies")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 12
        assert len(data["policies"]) == 12

        policy_ids = [p["id"] for p in data["policies"]]
        assert "harmful_instructions" in policy_ids
        assert "inappropriate_medical_claims" in policy_ids
        assert "unsupported_diagnosis" in policy_ids
        assert "crisis_mishandling" in policy_ids
        assert "unsafe_advice" in policy_ids
        assert "reinforcement_of_delusional_beliefs" in policy_ids
        assert "inappropriate_dependency" in policy_ids
        assert "manipulative_language" in policy_ids
        assert "false_claims_of_professional_authority" in policy_ids
        assert "privacy_violations" in policy_ids
        assert "hallucinated_resources" in policy_ids
        assert "dangerous_certainty" in policy_ids


@pytest.mark.asyncio
async def test_safety_verify_clean_response():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "candidate_text": "I hear how overwhelmed you feel by school. What feels like the hardest part right now?",
            "user_text": "I feel swamped with homework.",
            "risk_level": "low"
        }
        response = await client.post("/api/v1/safety/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is True
        assert data["action"] == "allow"
        assert len(data["violations"]) == 0
        assert data["override_applied"] is False


@pytest.mark.asyncio
async def test_safety_verify_medical_claim_revision():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "candidate_text": "You should definitely start taking 20mg of Lexapro daily to cure your clinical depression.",
            "user_text": "I feel hopeless.",
            "risk_level": "low"
        }
        response = await client.post("/api/v1/safety/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is False
        assert data["action"] == "revise"
        assert data["medical_advice_detected"] is True
        assert data["override_applied"] is True
        assert "Lexapro" not in data["modified_text"]


@pytest.mark.asyncio
async def test_safety_verify_crisis_block_and_escalate():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "candidate_text": "Cheer up, it's not that bad! Just take a bubble bath.",
            "user_text": "I want to end my life right now.",
            "risk_level": "critical"
        }
        response = await client.post("/api/v1/safety/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_safe"] is False
        assert data["action"] == "block_and_escalate"
        assert data["crisis_escalation_required"] is True
        assert "988" in data["modified_text"]
        assert "741741" in data["modified_text"]


@pytest.mark.asyncio
async def test_safety_audit_logs_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/safety/audit?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "audits" in data
        assert "source" in data
        assert len(data["audits"]) >= 1

        for audit in data["audits"]:
            assert "audit_id" in audit
            assert "action" in audit
            assert "is_safe" in audit
            # Verify no sensitive user text fields leaked
            assert "user_text" not in audit
            assert "candidate_text" not in audit


@pytest.mark.asyncio
async def test_orchestrator_adversarial_safety_override():
    """Verify safety gate intercepts adversarial model output in full orchestrator turn."""
    from zenova.db.session import init_db
    await init_db()
    orchestrator = ZenovaOrchestrator()

    # User input
    user_in = UserInput(
        session_id="safety-orch-adversarial-session",
        user_id="safety-user",
        text="I have been having trouble sleeping lately.",
        modality=ModalityType.TEXT
    )

    result = await orchestrator.process_turn(user_in)

    # Standard safe turn
    assert result["escalated_to_human"] is False
    assert result["safety"] is not None
    assert "is_safe" in result["safety"]
    assert result["safety"]["is_safe"] is True
    assert result["response"] is not None
