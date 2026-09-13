"""Integration tests for strategy-controlled response generation API and orchestrator pipeline."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app
from zenova.models.registry import ModelRegistry
from zenova.generation.generator import StrategyControlledGenerator
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType, SupportStrategy


@pytest.mark.asyncio
async def test_get_generation_providers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/generation/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) >= 4
        names = [p["name"] for p in data["providers"]]
        assert "local" in names
        assert "openai" in names


@pytest.mark.asyncio
async def test_generate_reflection_response_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "text": "I feel completely overwhelmed by everything happening at work right now.",
            "strategy": "Reflection of feelings",
            "emotion": "overwhelmed",
            "situation": "Heavy workload and impending deadlines"
        }
        response = await client.post("/api/v1/generation/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        assert len(data["response_text"]) > 15
        assert data["strategy_applied"] == "Reflection of feelings"
        assert data["validation_passed"] is True
        assert "provider" in data
        assert data["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_generate_question_response_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "text": "I don't know where to start sorting out my problems.",
            "strategy": "Question",
            "stage": "Exploration"
        }
        response = await client.post("/api/v1/generation/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response_text" in data
        assert data["strategy_applied"] == "Question"
        assert data["validation_passed"] is True


def test_model_registry_generator_resolution():
    registry = ModelRegistry()
    active_prov = registry.get_active_provider("generator")
    assert active_prov == "strategy_controlled"
    instance = registry.get_module_instance("generator")
    assert isinstance(instance, StrategyControlledGenerator)


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_strategy_controlled_turn():
    """Verify full pipeline: user input -> context engine -> strategy planner -> strategy-controlled generator -> response."""
    import uuid
    orchestrator = ZenovaOrchestrator()
    user_input = UserInput(
        session_id=f"gen-pipeline-session-{uuid.uuid4().hex[:6]}",
        user_id="gen-pipeline-user",
        text="I failed my exam and I feel like I let my family down completely.",
        modality=ModalityType.TEXT
    )
    result = await orchestrator.process_turn(user_input)

    assert "response" in result
    assert len(result["response"]) > 10
    assert "strategy" in result
    assert result["strategy"]["is_placeholder"] is False
    assert result["strategy"]["selected_strategy"] is not None

    assert "context" in result
    assert result["escalated_to_human"] is False
