"""Integration tests for Support Strategy API endpoints and Orchestrator integration."""
import pytest
from httpx import AsyncClient, ASGITransport
from zenova.api.app import app
from zenova.models.registry import ModelRegistry
from zenova.strategy.planner import ESConvStrategyPlanner
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType


@pytest.mark.asyncio
async def test_get_strategy_taxonomy():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/strategy/taxonomy")
        assert response.status_code == 200
        data = response.json()
        assert data["taxonomy_name"] == "ESConv-8"
        assert data["num_classes"] == 8
        assert "Question" in data["strategies"]
        assert "Reflection of feelings" in data["strategies"]


@pytest.mark.asyncio
async def test_predict_strategy_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "text": "I feel like giving up on my coursework because nothing makes sense.",
            "emotion": "sadness",
            "situation": "Struggling in university math class",
            "problem_type": "academic"
        }
        response = await client.post("/api/v1/strategy/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "selected_strategy" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "stage" in data
        assert "ranked_strategies" in data
        assert len(data["alternatives"]) > 0
        assert "rationale" in data


def test_model_registry_strategy_resolution():
    registry = ModelRegistry()
    active_prov = registry.get_active_provider("strategy")
    assert active_prov == "transformer"
    instance = registry.get_module_instance("strategy")
    assert isinstance(instance, ESConvStrategyPlanner)


@pytest.mark.asyncio
async def test_orchestrator_turn_with_strategy_planner():
    orchestrator = ZenovaOrchestrator()
    user_input = UserInput(
        session_id="strategy-test-session",
        user_id="strategy-test-user",
        text="I am really stressed about finding an internship before graduation.",
        modality=ModalityType.TEXT
    )
    result = await orchestrator.process_turn(user_input)
    assert "strategy" in result
    assert result["strategy"]["is_placeholder"] is False
    assert result["strategy"]["selected_strategy"] is not None
    assert result["strategy"]["confidence"] >= 0.0
