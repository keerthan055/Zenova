"""Integration tests for Curated RAG API endpoints and orchestrator integration."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType, SupportStrategy


@pytest.mark.asyncio
async def test_rag_status_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/rag/status")
        assert response.status_code == 200
        data = response.json()
        assert data["is_initialized"] is True
        assert data["indexed_chunks"] >= 18
        assert data["registered_sources"] >= 6
        assert data["embedding_dimension"] > 0
        assert data["vector_store_type"] == "InMemoryVectorStore"


@pytest.mark.asyncio
async def test_rag_sources_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/rag/sources")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 6
        assert len(data["sources"]) >= 6

        publishers = [s["publisher"] for s in data["sources"]]
        assert "World Health Organization" in publishers
        assert "National Institute of Mental Health" in publishers
        assert "American Psychological Association" in publishers
        assert any("National Health Service" in p for p in publishers)


@pytest.mark.asyncio
async def test_rag_query_confident_retrieval():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "diaphragmatic breathing and 4-7-8 relaxation method",
            "top_k": 3
        }
        response = await client.post("/api/v1/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_confident"] is True
        assert data["highest_score"] >= 0.30
        assert len(data["results"]) >= 1
        assert len(data["formatted_context"]) >= 1
        assert len(data["cited_sources"]) >= 1

        all_publishers = [s["publisher"] for s in data["cited_sources"]]
        assert any(
            "National Health Service" in p or "World Health Organization" in p or "National Institute of Mental Health" in p
            for p in all_publishers
        )
        assert data["cited_sources"][0]["url"].startswith("http")


@pytest.mark.asyncio
async def test_rag_query_low_confidence_antihallucination():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "cryptographic blockchain hash protocol proof of stake",
            "top_k": 3
        }
        response = await client.post("/api/v1/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_confident"] is False
        assert len(data["results"]) == 0
        assert len(data["formatted_context"]) == 0
        assert "NO_VERIFIED_KNOWLEDGE_FOUND" in data["status_message"]


@pytest.mark.asyncio
async def test_orchestrator_rag_grounded_turn():
    """Verify orchestrator activates RAG grounding on coping/breathing inquiries."""
    orchestrator = ZenovaOrchestrator()
    user_in = UserInput(
        session_id="rag-orch-test-session",
        user_id="user-rag-test",
        text="I am feeling stressed with work and would like some breathing exercises to calm down.",
        modality=ModalityType.TEXT
    )
    result = await orchestrator.process_turn(user_in)

    assert result["escalated_to_human"] is False
    assert result["response"] is not None
    assert len(result["response"]) > 10
    assert result["rag"] is not None
    assert result["rag"]["is_confident"] is True
    assert len(result["rag"]["results"]) >= 1


@pytest.mark.asyncio
async def test_orchestrator_crisis_bypasses_rag():
    """Verify crisis bypass safely takes precedence and does not perform RAG distraction."""
    orchestrator = ZenovaOrchestrator()
    user_in = UserInput(
        session_id="rag-crisis-bypass-session",
        user_id="user-crisis-test",
        text="I want to end my life right now, I have everything prepared to commit suicide.",
        modality=ModalityType.TEXT
    )
    result = await orchestrator.process_turn(user_in)

    assert result["escalated_to_human"] is True
    assert result["rag"] is None
    assert "988" in result["response"]
