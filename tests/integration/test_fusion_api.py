"""Integration tests for ZENOVA Multimodal Fusion API and Pipeline Orchestration."""
import pytest
from httpx import AsyncClient, ASGITransport

from zenova.api.app import app
from zenova.db.session import init_db
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput


@pytest.mark.asyncio
async def test_fusion_fuse_text_only_endpoint():
    """Test POST /api/v1/fusion/fuse with text-only payload."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": {
                "session_id": "sess_fusion_api_1",
                "user_id": "user_fusion_1",
                "text": "I feel a bit stressed about work."
            },
            "emotion": {
                "primary_emotion": "anxiety",
                "confidence": 0.85,
                "valence": -0.3,
                "arousal": 0.5,
                "dominance": -0.2,
                "probabilities": {"anxiety": 0.85, "neutral": 0.15}
            },
            "risk": {
                "risk_level": "low",
                "confidence": 0.95,
                "crisis_category": "none",
                "is_high_risk": False,
                "requires_escalation": False
            },
            "provider": "weighted_rule"
        }
        resp = await ac.post("/api/v1/fusion/fuse", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["fusion_method"] == "weighted_rule"
        assert data["primary_affect"] == "anxiety"
        assert data["fused_valence"] <= 0.0
        assert data["fused_risk_level"] == "low"
        assert data["modality_mask"]["text"] is True
        assert data["modality_mask"]["emotion"] is True
        assert data["modality_mask"]["voice"] is False
        assert data["modality_weights"]["voice"] == 0.0
        assert data["discrepancy"]["detected"] is False


@pytest.mark.asyncio
async def test_fusion_fuse_multimodal_acoustic_masking():
    """Test POST /api/v1/fusion/fuse with conflicting voice affect detecting discrepancy."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": {
                "session_id": "sess_fusion_api_2",
                "user_id": "user_fusion_2",
                "text": "I am totally fine, really nothing to worry about."
            },
            "emotion": {
                "primary_emotion": "neutral",
                "confidence": 0.80,
                "valence": 0.1,
                "arousal": 0.2,
                "dominance": 0.0
            },
            "voice": {
                "is_available": True,
                "acoustic_features": {"f0_mean": 280.0, "jitter": 0.04, "shimmer": 0.07},
                "primary_emotion": "fear",
                "valence": -0.8,
                "arousal": 0.9,
                "confidence": 0.95
            },
            "provider": "weighted_rule"
        }
        resp = await ac.post("/api/v1/fusion/fuse", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["modality_mask"]["voice"] is True
        assert data["discrepancy"]["detected"] is True
        assert "acoustic_semantic_masking" in data["discrepancy"]["discrepancy_types"]
        # Fused valence should be negative due to voice contribution
        assert data["fused_valence"] < 0.0


@pytest.mark.asyncio
async def test_fusion_fuse_learnable_gmu_provider():
    """Test POST /api/v1/fusion/fuse using the learnable GMU provider."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": {
                "session_id": "sess_fusion_gmu_1",
                "user_id": "user_fusion_gmu",
                "text": "Everything is feeling overwhelming right now."
            },
            "emotion": {
                "primary_emotion": "sadness",
                "confidence": 0.88,
                "valence": -0.6,
                "arousal": 0.7,
                "dominance": -0.4
            },
            "provider": "learnable_gmu"
        }
        resp = await ac.post("/api/v1/fusion/fuse", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["fusion_method"] == "learnable_gmu"
        assert -1.0 <= data["fused_valence"] <= 1.0
        assert 0.0 <= data["fused_distress_score"] <= 1.0


@pytest.mark.asyncio
async def test_fusion_evaluate_endpoint():
    """Test GET /api/v1/fusion/evaluate runs comparative benchmark across 4 regimes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/fusion/evaluate?num_samples=10")
        assert resp.status_code == 200
        report = resp.json()

        assert "models" in report
        assert "text_only_baseline" in report["models"]
        assert "weighted_rule_fusion" in report["models"]
        assert "learnable_gmu" in report["models"]
        assert report["total_eval_samples"] == 40  # 4 regimes * 10
        assert report["multimodal_improves_performance"] is True


@pytest.mark.asyncio
async def test_fusion_weights_endpoint():
    """Test GET /api/v1/fusion/weights returns active priors and configuration."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/fusion/weights")
        assert resp.status_code == 200
        data = resp.json()

        assert "base_domain_priors" in data
        assert "emotion" in data["base_domain_priors"]
        assert "discrepancy_types_supported" in data
        assert len(data["discrepancy_types_supported"]) >= 3


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_fused_state():
    """Verify that ZenovaOrchestrator executes multimodal fusion and returns fused_state in turn output."""
    await init_db()
    orchestrator = ZenovaOrchestrator()

    user_in = UserInput(
        session_id="sess_orch_fusion_test",
        user_id="user_orch_fusion",
        text="I am feeling a little exhausted from everything this week."
    )
    result = await orchestrator.process_turn(user_in)

    assert "fused_state" in result
    fused = result["fused_state"]
    assert fused is not None
    assert "primary_affect" in fused
    assert "fused_valence" in fused
    assert "fused_distress_score" in fused
    assert "modality_weights" in fused
    assert "modality_mask" in fused
    assert fused["modality_mask"]["text"] is True

    # Verify context block also carries fused state metadata
    assert result["context"] is not None
    meta = result["context"]["metadata"]
    assert "fused_multimodal_state" in meta
    assert meta["fused_multimodal_state"] is not None
