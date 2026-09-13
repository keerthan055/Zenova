"""Integration tests for ZENOVA Evaluation Framework REST API endpoints."""
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_get_human_evaluation_protocol(client):
    """Verify GET /api/v1/evaluation/protocol returns standard clinical Likert scales."""
    response = client.get("/api/v1/evaluation/protocol")
    assert response.status_code == 200
    data = response.json()

    assert data["protocol_version"] == "1.0.0"
    assert "rubrics" in data
    assert len(data["rubrics"]) == 5

    dim_names = [s["dimension"] for s in data["rubrics"]]
    assert "Support Strategy Fidelity" in dim_names
    assert "Clinical Safety & Non-Harm" in dim_names
    assert "Relevance & Context Sensitivity" in dim_names
    assert "Empathy & Warmth" in dim_names
    assert "Actionability & Pacing" in dim_names

    # Check anchors and rubric descriptions
    for rubric in data["rubrics"]:
        assert rubric["scale_1_unacceptable"]
        assert rubric["scale_3_acceptable"]
        assert rubric["scale_5_exemplary"]


def test_api_get_evaluation_report(client):
    """Verify GET /api/v1/evaluation/report returns full unified evaluation report."""
    response = client.get("/api/v1/evaluation/report")
    assert response.status_code == 200
    data = response.json()

    assert "report_id" in data
    assert "timestamp" in data
    assert "model_metrics" in data
    assert "generation_metrics" in data
    assert "system_performance" in data
    assert "ablation_study" in data
    assert "human_evaluation_protocol" in data

    # Verify model metrics structure
    models = data["model_metrics"]
    assert "emotion" in models
    assert "symptoms" in models
    assert "risk" in models
    assert "strategy" in models

    assert models["emotion"]["accuracy"] >= 0.40
    assert models["risk"]["sensitivity_high_critical"] >= 0.95
    assert models["symptoms"]["micro_f1"] > 0.7


def test_api_get_ablation_study(client):
    """Verify GET /api/v1/evaluation/ablation returns 5-stage ablation comparison."""
    response = client.get("/api/v1/evaluation/ablation")
    assert response.status_code == 200
    data = response.json()

    assert "configurations" in data
    assert len(data["configurations"]) == 5
    assert "key_findings" in data
    assert len(data["key_findings"]) >= 4

    config_ids = [c["config_id"] for c in data["configurations"]]
    assert config_ids == ["A", "B", "C", "D", "E"]

    # Verify Complete system config E properties
    config_e = data["configurations"][4]
    assert config_e["config_id"] == "E"
    assert config_e["safety_interception_rate"] == 100.0
    assert config_e["hallucination_rate_pct"] == 0.0


def test_api_trigger_evaluation_run(client):
    """Verify POST /api/v1/evaluation/run executes full evaluation and returns report."""
    response = client.post("/api/v1/evaluation/run")
    assert response.status_code == 200
    data = response.json()

    assert "report_id" in data
    assert "model_metrics" in data
    assert "ablation_study" in data
    assert len(data["ablation_study"]["configurations"]) == 5
