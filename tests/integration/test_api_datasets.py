"""Integration tests for Dataset API routes."""
from fastapi.testclient import TestClient
from zenova.api.app import app

client = TestClient(app)


def test_list_datasets_endpoint():
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "datasets" in data
    assert data["count"] >= 1


def test_get_dataset_detail():
    resp = client.get("/api/v1/datasets/Synthetic Wellbeing Benchmark")
    if resp.status_code != 200:
        resp = client.get("/api/v1/datasets/synthetic_wellbeing_benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert data["intended_task"] == "support_strategy_planning"
    assert "statistics" in data
