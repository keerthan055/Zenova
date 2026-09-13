"""Integration tests for Kubernetes liveness, readiness, and system health endpoints."""
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_live_endpoint(client):
    """Verify GET /health/live returns HTTP 200 alive for Kubernetes liveness probes."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


def test_health_ready_endpoint(client):
    """Verify GET /health/ready checks database, models, vector store, and disk space."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["ready"] is True
    assert "checks" in data
    assert data["checks"]["database"] is True
    assert data["checks"]["models"] is True
    assert data["checks"]["disk_space"] is True


def test_health_overview_endpoint(client):
    """Verify GET /health returns service status, environment, and strict safety mode."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ZENOVA"
    assert "environment" in data
    assert data["safety_strict_mode"] is True


def test_status_endpoint(client):
    """Verify GET /status lists active model providers."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "active_providers" in data
