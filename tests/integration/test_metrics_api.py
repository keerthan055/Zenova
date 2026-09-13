"""Integration tests for Prometheus metrics scraping endpoint."""
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_prometheus_format(client):
    """Verify GET /metrics returns valid Prometheus exposition text."""
    # First make a normal call to ensure traffic is recorded in counters
    client.get("/health/live")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    # Verify standard Prometheus metric lines
    assert "# HELP zenova_uptime_seconds" in text
    assert "# TYPE zenova_uptime_seconds gauge" in text
    assert "zenova_uptime_seconds" in text

    assert "# HELP zenova_active_requests" in text
    assert "zenova_active_requests" in text

    assert "# HELP zenova_http_requests_total" in text
    assert "zenova_http_requests_total" in text

    assert "# HELP zenova_crisis_escalations_total" in text
    assert "zenova_crisis_escalations_total" in text

    assert "# HELP zenova_safety_interventions_total" in text
    assert "zenova_safety_interventions_total" in text
