"""Unit tests for security headers and token-bucket rate limiting middleware."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from zenova.api.middleware.security import SecurityHeadersMiddleware, RateLimitingMiddleware


def test_security_headers_middleware():
    """Verify security headers are attached to responses."""
    test_app = FastAPI()
    test_app.add_middleware(SecurityHeadersMiddleware)

    @test_app.get("/ping")
    def ping():
        return {"ping": "pong"}

    client = TestClient(test_app)
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert "Content-Security-Policy" in response.headers


def test_rate_limiting_middleware_burst_and_rejection():
    """Verify rate limiter allows requests within burst limit and rejects excess with 429."""
    test_app = FastAPI()
    # Configure tight rate limit: 60 rpm, burst of 3
    test_app.add_middleware(RateLimitingMiddleware, requests_per_minute=60, burst_limit=3)

    @test_app.get("/test-rate")
    def test_rate():
        return {"status": "ok"}

    client = TestClient(test_app)

    # First 3 requests should succeed (burst capacity = 3)
    r1 = client.get("/test-rate")
    assert r1.status_code == 200
    assert "X-RateLimit-Remaining" in r1.headers

    r2 = client.get("/test-rate")
    assert r2.status_code == 200

    r3 = client.get("/test-rate")
    assert r3.status_code == 200

    # 4th request exceeds burst capacity and should return 429 Too Many Requests
    r4 = client.get("/test-rate")
    assert r4.status_code == 429
    assert r4.json()["error"] == "RateLimitExceeded"
    assert "Retry-After" in r4.headers


def test_rate_limiting_exempt_routes():
    """Verify internal diagnostic routes like /health and /metrics are exempt from rate limiting."""
    test_app = FastAPI()
    test_app.add_middleware(RateLimitingMiddleware, requests_per_minute=1, burst_limit=1)

    @test_app.get("/health/live")
    def live():
        return {"status": "alive"}

    client = TestClient(test_app)
    # Even after multiple rapid calls, /health/live should never return 429
    for _ in range(5):
        res = client.get("/health/live")
        assert res.status_code == 200
