import os
import sys
import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Applies OWASP-recommended HTTP security headers to all outbound responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Inject standard defensive headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy (allows Swagger UI assets and embedded web UI scripts)
        if request.url.path in ("/docs", "/redoc", "/openapi.json"):
            response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com;"
        elif request.url.path in ("/app", "/dashboard", "/", "/login", "/register", "/forgot-password", "/reset-password", "/verify-email"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """In-memory sliding token-bucket rate limiter with burst allowance."""

    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 15, bypass_testclient: bool = False):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.burst = burst_limit
        self.bypass_testclient = bypass_testclient
        # Storage: client_key -> (token_count, last_updated_timestamp)
        self.clients: Dict[str, Tuple[float, float]] = defaultdict(lambda: (float(self.burst), time.time()))

    def _get_client_key(self, request: Request) -> str:
        """Derive rate-limit key from authorization header or client IP."""
        auth_header = request.headers.get("Authorization")
        if auth_header and len(auth_header) > 10:
            return f"auth_{auth_header[-16:]}"
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        return f"ip_{client_ip}"

    async def dispatch(self, request: Request, call_next):
        # Allow bypassing rate limits during automated test runs when configured on app
        if self.bypass_testclient and ("pytest" in sys.modules or os.getenv("TESTING") == "true" or (request.client and request.client.host == "testclient")):
            return await call_next(request)

        # Exclude internal/diagnostic routes and static UI pages from rate limiting
        if request.url.path in ("/health", "/health/live", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json", "/app", "/dashboard", "/login", "/register", "/forgot-password", "/reset-password", "/verify-email"):
            return await call_next(request)

        client_key = self._get_client_key(request)
        now = time.time()
        tokens, last_time = self.clients[client_key]

        # Refill tokens according to elapsed time
        elapsed = now - last_time
        refill_rate = self.rpm / 60.0  # tokens per second
        tokens = min(float(self.burst), tokens + (elapsed * refill_rate))

        if tokens < 1.0:
            retry_after = max(1, int((1.0 - tokens) / refill_rate))
            logger.warning(f"Rate limit exceeded for client {client_key} on {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after_seconds": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Deduct 1 token and update state
        self.clients[client_key] = (tokens - 1.0, now)

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(int(tokens - 1.0))
        return response
