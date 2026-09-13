"""Prometheus operational metrics collector and telemetry middleware for ZENOVA."""
import time
from collections import defaultdict
from typing import Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse


class MetricsRegistry:
    """Thread-safe in-memory metrics registry formatted for Prometheus scraping."""

    def __init__(self):
        # request counts: (method, endpoint, status_code) -> count
        self.request_counts: Dict[str, int] = defaultdict(int)
        # request durations: (method, endpoint) -> [durations]
        self.request_durations: Dict[str, float] = defaultdict(float)
        # active requests gauge
        self.active_requests: int = 0
        # clinical event counters
        self.crisis_escalations_total: int = 0
        self.safety_interventions: Dict[str, int] = defaultdict(int)
        self.start_time: float = time.time()

    def record_request(self, method: str, path: str, status_code: int, duration_sec: float) -> None:
        norm_path = self._normalize_path(path)
        key = f'{method}:{norm_path}:{status_code}'
        self.request_counts[key] += 1
        dur_key = f'{method}:{norm_path}'
        self.request_durations[dur_key] += duration_sec

    def record_crisis_escalation(self) -> None:
        self.crisis_escalations_total += 1

    def record_safety_action(self, action: str) -> None:
        self.safety_interventions[action] += 1

    def _normalize_path(self, path: str) -> str:
        # Group dynamic paths
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "api":
            # e.g., /api/v1/orchestrator/traces/123 -> /api/v1/orchestrator/traces/{id}
            if parts[1] == "v1" and len(parts) > 3:
                return f"/api/v1/{parts[2]}/{parts[3]}"
        return path

    def export_prometheus_text(self) -> str:
        """Render metrics in official Prometheus text exposition format."""
        lines = [
            "# HELP zenova_uptime_seconds Total runtime of ZENOVA instance in seconds",
            "# TYPE zenova_uptime_seconds gauge",
            f"zenova_uptime_seconds {time.time() - self.start_time:.2f}",
            "",
            "# HELP zenova_active_requests Currently in-flight HTTP requests",
            "# TYPE zenova_active_requests gauge",
            f"zenova_active_requests {self.active_requests}",
            "",
            "# HELP zenova_http_requests_total Total number of HTTP requests processed",
            "# TYPE zenova_http_requests_total counter"
        ]

        for k, count in self.request_counts.items():
            method, path, status = k.split(":")
            lines.append(f'zenova_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')

        lines.extend([
            "",
            "# HELP zenova_http_request_duration_seconds_total Total seconds spent processing HTTP requests",
            "# TYPE zenova_http_request_duration_seconds_total counter"
        ])
        for k, total_dur in self.request_durations.items():
            method, path = k.split(":")
            lines.append(f'zenova_http_request_duration_seconds_total{{method="{method}",path="{path}"}} {total_dur:.4f}')

        lines.extend([
            "",
            "# HELP zenova_crisis_escalations_total Total clinical crisis escalations triggered",
            "# TYPE zenova_crisis_escalations_total counter",
            f"zenova_crisis_escalations_total {self.crisis_escalations_total}",
            "",
            "# HELP zenova_safety_interventions_total Total safety gate actions executed",
            "# TYPE zenova_safety_interventions_total counter"
        ])
        for action, count in self.safety_interventions.items():
            lines.append(f'zenova_safety_interventions_total{{action="{action}"}} {count}')

        return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsRegistry()


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware measuring request durations, statuses, and concurrency for Prometheus."""

    async def dispatch(self, request: Request, call_next):
        GLOBAL_METRICS.active_requests += 1
        t0 = time.time()
        try:
            response: Response = await call_next(request)
            duration = time.time() - t0
            GLOBAL_METRICS.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_sec=duration
            )
            return response
        finally:
            GLOBAL_METRICS.active_requests = max(0, GLOBAL_METRICS.active_requests - 1)


async def metrics_endpoint_handler(request: Request) -> Response:
    """Endpoint serving raw Prometheus metrics for scraping."""
    content = GLOBAL_METRICS.export_prometheus_text()
    return PlainTextResponse(content=content, media_type="text/plain; version=0.0.4")
