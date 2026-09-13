"""Main FastAPI application for ZENOVA with production middleware, metrics, and security."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from zenova.core.logging import get_logger
from zenova.core.config import get_system_config
from zenova.db.session import init_db
from zenova.core.exceptions import ZenovaException
from zenova.api.middleware.security import SecurityHeadersMiddleware, RateLimitingMiddleware
from zenova.api.middleware.metrics import PrometheusMetricsMiddleware, metrics_endpoint_handler

# Route imports
from zenova.api.routes.health import router as health_router
from zenova.api.routes.conversation import router as conversation_router
from zenova.api.routes.escalation import router as escalation_router
from zenova.api.routes.models import router as models_router
from zenova.api.routes.datasets import router as datasets_router
from zenova.api.routes.emotion import router as emotion_router
from zenova.api.routes.symptoms import router as symptoms_router
from zenova.api.routes.risk import router as risk_router
from zenova.api.routes.baseline import router as baseline_router
from zenova.api.routes.behavior import router as behavior_router
from zenova.api.routes.voice import router as voice_router
from zenova.api.routes.context import router as context_router
from zenova.api.routes.strategy import router as strategy_router
from zenova.api.routes.generation import router as generation_router
from zenova.api.routes.rag import router as rag_router
from zenova.api.routes.safety import router as safety_router
from zenova.api.routes.dashboard import router as dashboard_router
from zenova.api.routes.fusion import router as fusion_router
from zenova.api.routes.orchestrator import router as orchestrator_router
from zenova.api.routes.evaluation import router as evaluation_router
from zenova.api.routes.user import router as user_router

logger = get_logger("zenova.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycles."""
    logger.info("Initializing ZENOVA database tables...")
    await init_db()
    logger.info("ZENOVA backend initialized successfully.")
    yield
    logger.info("ZENOVA backend shutting down.")


cfg = get_system_config()

OPENAPI_TAGS = [
    {"name": "Health & Status", "description": "Liveness, readiness, and architectural diagnostic probes."},
    {"name": "Orchestrator", "description": "Central conversational pipeline execution and trace inspection."},
    {"name": "Safety", "description": "Independent response safety gate verification and audit logging."},
    {"name": "Escalation", "description": "Human-in-the-loop clinician escalation and alert triaging."},
    {"name": "Evaluation", "description": "Unified model, generation, system, and ablation benchmarks."},
    {"name": "Clinician Dashboard", "description": "Clinical analytics, risk timelines, and longitudinal charts."},
    {"name": "Multimodal Fusion", "description": "Acoustic, behavioral, and text feature integration."},
    {"name": "User Application", "description": "User-facing conversational experience, wellbeing check-ins, privacy controls, and support resources."}
]

app = FastAPI(
    title="ZENOVA Platform API",
    description="Research-Grade Mental-Health Wellbeing and Conversational Support Platform API",
    version=cfg.version,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    contact={
        "name": "ZENOVA Engineering & Clinical Team",
        "url": "https://zenova.ai",
        "email": "support@zenova.ai"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# 1. Prometheus Metrics Middleware (outermost for accurate HTTP timing)
app.add_middleware(PrometheusMetricsMiddleware)

# 2. Defensive Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# 3. Rate Limiting Middleware (configurable per minute & burst)
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=cfg.security.rate_limit_per_minute,
    burst_limit=cfg.security.rate_limit_burst,
    bypass_testclient=True
)

# 4. CORS Middleware
cors_origins = cfg.security.cors_allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ZenovaException)
async def zenova_exception_handler(request: Request, exc: ZenovaException):
    logger.error(f"Domain exception: {exc.message} (details: {exc.details})")
    return JSONResponse(
        status_code=400,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details}
    )


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root path directly to the user-facing web interface."""
    return RedirectResponse(url="/app")


# Metrics endpoint
app.add_api_route("/metrics", metrics_endpoint_handler, methods=["GET"], tags=["Health & Status"])

# Include application routers
app.include_router(health_router)
app.include_router(conversation_router)
app.include_router(escalation_router)
app.include_router(models_router)
app.include_router(datasets_router)
app.include_router(emotion_router)
app.include_router(symptoms_router)
app.include_router(risk_router)
app.include_router(baseline_router)
app.include_router(behavior_router)
app.include_router(voice_router)
app.include_router(context_router)
app.include_router(strategy_router)
app.include_router(generation_router)
app.include_router(rag_router)
app.include_router(safety_router)
app.include_router(dashboard_router)
app.include_router(fusion_router)
app.include_router(orchestrator_router)
app.include_router(evaluation_router)
app.include_router(user_router)

logger.info("Loaded API routes and production middlewares successfully.")


def main():
    """CLI entrypoint to run the ZENOVA development server."""
    import uvicorn
    uvicorn.run("zenova.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
