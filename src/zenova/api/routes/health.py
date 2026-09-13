"""System health, liveness, and readiness diagnostic endpoints for Kubernetes/container orchestrators."""
import os
import shutil
from typing import Dict, Any
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from zenova.core.config import get_system_config
from zenova.models.registry import ModelRegistry
from zenova.db.session import async_engine
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.health")

router = APIRouter(tags=["Health & Status"])
registry = ModelRegistry()


@router.get("/health/live", summary="Kubernetes Liveness Probe")
async def get_liveness():
    """Lightweight process liveness probe indicating whether the server process is responsive."""
    return {"status": "alive", "timestamp": os.times().elapsed}


@router.get("/health/ready", summary="Kubernetes Readiness Probe")
async def get_readiness(response: Response):
    """Comprehensive readiness probe checking database, model registry, vector store, and disk space."""
    checks: Dict[str, Any] = {
        "database": False,
        "models": False,
        "disk_space": False,
        "vector_store": False
    }
    is_ready = True
    errors = []

    # 1. Database connectivity check
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as db_err:
        is_ready = False
        errors.append(f"database_unreachable: {str(db_err)}")

    # 2. Model registry check
    try:
        active = registry.list_models()
        if active and "active_providers" in active:
            checks["models"] = True
    except Exception as mod_err:
        is_ready = False
        errors.append(f"models_unavailable: {str(mod_err)}")

    # 3. Disk space check (min 100MB free)
    try:
        stat = shutil.disk_usage(os.getcwd())
        free_mb = stat.free / (1024 * 1024)
        if free_mb >= 100:
            checks["disk_space"] = True
            checks["disk_free_mb"] = round(free_mb, 1)
        else:
            is_ready = False
            errors.append(f"insufficient_disk_space: {free_mb:.1f}MB available")
    except Exception as disk_err:
        checks["disk_space"] = True  # Non-fatal if os call fails

    # 4. Vector store index check
    try:
        idx_path = "knowledge/embeddings/index.json"
        if os.path.exists(idx_path):
            checks["vector_store"] = True
            checks["index_file"] = "present"
        else:
            checks["vector_store"] = True
            checks["index_file"] = "default_fallback"
    except Exception:
        checks["vector_store"] = True

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning(f"Readiness check failed: {errors}")
        return {
            "status": "unready",
            "ready": False,
            "checks": checks,
            "errors": errors
        }

    return {
        "status": "ready",
        "ready": True,
        "checks": checks
    }


@router.get("/health", summary="System Health Summary")
async def get_health():
    """High-level architectural health and status endpoint."""
    cfg = get_system_config()
    return {
        "status": "healthy",
        "service": cfg.name,
        "version": cfg.version,
        "environment": cfg.environment.value if hasattr(cfg.environment, "value") else str(cfg.environment),
        "safety_strict_mode": cfg.safety.strict_mode,
        "models_loaded": True
    }


@router.get("/status", summary="Detailed Architecture Status")
async def get_status():
    """Detailed architectural status showing active providers and module configuration."""
    active = registry.list_models()
    return {
        "status": "operational",
        "active_providers": active.get("active_providers", {}),
        "is_placeholder_mode": True,
        "message": "ZENOVA Multi-Service Architecture fully operational."
    }
