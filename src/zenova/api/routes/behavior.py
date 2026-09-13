"""Passive behavioral sensing, wearable telemetry ingestion, and disruption evaluation routes."""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from zenova.schemas.behavior import (
    BehavioralObservation,
    BehavioralAnomalyReport,
    DeviceType,
)
from zenova.schemas.standard import BehavioralResult
from zenova.behavior.analyzer import BehavioralAnalyzer
from zenova.behavior.wearable import WearableAdapterRegistry
from zenova.models.registry import ModelRegistry
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.behavior")
router = APIRouter(prefix="/api/v1/behavior", tags=["Behavioral & Passive Sensing"])

# Singleton analyzer instance
_behavior_analyzer: Optional[BehavioralAnalyzer] = None


def get_behavior_analyzer() -> BehavioralAnalyzer:
    global _behavior_analyzer
    if _behavior_analyzer is None:
        try:
            registry = ModelRegistry()
            analyzer = registry.get_module_instance("behavior")
            if isinstance(analyzer, BehavioralAnalyzer):
                _behavior_analyzer = analyzer
            else:
                _behavior_analyzer = BehavioralAnalyzer()
        except Exception:
            _behavior_analyzer = BehavioralAnalyzer()
    return _behavior_analyzer


class RawWearableIngestRequest(BaseModel):
    user_id: str
    device_type: Optional[str] = "generic_wearable"
    payload: Dict[str, Any] = Field(default_factory=dict)


@router.get("/devices", response_model=List[str])
async def list_supported_devices():
    """List all supported wearable and passive sensing device formats."""
    return [d.value for d in DeviceType]


@router.post("/observation", response_model=BehavioralAnomalyReport)
async def record_observation(
    observation: BehavioralObservation,
    analyzer: BehavioralAnalyzer = Depends(get_behavior_analyzer),
):
    """Ingest a validated behavioral observation and evaluate multi-feature disruptions.
    
    Observational disclaimer attached. Enforces single-feature non-inference constraint.
    """
    try:
        report = analyzer.record_observation(observation)

        # Also persist to database if available
        try:
            from zenova.db.session import get_db_session
            from zenova.db.repositories import BehaviorRepository

            async with get_db_session() as db:
                repo = BehaviorRepository(db)
                await repo.record_observation(
                    user_id=observation.user_id,
                    metrics_dict=observation.model_dump(mode="json"),
                    source_device=observation.source_device.value,
                    timestamp=observation.timestamp,
                )
        except Exception as db_err:
            logger.warning(f"Database persistence for behavioral observation skipped: {db_err}")

        return report
    except Exception as e:
        logger.error(f"Error ingesting behavioral observation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Observation ingestion failed: {str(e)}")


@router.post("/ingest-raw", response_model=BehavioralAnomalyReport)
async def ingest_raw_device_telemetry(
    request: RawWearableIngestRequest,
    analyzer: BehavioralAnalyzer = Depends(get_behavior_analyzer),
):
    """Ingest raw vendor-specific telemetry payload (Apple HealthKit, Fitbit, etc.)."""
    try:
        raw_data = request.payload.copy()
        raw_data["user_id"] = request.user_id
        report = analyzer.record_raw_payload(raw_data, device_key=request.device_type)
        return report
    except Exception as e:
        logger.error(f"Error ingesting raw device payload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Raw payload ingestion failed: {str(e)}")


@router.get("/{user_id}/summary", response_model=Dict[str, Any])
async def get_user_behavioral_summary(
    user_id: str,
    analyzer: BehavioralAnalyzer = Depends(get_behavior_analyzer),
):
    """Retrieve longitudinal behavioral baseline summary and tracked distributions for a user."""
    profile = analyzer.profiles.get(user_id)
    if not profile:
        return {
            "user_id": user_id,
            "is_established": False,
            "confidence": 0.0,
            "observation_count": 0,
            "message": "No passive behavioral data recorded for this user yet.",
            "tracked_features": {},
        }

    return {
        "user_id": user_id,
        "is_established": profile.is_established,
        "confidence": profile.confidence,
        "observation_count": profile.observation_count,
        "tracked_features": {
            k: {
                "mean": dist.mean,
                "std": dist.std,
                "count": dist.count,
                "min": dist.min_val,
                "max": dist.max_val,
            }
            for k, dist in profile.distributions.items()
        },
        "disclaimer": "Passive behavioral signals are observational proxies; do not interpret as clinical diagnoses.",
    }


@router.post("/{user_id}/clear")
async def clear_user_behavioral_data(
    user_id: str,
    analyzer: BehavioralAnalyzer = Depends(get_behavior_analyzer),
):
    """Reset and clear behavioral history and profile for a user."""
    try:
        analyzer.profiles.pop(user_id, None)
        analyzer.history.pop(user_id, None)

        try:
            from zenova.db.session import get_db_session
            from zenova.db.repositories import BehaviorRepository

            async with get_db_session() as db:
                repo = BehaviorRepository(db)
                await repo.clear_observations(user_id)
        except Exception as db_err:
            logger.warning(f"Database clearing for behavior skipped: {db_err}")

        return {
            "status": "success",
            "message": f"Cleared behavioral data and personal profile for user '{user_id}'.",
        }
    except Exception as e:
        logger.error(f"Error clearing behavioral data for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clearing behavioral data failed: {str(e)}")
