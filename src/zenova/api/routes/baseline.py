"""Personal baseline evaluation, observation ingestion, and longitudinal tracking routes."""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from zenova.baseline.engine import PersonalBaselineEngine
from zenova.schemas.baseline import (
    UserBaselineProfile,
    BaselineDeviationReport,
    MultimodalObservation,
    BaselineStatus
)
from zenova.models.registry import ModelRegistry
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.baseline")
router = APIRouter(prefix="/api/v1/baseline", tags=["Personal Baseline"])

# Singleton engine instance
_baseline_engine: Optional[PersonalBaselineEngine] = None


def get_baseline_engine() -> PersonalBaselineEngine:
    global _baseline_engine
    if _baseline_engine is None:
        registry = ModelRegistry()
        engine = registry.get_module_instance("baseline")
        if isinstance(engine, PersonalBaselineEngine):
            _baseline_engine = engine
        else:
            _baseline_engine = PersonalBaselineEngine()
    return _baseline_engine


class ResetBaselineRequest(BaseModel):
    user_id: str


@router.get("/{user_id}", response_model=Dict[str, Any])
async def get_user_baseline(
    user_id: str,
    engine: PersonalBaselineEngine = Depends(get_baseline_engine)
):
    """Retrieve the current personal baseline profile and tracking statistics for a user."""
    profile = engine.storage.get_profile(user_id)
    if not profile:
        return {
            "user_id": user_id,
            "status": BaselineStatus.INSUFFICIENT_DATA.value,
            "confidence": 0.0,
            "total_observations": 0,
            "message": "No historical baseline data recorded for this user yet.",
            "feature_baselines": {}
        }

    return {
        "user_id": user_id,
        "status": profile.status,
        "confidence": profile.confidence,
        "total_observations": profile.total_interactions_recorded,
        "baseline_valence_mean": profile.baseline_valence_mean,
        "baseline_valence_std": profile.baseline_valence_std,
        "established_at": profile.established_at.isoformat(),
        "last_updated_at": profile.last_updated_at.isoformat(),
        "feature_baselines": {
            feat: stats.model_dump()
            for feat, stats in profile.feature_baselines.items()
        }
    }


@router.post("/evaluate", response_model=BaselineDeviationReport)
async def evaluate_observation(
    observation: MultimodalObservation,
    engine: PersonalBaselineEngine = Depends(get_baseline_engine)
):
    """Evaluate an inbound multimodal observation against the user's personal baseline.
    
    Returns statistical deviations (z-scores), qualitative interpretations, and
    a non-diagnostic clinical disclaimer.
    """
    try:
        report = engine.evaluate_observation(observation)
        return report
    except Exception as e:
        logger.error(f"Error evaluating baseline for user {observation.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Baseline evaluation failed: {str(e)}")


@router.post("/observation")
async def record_observation(
    observation: MultimodalObservation,
    engine: PersonalBaselineEngine = Depends(get_baseline_engine)
):
    """Record an out-of-band multimodal observation and update longitudinal baseline."""
    try:
        report = engine.evaluate_observation(observation)
        return {
            "status": "recorded",
            "user_id": observation.user_id,
            "total_observations": report.total_observations + 1,
            "baseline_status": report.status,
            "confidence": report.confidence
        }
    except Exception as e:
        logger.error(f"Error recording observation for user {observation.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recording observation failed: {str(e)}")


@router.post("/reset")
async def reset_user_baseline(
    request: ResetBaselineRequest,
    engine: PersonalBaselineEngine = Depends(get_baseline_engine)
):
    """Reset baseline records and history for a specific user (for tests or patient consent)."""
    try:
        engine.storage.reset_user(request.user_id)
        from zenova.db.session import get_db_session
        from zenova.db.repositories import BaselineRepository

        async with get_db_session() as db:
            repo = BaselineRepository(db)
            await repo.reset_baseline(request.user_id)

        return {
            "status": "success",
            "message": f"Reset personal baseline for user '{request.user_id}'."
        }
    except Exception as e:
        logger.error(f"Error resetting baseline for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resetting baseline failed: {str(e)}")
