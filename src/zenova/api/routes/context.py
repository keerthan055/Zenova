"""REST API endpoints for ZENOVA Multimodal Context Engine (Step 9)."""
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from zenova.core.logging import get_logger
from zenova.db.session import get_db_session
from zenova.db.repositories import ContextRepository, TurnRepository
from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    DialogStage
)
from zenova.schemas.context import (
    MultimodalContext,
    PrivacyLevel,
    UserFeedback
)
from zenova.context.engine import MultimodalContextEngine
from zenova.context.privacy import PrivacyEngine

logger = get_logger("zenova.api.context")

router = APIRouter(prefix="/api/v1/context", tags=["Context Engine"])

_engine = MultimodalContextEngine()
_privacy = PrivacyEngine()


class ContextBuildRequest(BaseModel):
    user_input: UserInput
    emotion: Optional[EmotionResult] = None
    symptoms: Optional[SymptomResult] = None
    risk: Optional[RiskResult] = None
    baseline: Optional[BaselineResult] = None
    behavior: Optional[BehavioralResult] = None
    voice: Optional[VoiceResult] = None
    dialog_stage: DialogStage = DialogStage.EXPLORATION
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD


class AnonymizeRequest(BaseModel):
    context: Dict[str, Any]
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED


@router.post("/build", response_model=MultimodalContext)
async def build_context_endpoint(request: ContextBuildRequest):
    """Normalize heterogeneous module outputs into the unified 9-block MultimodalContext."""
    try:
        ctx = _engine.build_context(
            user_input=request.user_input,
            emotion=request.emotion,
            symptoms=request.symptoms,
            risk=request.risk,
            baseline=request.baseline,
            behavior=request.behavior,
            voice=request.voice,
            dialog_stage=request.dialog_stage,
            privacy_level=request.privacy_level
        )
        return ctx
    except Exception as e:
        logger.error(f"Failed to build multimodal context: {e}")
        raise HTTPException(status_code=500, detail=f"Context aggregation error: {str(e)}")


@router.post("/anonymize")
async def anonymize_context_endpoint(request: AnonymizeRequest):
    """Apply PII scrubbing and ID pseudonymization to a context payload."""
    try:
        sanitized, count = _privacy.apply_privacy(
            request.context,
            privacy_level=request.privacy_level
        )
        return {
            "privacy_level": request.privacy_level.value,
            "redacted_tokens_count": count,
            "anonymized_context": sanitized
        }
    except Exception as e:
        logger.error(f"Failed to anonymize context: {e}")
        raise HTTPException(status_code=500, detail=f"Anonymization error: {str(e)}")


@router.get("/{session_id}/latest")
async def get_latest_context_snapshot(session_id: str):
    """Retrieve the latest recorded MultimodalContext snapshot for a given session."""
    async with get_db_session() as db:
        repo = ContextRepository(db)
        snapshot = await repo.get_latest_snapshot(session_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail=f"No context snapshot found for session '{session_id}'")

        return {
            "context_id": snapshot.context_id,
            "session_id": snapshot.session_id,
            "turn_id": snapshot.turn_id,
            "context_hash": snapshot.context_hash,
            "privacy_level": snapshot.privacy_level,
            "created_at": snapshot.created_at.isoformat() if snapshot.created_at else None,
            "context": json.loads(snapshot.context_json)
        }


@router.post("/{session_id}/feedback")
async def record_feedback_endpoint(session_id: str, feedback: UserFeedback):
    """Record explicit user feedback on a turn and link it to the session history."""
    async with get_db_session() as db:
        repo = ContextRepository(db)
        turn = await repo.record_turn_feedback(
            session_id=session_id,
            turn_id=feedback.turn_id,
            feedback_dict=feedback.model_dump()
        )
        if not turn:
            raise HTTPException(
                status_code=404,
                detail=f"Turn {feedback.turn_id} not found for session '{session_id}'"
            )

        return {
            "status": "feedback_recorded",
            "session_id": session_id,
            "turn_id": feedback.turn_id,
            "rating": feedback.rating,
            "is_helpful": feedback.is_helpful
        }
