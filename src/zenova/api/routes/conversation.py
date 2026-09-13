"""Conversation turn processing and session history endpoints."""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from zenova.schemas.standard import UserInput
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.db.session import get_db_session
from zenova.db.repositories import TurnRepository

router = APIRouter(prefix="/api/v1/conversation", tags=["Conversation"])
orchestrator = ZenovaOrchestrator()


@router.post("/turn")
async def process_turn(user_input: UserInput) -> Dict[str, Any]:
    """Process an inbound conversational turn through the safety-gated pipeline."""
    try:
        result = await orchestrator.process_turn(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve stored conversational history for a session."""
    async with get_db_session() as db:
        repo = TurnRepository(db)
        history = await repo.get_history(session_id)
        return {
            "session_id": session_id,
            "turns_count": len(history),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "speaker": t.speaker,
                    "content": t.content,
                    "created_at": t.created_at.isoformat() if t.created_at else None
                }
                for t in history
            ]
        }
