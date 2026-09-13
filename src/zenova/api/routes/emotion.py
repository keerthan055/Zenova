"""Dedicated text emotion detection endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from zenova.emotion.inference import EmotionInferenceEngine

router = APIRouter(prefix="/api/v1/emotion", tags=["Emotion Detection"])
engine = EmotionInferenceEngine()


class EmotionAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw input text to analyze for emotional signals")


@router.post("/analyze")
async def analyze_emotion(req: EmotionAnalyzeRequest) -> Dict[str, Any]:
    """Analyze emotion for an input text utterance returning ranked emotions and confidence scores.
    
    Adheres strictly to non-diagnostic wellbeing boundaries.
    """
    try:
        result = engine.predict(req.text)
        return {
            "emotions": result["emotions"],
            "primary_emotion": result["primary_emotion"],
            "valence": result["valence"],
            "arousal": result["arousal"],
            "dominance": result["dominance"],
            "model_version": result["model_version"],
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")
