"""Dedicated crisis and suicide risk detection endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from zenova.risk.inference import RiskInferenceEngine

router = APIRouter(prefix="/api/v1/risk", tags=["Crisis & Risk Detection"])
engine = RiskInferenceEngine()


class RiskDetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw conversational text to analyze for crisis and self-harm risk")
    crisis_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Decision threshold for escalating to high risk")


@router.post("/detect")
async def detect_risk(req: RiskDetectRequest) -> Dict[str, Any]:
    """Detect crisis, self-harm, and suicidal ideation risk from text.
    
    Adheres strictly to safety triage protocols:
    - Categorizes text into C-SSRS 4-tier taxonomy (low, moderate, high, critical).
    - Flags requires_escalation = True for high and critical risk.
    - Extracts verbatim trigger cues.
    - Does NOT represent clinical certainty; life-safety risks must be routed to human crisis professionals.
    """
    try:
        result = engine.predict(req.text, crisis_threshold=req.crisis_threshold)
        return {
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "crisis_category": result["crisis_category"],
            "signals": result["signals"],
            "trigger_cues": result["trigger_cues"],
            "requires_escalation": result["requires_escalation"],
            "action": result["action"],
            "disclaimer": result["disclaimer"],
            "model_version": result["model_version"],
            "probabilities": result.get("probabilities", {}),
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crisis risk detection failed: {str(e)}")
