"""Dedicated mental health symptom and observational signal identification endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from zenova.symptoms.inference import SymptomInferenceEngine

router = APIRouter(prefix="/api/v1/symptoms", tags=["Symptom Identification"])
engine = SymptomInferenceEngine()


class SymptomIdentifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw input text to analyze for observational symptom signals")
    threshold: float = Field(default=0.40, ge=0.0, le=1.0, description="Detection confidence threshold")


@router.post("/identify")
async def identify_symptoms(req: SymptomIdentifyRequest) -> Dict[str, Any]:
    """Identify observable mental health symptom signals from text.
    
    Adheres strictly to clinical safety boundaries:
    - Never generates diagnostic statements (no 'You have depression').
    - Returns ranked observational markers with confidence and evidence spans.
    - Flags crisis signals for downstream escalation.
    """
    try:
        result = engine.predict(req.text, threshold=req.threshold)
        return {
            "signals": [
                {
                    "label": s["label"],
                    "confidence": s["confidence"],
                    "severity": s["severity"],
                    "evidence_spans": s["evidence_spans"]
                }
                for s in result["signals"]
            ],
            "aggregate_severity": result["aggregate_severity"],
            "is_crisis_flagged": result["is_crisis_flagged"],
            "disclaimer": result["disclaimer"],
            "notes": result["notes"],
            "model_version": result["model_version"],
            "timestamp": result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom identification failed: {str(e)}")
