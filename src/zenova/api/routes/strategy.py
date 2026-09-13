"""API routes for emotional support strategy recommendation and taxonomy."""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from zenova.schemas.strategy import SupportStrategy, DialogStage
from zenova.strategy.inference import StrategyInferenceEngine
from zenova.strategy.taxonomy import StrategyTaxonomy
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.strategy")

router = APIRouter(prefix="/api/v1/strategy", tags=["Support Strategy Planner"])

# Lazily initialized or shared engine
_engine: Optional[StrategyInferenceEngine] = None


def get_engine() -> StrategyInferenceEngine:
    global _engine
    if _engine is None:
        _engine = StrategyInferenceEngine(model_dir="models/strategy", use_transformer=True)
    return _engine


class StrategyPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Current user input text or sequential conversational turn")
    emotion: Optional[str] = Field(default=None, description="Inferred or acoustic emotion signal")
    situation: Optional[str] = Field(default=None, description="Underlying user situation or context")
    problem_type: Optional[str] = Field(default=None, description="Domain category of the problem (e.g. academic, relational)")
    stage: Optional[DialogStage] = Field(default=None, description="Hill's helping model stage")


@router.post("/predict")
async def predict_strategy(req: StrategyPredictRequest) -> Dict[str, Any]:
    """Recommend an emotional support strategy from conversational and multimodal inputs."""
    try:
        engine = get_engine()
        result = engine.predict_strategy(
            text=req.text,
            emotion=req.emotion,
            situation=req.situation,
            problem_type=req.problem_type,
            stage=req.stage
        )
        return {
            "selected_strategy": result["selected_strategy"].value if hasattr(result["selected_strategy"], "value") else str(result["selected_strategy"]),
            "confidence": result["confidence"],
            "stage": result["stage"].value if hasattr(result["stage"], "value") else str(result["stage"]),
            "probabilities": result["probabilities"],
            "ranked_strategies": result["ranked_strategies"],
            "alternatives": result["alternatives"],
            "rationale": result["rationale"],
            "model_version": result["model_version"]
        }
    except Exception as e:
        logger.error(f"Strategy recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy prediction error: {str(e)}")


@router.get("/taxonomy")
async def get_taxonomy() -> Dict[str, Any]:
    """Retrieve current configurable emotional support strategy taxonomy and descriptions."""
    engine = get_engine()
    return engine.taxonomy.to_dict()


@router.get("/ablation-results")
async def get_ablation_results() -> Dict[str, Any]:
    """Retrieve controlled ablation experiment metrics (Conditions A vs B vs C for Baseline and Transformer)."""
    p = Path("experiments/strategy_ablation_results.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Ablation results not yet generated. Run training first.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
