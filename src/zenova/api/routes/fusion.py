"""API routes for multimodal fusion, discrepancy detection, and comparative evaluation."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
)
from zenova.schemas.fusion import FusedMultimodalState, ComparativeEvaluationReport
from zenova.fusion.engine import UnifiedMultimodalFusionEngine
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.fusion")

router = APIRouter(prefix="/api/v1/fusion", tags=["Multimodal Fusion"])

_engine: Optional[UnifiedMultimodalFusionEngine] = None


def get_fusion_engine(provider: str = "weighted_rule") -> UnifiedMultimodalFusionEngine:
    global _engine
    if _engine is None or _engine.provider != provider:
        _engine = UnifiedMultimodalFusionEngine(provider=provider)
    return _engine


class FusionRequest(BaseModel):
    user_input: Optional[UserInput] = None
    emotion: Optional[EmotionResult] = None
    symptoms: Optional[SymptomResult] = None
    risk: Optional[RiskResult] = None
    voice: Optional[VoiceResult] = None
    behavior: Optional[BehavioralResult] = None
    baseline: Optional[BaselineResult] = None
    history: Optional[List[Dict[str, Any]]] = None
    provider: Optional[str] = Field(default="weighted_rule", description="'weighted_rule' or 'learnable_gmu'")


@router.post("/fuse", response_model=FusedMultimodalState)
async def fuse_modalities(req: FusionRequest) -> FusedMultimodalState:
    """Fuse heterogeneous multimodal inputs into a unified clinical affective assessment."""
    try:
        engine = get_fusion_engine(provider=req.provider or "weighted_rule")
        fused_state = engine.fuse(
            user_input=req.user_input,
            emotion=req.emotion,
            symptoms=req.symptoms,
            risk=req.risk,
            voice=req.voice,
            behavior=req.behavior,
            baseline=req.baseline,
            history=req.history
        )
        return fused_state
    except Exception as e:
        logger.error(f"Multimodal fusion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal fusion failed: {str(e)}")


@router.get("/evaluate", response_model=ComparativeEvaluationReport)
async def evaluate_multimodal_fusion(
    num_samples: int = Query(default=30, ge=5, le=100, description="Evaluation samples per modality regime")
) -> ComparativeEvaluationReport:
    """Execute the comparative evaluation benchmark comparing Text-Only, Rule Fusion, and Learnable GMU."""
    try:
        engine = get_fusion_engine()
        report = engine.evaluate(num_samples_per_regime=num_samples)
        return report
    except Exception as e:
        logger.error(f"Multimodal evaluation benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/weights")
async def get_fusion_weights() -> Dict[str, Any]:
    """Inspect active domain prior weights and discrepancy configuration."""
    engine = get_fusion_engine()
    return {
        "active_provider": engine.provider,
        "base_domain_priors": engine.rule_engine.domain_priors,
        "discrepancy_types_supported": [
            "acoustic_semantic_masking",
            "behavioral_verbal_masking",
            "baseline_affect_discrepancy",
            "symptom_affect_incongruity"
        ],
        "regimes_supported": [
            "text_only",
            "text_voice",
            "text_behavior",
            "full_multimodal"
        ]
    }
