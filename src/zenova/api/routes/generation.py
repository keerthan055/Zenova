"""API routes for strategy-controlled response generation."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from zenova.schemas.standard import UserInput, StrategyResult, SupportStrategy, DialogStage
from zenova.generation.generator import StrategyControlledGenerator
from zenova.generation.providers import list_available_providers
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.generation")

router = APIRouter(prefix="/api/v1/generation", tags=["Response Generation"])

# Shared generator instance
_generator: Optional[StrategyControlledGenerator] = None


def get_generator() -> StrategyControlledGenerator:
    global _generator
    if _generator is None:
        _generator = StrategyControlledGenerator()
    return _generator


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Current user utterance")
    strategy: Optional[SupportStrategy] = Field(default=None, description="Explicit emotional support strategy to apply")
    emotion: Optional[str] = Field(default=None, description="Inferred emotion label")
    symptoms: Optional[List[str]] = Field(default_factory=list, description="Observational symptom signals")
    situation: Optional[str] = Field(default=None, description="Contextual user situation")
    stage: Optional[DialogStage] = Field(default=None, description="Hill's helping model stage")
    provider: Optional[str] = Field(default=None, description="Requested LLM provider backend")
    rag_context: Optional[List[str]] = Field(default_factory=list, description="Evidence-grounded psychoeducational snippets")


@router.post("/generate")
async def generate_response(req: GenerateRequest) -> Dict[str, Any]:
    """Generate an empathetic, clinically bounded conversational response strictly conditioned on the specified support strategy."""
    try:
        gen = get_generator()
        if req.provider:
            gen = StrategyControlledGenerator(provider_name=req.provider)

        strat = req.strategy or SupportStrategy.REFLECTION_OF_FEELINGS
        stage = req.stage or DialogStage.COMFORTING

        user_input = UserInput(
            session_id="api-direct-generation",
            user_id="api-user",
            text=req.text,
            metadata={"situation": req.situation, "emotion": req.emotion} if req.situation or req.emotion else {}
        )

        strategy_res = StrategyResult(
            selected_strategy=strat,
            confidence=0.85,
            stage=stage,
            rationale=f"API requested strategy: {strat.value if hasattr(strat, 'value') else strat}"
        )

        result = await gen.agenerate(
            user_input=user_input,
            strategy=strategy_res,
            rag_context=req.rag_context
        )

        return {
            "response_text": result.response_text,
            "strategy_applied": result.strategy_applied.value if hasattr(result.strategy_applied, "value") else str(result.strategy_applied),
            "model_name": result.model_name,
            "provider": result.generation_metadata.get("provider", "unknown"),
            "tokens_used": result.tokens_used,
            "validation_passed": result.validation_passed,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/providers")
async def get_providers() -> Dict[str, Any]:
    """List supported and configured LLM providers."""
    return {
        "active_default": "local",
        "providers": list_available_providers()
    }
