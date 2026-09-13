"""Model registry inspection and dynamic switcher endpoints."""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from zenova.models.registry import ModelRegistry
from zenova.core.exceptions import ModelNotFoundException

router = APIRouter(prefix="/api/v1/models", tags=["Model Registry"])
registry = ModelRegistry()


class SwitchModelRequest(BaseModel):
    task: str
    provider_name: str


@router.get("")
async def list_models():
    """List all registered model providers and active configurations."""
    return registry.list_models()


@router.post("/switch")
async def switch_model(req: SwitchModelRequest):
    """Dynamically switch active provider for a task without code changes."""
    try:
        registry.set_active_provider(req.task, req.provider_name)
        return {
            "status": "success",
            "task": req.task,
            "active_provider": req.provider_name
        }
    except ModelNotFoundException as e:
        raise HTTPException(status_code=400, detail=str(e))
