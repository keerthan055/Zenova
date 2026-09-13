"""ZENOVA Central Pipeline Orchestrator.

Provides backward-compatible facade delegating to the modular ZenovaOrchestrationEngine.
"""
from typing import Optional, Dict, Any

from zenova.core.logging import get_logger
from zenova.core.config import get_system_config
from zenova.models.registry import ModelRegistry
from zenova.schemas.standard import UserInput, ConversationContext
from zenova.orchestration.engine import ZenovaOrchestrationEngine
from zenova.escalation.engine import EscalationDecisionEngine

logger = get_logger("zenova.core.orchestrator")


class ZenovaOrchestrator:
    """Modular execution engine orchestrating all ZENOVA components."""

    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.config = get_system_config()
        self.registry = model_registry or ModelRegistry()
        self.escalation_engine = EscalationDecisionEngine()
        self.engine = ZenovaOrchestrationEngine(model_registry=self.registry)

    def get_module(self, module_name: str) -> Any:
        """Preserve individual module API queryability."""
        return self.registry.get_module_instance(module_name)

    async def process_turn(
        self,
        user_input: UserInput,
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """Process an inbound conversational turn end-to-end via modular stages."""
        return await self.engine.process_turn(user_input, context)
