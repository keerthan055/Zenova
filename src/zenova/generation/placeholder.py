"""Explicit placeholder implementation for Response Generation."""
from typing import Optional, List
from datetime import datetime, timezone
from zenova.core.interfaces import BaseResponseGenerator
from zenova.schemas.standard import (
    UserInput,
    StrategyResult,
    GeneratedResponse,
    SupportStrategy,
    ConversationContext,
    MultimodalContext
)


class ResponsePlaceholderGenerator(BaseResponseGenerator):
    MODULE_NAME = "ResponsePlaceholderGenerator"
    VERSION = "placeholder-v0.1.0"

    def generate(
        self,
        user_input: UserInput,
        strategy: StrategyResult,
        context: Optional[ConversationContext] = None,
        rag_context: Optional[List[str]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> GeneratedResponse:
        strat = strategy.selected_strategy
        # Transparent rule-based placeholder responses aligned with Hill's helping skills
        templates = {
            SupportStrategy.QUESTION: "Thank you for sharing that with me. Could you tell me more about what has been on your mind recently?",
            SupportStrategy.REFLECTION_OF_FEELINGS: "It sounds like you have been experiencing a lot of heavy emotions lately.",
            SupportStrategy.AFFIRMATION_AND_REASSURANCE: "It takes courage to open up about these feelings. You are doing the best you can.",
            SupportStrategy.RESTATEMENT_OR_PARAPHRASING: "What I am hearing is that things have felt overwhelming for you recently.",
            SupportStrategy.PROVIDING_SUGGESTIONS: "When you feel ready, perhaps taking a short gentle pause or deep breath could help.",
            SupportStrategy.INFORMATION: "It is normal to experience fluctuations in stress and energy when facing challenging situations.",
            SupportStrategy.SELF_DISCLOSURE: "Many people experience similar feelings during difficult moments.",
            SupportStrategy.OTHERS: "I am here with you. Please take all the time you need."
        }
        text = templates.get(strat, "I am here to support you. How are you feeling right now?")

        return GeneratedResponse(
            is_placeholder=True,
            module_version=self.VERSION,
            response_text=text,
            strategy_applied=strat,
            model_name="rule_template_placeholder",
            rag_sources=[],
            latency_ms=1.5,
            timestamp=datetime.now(timezone.utc)
        )
