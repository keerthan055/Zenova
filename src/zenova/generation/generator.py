"""Production Strategy-Controlled Conversational Response Generator."""
import time
from typing import Optional, List, Dict, Any
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
from zenova.generation.prompt import StrategyPromptBuilder
from zenova.generation.providers import (
    BaseLLMProvider,
    LocalFallbackProvider,
    get_llm_provider
)
from zenova.generation.validator import ResponseValidator
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.generator")


class StrategyControlledGenerator(BaseResponseGenerator):
    """Production response generator strictly conditioned on the selected support strategy and multimodal context."""
    MODULE_NAME = "StrategyControlledGenerator"
    VERSION = "strategy-llm-v1.0.0"

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        self.provider = get_llm_provider(provider_name=provider_name, model_name=model_name)
        self.fallback_provider = LocalFallbackProvider()
        self.prompt_builder = StrategyPromptBuilder()
        self.validator = ResponseValidator()
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Initialized StrategyControlledGenerator with active provider: {self.provider.provider_name} ({self.provider.model_name})")

    def generate(
        self,
        user_input: UserInput,
        strategy: StrategyResult,
        context: Optional[ConversationContext] = None,
        rag_context: Optional[List[str]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> GeneratedResponse:
        """Generate response conditioned strictly on the chosen strategy."""
        t0 = time.time()
        prompts = self.prompt_builder.build_prompt(
            user_input=user_input,
            strategy=strategy,
            context=context,
            rag_context=rag_context,
            multimodal_context=multimodal_context
        )

        try:
            llm_res = self.provider.generate(
                prompt=prompts["user_prompt"],
                system_prompt=prompts["system_prompt"],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        except Exception as e:
            logger.warning(f"Primary provider {self.provider.provider_name} failed: {e}. Executing local fallback...")
            llm_res = self.fallback_provider.generate(prompt=prompts["user_prompt"])

        # Validate response against clinical guardrails and strategy fidelity
        strat_applied = strategy.selected_strategy
        val_res = self.validator.validate(llm_res.text, strat_applied)

        final_text = llm_res.text
        if not val_res.is_valid:
            logger.warning(
                f"Candidate response failed validation: {val_res.violations}. Falling back to clinical template..."
            )
            fallback_res = self.fallback_provider.generate(prompt=prompts["user_prompt"])
            final_text = fallback_res.text

        total_latency = (time.time() - t0) * 1000

        # Non-sensitive sanitized audit logging
        sanitized_summary = self.validator.sanitize_log_content(final_text)
        strat_name = strat_applied.value if hasattr(strat_applied, "value") else str(strat_applied)
        logger.info(
            f"Response synthesized: strategy='{strat_name}', provider='{llm_res.provider}', "
            f"latency={total_latency:.1f}ms, valid={val_res.is_valid}, output={sanitized_summary}"
        )

        return GeneratedResponse(
            is_placeholder=False,
            module_version=self.VERSION,
            response_text=final_text,
            strategy_applied=strat_applied,
            model_name=llm_res.model_name,
            rag_sources=rag_context or [],
            tokens_used=llm_res.tokens_used,
            validation_passed=val_res.is_valid,
            generation_metadata={
                "provider": llm_res.provider,
                "strategy_rationale": strategy.rationale or "",
                "validation_violations": val_res.violations,
                "fallback_applied": not val_res.is_valid
            },
            latency_ms=round(total_latency, 2),
            timestamp=datetime.now(timezone.utc)
        )

    async def agenerate(
        self,
        user_input: UserInput,
        strategy: StrategyResult,
        context: Optional[ConversationContext] = None,
        rag_context: Optional[List[str]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> GeneratedResponse:
        """Asynchronous strategy-controlled response generation."""
        t0 = time.time()
        prompts = self.prompt_builder.build_prompt(
            user_input=user_input,
            strategy=strategy,
            context=context,
            rag_context=rag_context,
            multimodal_context=multimodal_context
        )

        try:
            llm_res = await self.provider.agenerate(
                prompt=prompts["user_prompt"],
                system_prompt=prompts["system_prompt"],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        except Exception as e:
            logger.warning(f"Async primary provider {self.provider.provider_name} failed: {e}. Executing local fallback...")
            llm_res = await self.fallback_provider.agenerate(prompt=prompts["user_prompt"])

        strat_applied = strategy.selected_strategy
        val_res = self.validator.validate(llm_res.text, strat_applied)

        final_text = llm_res.text
        if not val_res.is_valid:
            logger.warning(
                f"Candidate response failed validation: {val_res.violations}. Falling back to clinical template..."
            )
            fallback_res = await self.fallback_provider.agenerate(prompt=prompts["user_prompt"])
            final_text = fallback_res.text

        total_latency = (time.time() - t0) * 1000
        sanitized_summary = self.validator.sanitize_log_content(final_text)
        strat_name = strat_applied.value if hasattr(strat_applied, "value") else str(strat_applied)
        logger.info(
            f"Async response synthesized: strategy='{strat_name}', provider='{llm_res.provider}', "
            f"latency={total_latency:.1f}ms, valid={val_res.is_valid}, output={sanitized_summary}"
        )

        return GeneratedResponse(
            is_placeholder=False,
            module_version=self.VERSION,
            response_text=final_text,
            strategy_applied=strat_applied,
            model_name=llm_res.model_name,
            rag_sources=rag_context or [],
            tokens_used=llm_res.tokens_used,
            validation_passed=val_res.is_valid,
            generation_metadata={
                "provider": llm_res.provider,
                "strategy_rationale": strategy.rationale or "",
                "validation_violations": val_res.violations,
                "fallback_applied": not val_res.is_valid
            },
            latency_ms=round(total_latency, 2),
            timestamp=datetime.now(timezone.utc)
        )
