"""Factory for configurable LLM providers with automatic fallback."""
import os
from typing import Optional, List, Dict, Any

from zenova.generation.providers.base import BaseLLMProvider
from zenova.generation.providers.local_provider import LocalFallbackProvider
from zenova.generation.providers.openai_provider import OpenAIProvider
from zenova.generation.providers.anthropic_provider import AnthropicProvider
from zenova.generation.providers.gemini_provider import GeminiProvider
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.providers.factory")


def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMProvider:
    """Resolve and instantiate configured LLM provider with graceful local fallback."""
    name = (provider_name or os.getenv("LLM_PROVIDER", "local")).lower().strip()

    if name == "openai":
        provider = OpenAIProvider(model_name=model_name)
        if provider.is_configured:
            return provider
        logger.warning("OpenAI provider requested but OPENAI_API_KEY is not set. Falling back to LocalFallbackProvider.")

    elif name == "anthropic":
        provider = AnthropicProvider(model_name=model_name)
        if provider.is_configured:
            return provider
        logger.warning("Anthropic provider requested but ANTHROPIC_API_KEY is not set. Falling back to LocalFallbackProvider.")

    elif name == "gemini":
        provider = GeminiProvider(model_name=model_name)
        if provider.is_configured:
            return provider
        logger.warning("Gemini provider requested but GEMINI_API_KEY is not set. Falling back to LocalFallbackProvider.")

    # Default robust local fallback provider
    return LocalFallbackProvider()


def list_available_providers() -> List[Dict[str, Any]]:
    """Enumerate status of supported LLM providers."""
    return [
        {
            "name": "local",
            "description": "Deterministic offline rule-and-slot strategy generator",
            "is_configured": True,
            "is_local": True
        },
        {
            "name": "openai",
            "description": "OpenAI Chat Completions API (gpt-4o-mini)",
            "is_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "is_local": False
        },
        {
            "name": "anthropic",
            "description": "Anthropic Claude Messages API (claude-3-haiku-20240307)",
            "is_configured": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
            "is_local": False
        },
        {
            "name": "gemini",
            "description": "Google Gemini Content Generation API (gemini-1.5-flash)",
            "is_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "is_local": False
        }
    ]
