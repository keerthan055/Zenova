"""Base interface and schemas for configurable LLM providers."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Normalized response schema from any LLM provider."""
    text: str
    model_name: str
    provider: str
    tokens_used: Optional[int] = None
    latency_ms: float = 0.0
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers supporting configurable backends."""

    def __init__(
        self,
        model_name: str,
        provider_name: str,
        timeout: float = 15.0,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Synchronously generate text response from LLM."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Asynchronously generate text response from LLM."""
        pass
