"""Anthropic Claude LLM provider integration with retry logic."""
import os
import time
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from zenova.generation.providers.base import BaseLLMProvider, LLMResponse
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.providers.anthropic")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Messages API client."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 3
    ):
        model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        super().__init__(
            model_name=model,
            provider_name="anthropic",
            timeout=timeout,
            max_retries=max_retries
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = "https://api.anthropic.com/v1/messages"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = self._build_payload(prompt, system_prompt, max_tokens, temperature)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
        )
        def _call() -> httpx.Response:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(self.base_url, json=payload, headers=headers)
                res.raise_for_status()
                return res

        t0 = time.time()
        res = _call()
        latency = (time.time() - t0) * 1000

        data = res.json()
        content = data.get("content", [{}])
        text = content[0].get("text", "").strip() if content else ""
        usage = data.get("usage", {})
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens_used,
            latency_ms=round(latency, 2),
            raw_metadata={"stop_reason": data.get("stop_reason")}
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = self._build_payload(prompt, system_prompt, max_tokens, temperature)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(self.base_url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

        latency = (time.time() - t0) * 1000
        content = data.get("content", [{}])
        text = content[0].get("text", "").strip() if content else ""
        usage = data.get("usage", {})
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens_used,
            latency_ms=round(latency, 2),
            raw_metadata={"stop_reason": data.get("stop_reason")}
        )
