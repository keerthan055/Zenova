"""OpenAI LLM provider integration with retry logic and non-sensitive error handling."""
import os
import time
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from zenova.generation.providers.base import BaseLLMProvider, LLMResponse
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.providers.openai")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API client using standard HTTP REST interface."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 3
    ):
        model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        super().__init__(
            model_name=model,
            provider_name="openai",
            timeout=timeout,
            max_retries=max_retries
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
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
                res = client.post(url, json=payload, headers=headers)
                res.raise_for_status()
                return res

        t0 = time.time()
        response = _call()
        latency = (time.time() - t0) * 1000

        data = response.json()
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens")

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens_used,
            latency_ms=round(latency, 2),
            raw_metadata={"finish_reason": choice.get("finish_reason")}
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = self._build_payload(prompt, system_prompt, max_tokens, temperature)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

        latency = (time.time() - t0) * 1000
        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=usage.get("total_tokens"),
            latency_ms=round(latency, 2),
            raw_metadata={"finish_reason": choice.get("finish_reason")}
        )
