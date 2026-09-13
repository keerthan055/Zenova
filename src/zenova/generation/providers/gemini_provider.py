"""Google Gemini LLM provider integration with retry logic."""
import os
import time
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from zenova.generation.providers.base import BaseLLMProvider, LLMResponse
from zenova.core.logging import get_logger

logger = get_logger("zenova.generation.providers.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini REST API client."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: int = 3
    ):
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        super().__init__(
            model_name=model,
            provider_name="gemini",
            timeout=timeout,
            max_retries=max_retries
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

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
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        return payload

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
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
        res = _call()
        latency = (time.time() - t0) * 1000

        data = res.json()
        candidates = data.get("candidates", [{}])
        parts = candidates[0].get("content", {}).get("parts", [{}])
        text = parts[0].get("text", "").strip() if parts else ""
        usage = data.get("usageMetadata", {})
        tokens = usage.get("totalTokenCount")

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens,
            latency_ms=round(latency, 2),
            raw_metadata={"finish_reason": candidates[0].get("finishReason")}
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(prompt, system_prompt, max_tokens, temperature)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

        latency = (time.time() - t0) * 1000
        candidates = data.get("candidates", [{}])
        parts = candidates[0].get("content", {}).get("parts", [{}])
        text = parts[0].get("text", "").strip() if parts else ""
        usage = data.get("usageMetadata", {})
        tokens = usage.get("totalTokenCount")

        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=tokens,
            latency_ms=round(latency, 2),
            raw_metadata={"finish_reason": candidates[0].get("finishReason")}
        )
