"""High-quality offline rule-and-slot strategy-conditioned generator."""
import re
import time
from typing import Optional, Dict, Any, List
from zenova.schemas.strategy import SupportStrategy
from zenova.generation.providers.base import BaseLLMProvider, LLMResponse


class LocalFallbackProvider(BaseLLMProvider):
    """Deterministic, high-quality, clinically structured offline strategy generator.

    Provides guaranteed zero-cost, zero-latency, offline response generation strictly
    conditioned on the assigned support strategy and user affective context.
    """

    def __init__(
        self,
        model_name: str = "zenova-offline-slot-v1",
        provider_name: str = "local"
    ):
        super().__init__(model_name=model_name, provider_name=provider_name)

    def _extract_strategy_from_prompt(self, prompt: str) -> str:
        """Parse the assigned strategy from structured prompt if available."""
        match = re.search(r"STRATEGY:\s*([^\n]+)", prompt)
        if match:
            return match.group(1).strip()
        return SupportStrategy.OTHERS.value

    def _extract_user_text(self, prompt: str) -> str:
        """Extract user input text from structured prompt."""
        match = re.search(r"Seeker:\s*([^\n]+)", prompt)
        if match:
            return match.group(1).strip()
        return ""

    def _generate_text(self, prompt: str) -> str:
        strat = self._extract_strategy_from_prompt(prompt)
        user_msg = self._extract_user_text(prompt)

        # Strategy-conditioned responses strictly adhering to Hill's helping skills
        templates: Dict[str, List[str]] = {
            SupportStrategy.QUESTION.value: [
                "Thank you for sharing that with me. What part of this situation has felt the hardest or most exhausting for you to carry?",
                "I hear you. When you reflect on what has been happening, what is the main thought that comes to mind?",
                "Could you tell me a little more about how this has been affecting your day-to-day routine?"
            ],
            SupportStrategy.RESTATEMENT_OR_PARAPHRASING.value: [
                "What I am hearing is that things have been feeling deeply overwhelming and demanding a lot of your energy lately.",
                "It sounds like you are navigating a situation with heavy expectations, and it feels like a lot to manage all at once.",
                "So in essence, you are dealing with a significant amount of stress while trying your best to keep up."
            ],
            SupportStrategy.REFLECTION_OF_FEELINGS.value: [
                "It sounds like you are feeling deeply exhausted and emotionally drained by everything you are carrying right now.",
                "I can really hear the weight and sadness in what you are describing. It is completely natural to feel worn down by this.",
                "It sounds like you are experiencing a heavy mixture of anxiety and frustration, and that is a very painful place to be."
            ],
            SupportStrategy.AFFIRMATION_AND_REASSURANCE.value: [
                "It takes a great deal of courage to speak honestly about how hard things are. You are doing the best you can under heavy pressure.",
                "Please give yourself credit for continuing to navigate this day by day, even when it feels heavy and uncertain.",
                "You are dealing with an enormous amount of strain right now, and your feelings are completely understandable and valid."
            ],
            SupportStrategy.SELF_DISCLOSURE.value: [
                "Many people experience this exact kind of exhaustion when pressure builds up, and it is easy to feel completely alone in it.",
                "It is a very common and understandable human reaction to feel like stepping back when everything demands our attention at once.",
                "It is widely shared that during intense periods, even simple tasks can suddenly feel like mountains to climb."
            ],
            SupportStrategy.PROVIDING_SUGGESTIONS.value: [
                "If you feel up to it, perhaps taking just a few gentle breaths or stepping away for five quiet minutes might offer a small pocket of relief.",
                "Would it feel manageable today to write down just one single priority and give yourself permission to let the rest wait?",
                "Sometimes gently grounding ourselves with a glass of water or stepping outside into fresh air can help break that sense of tightness."
            ],
            SupportStrategy.INFORMATION.value: [
                "When our stress levels remain heightened, our nervous system naturally stays on high alert, which can make everyday tasks feel physically and mentally draining.",
                "Fluctuations in energy, focus, and emotional bandwidth are standard physiological responses to sustained stress, rather than personal shortcomings.",
                "Taking regular micro-breaks throughout the day can assist in gradually resetting our nervous system's stress response."
            ],
            SupportStrategy.OTHERS.value: [
                "I am here with you. Please take all the time you need, and we can take this one step at a time.",
                "Thank you for checking in today. I am listening whenever you are ready to share more.",
                "I hear you, and I am glad you reached out. How are you feeling in this present moment?"
            ]
        }

        strat_options = templates.get(strat, templates[SupportStrategy.OTHERS.value])
        # Select first or hash-based template deterministically
        idx = hash(user_msg) % len(strat_options)
        return strat_options[idx]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        t0 = time.time()
        text = self._generate_text(prompt)
        latency = (time.time() - t0) * 1000
        words = len(text.split())
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider_name,
            tokens_used=int(words * 1.3),
            latency_ms=round(latency, 2),
            raw_metadata={"backend": "rule_slot_offline"}
        )

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> LLMResponse:
        return self.generate(prompt, system_prompt, max_tokens, temperature)
