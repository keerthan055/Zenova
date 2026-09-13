"""Unified Multimodal Fusion Engine connecting rule-based and learnable GMU fusion."""
from typing import Optional, List, Dict, Any

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    VoiceResult,
    BehavioralResult,
    BaselineResult,
)
from zenova.schemas.fusion import FusedMultimodalState, ComparativeEvaluationReport
from zenova.fusion.baseline_rule import WeightedRuleFusion
from zenova.fusion.learnable_gmu import LearnableGMUFusionEngine
from zenova.fusion.evaluator import MultimodalFusionEvaluator
from zenova.core.logging import get_logger

logger = get_logger("zenova.fusion.engine")


class UnifiedMultimodalFusionEngine:
    """Production multimodal fusion engine with configurable provider resolution."""

    def __init__(self, provider: str = "weighted_rule", **kwargs):
        self.provider = provider
        self.rule_engine = WeightedRuleFusion()
        self._gmu_engine: Optional[LearnableGMUFusionEngine] = None
        self._evaluator: Optional[MultimodalFusionEvaluator] = None

    @property
    def gmu_engine(self) -> LearnableGMUFusionEngine:
        if self._gmu_engine is None:
            self._gmu_engine = LearnableGMUFusionEngine()
        return self._gmu_engine

    @property
    def evaluator(self) -> MultimodalFusionEvaluator:
        if self._evaluator is None:
            self._evaluator = MultimodalFusionEvaluator()
        return self._evaluator

    def fuse(
        self,
        user_input: Optional[UserInput] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        voice: Optional[VoiceResult] = None,
        behavior: Optional[BehavioralResult] = None,
        baseline: Optional[BaselineResult] = None,
        history: Optional[List[Any]] = None,
    ) -> FusedMultimodalState:
        """Fuse multimodal inputs into a unified clinical affective assessment."""
        if self.provider == "learnable_gmu":
            try:
                return self.gmu_engine.fuse(
                    user_input=user_input,
                    emotion=emotion,
                    symptoms=symptoms,
                    risk=risk,
                    voice=voice,
                    behavior=behavior,
                    baseline=baseline,
                    history=history
                )
            except Exception as e:
                logger.warning(f"Learnable GMU fusion failed ({e}), falling back to weighted rule fusion.")
                return self.rule_engine.fuse(
                    user_input=user_input,
                    emotion=emotion,
                    symptoms=symptoms,
                    risk=risk,
                    voice=voice,
                    behavior=behavior,
                    baseline=baseline,
                    history=history
                )

        # Default: weighted_rule
        return self.rule_engine.fuse(
            user_input=user_input,
            emotion=emotion,
            symptoms=symptoms,
            risk=risk,
            voice=voice,
            behavior=behavior,
            baseline=baseline,
            history=history
        )

    def evaluate(self, num_samples_per_regime: int = 30) -> ComparativeEvaluationReport:
        """Execute the comparative evaluation benchmark."""
        return self.evaluator.run_benchmark(num_samples_per_regime=num_samples_per_regime)
