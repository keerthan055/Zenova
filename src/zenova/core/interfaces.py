"""Versioned abstract interfaces for ZENOVA components."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    GeneratedResponse,
    SafetyResult,
    ConversationContext,
    MultimodalContext,
    ConversationTurn,
)


class BaseMultimodalContextEngine(ABC):
    """Version 1.0 Interface for Multimodal Context Aggregation Engine."""
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def build_context(
        self,
        user_input: UserInput,
        history: Optional[List[ConversationTurn]] = None,
        emotion: Optional[EmotionResult] = None,
        symptoms: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        baseline: Optional[BaselineResult] = None,
        behavior: Optional[BehavioralResult] = None,
        voice: Optional[VoiceResult] = None,
        previous_strategies: Optional[List[StrategyResult]] = None,
        previous_outcomes: Optional[List[Dict[str, Any]]] = None,
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MultimodalContext:
        """Combine heterogeneous module outputs into a normalized 9-block context."""
        pass


class BaseAnalyzer(ABC):
    """Version 1.0 Interface for analytical signal extractors."""
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> Any:
        """Extract signals and return a typed result."""
        pass


class BaseEmotionAnalyzer(BaseAnalyzer):
    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> EmotionResult:
        pass


class BaseSymptomAnalyzer(BaseAnalyzer):
    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> SymptomResult:
        pass


class BaseRiskAnalyzer(BaseAnalyzer):
    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> RiskResult:
        pass


class BaseBehaviorAnalyzer(BaseAnalyzer):
    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> BehavioralResult:
        pass


BaseBehavioralAnalyzer = BaseBehaviorAnalyzer


class BaseVoiceAnalyzer(BaseAnalyzer):
    @abstractmethod
    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> VoiceResult:
        pass


class BaseBaselineEngine(ABC):
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def evaluate(
        self,
        user_input: UserInput,
        emotion: Optional[EmotionResult] = None,
        context: Optional[ConversationContext] = None,
        symptom: Optional[SymptomResult] = None,
        risk: Optional[RiskResult] = None,
        extra_features: Optional[Dict[str, Any]] = None
    ) -> BaselineResult:
        pass


class BaseStrategyPlanner(ABC):
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def predict_strategy(
        self,
        user_input: UserInput,
        emotion: EmotionResult,
        symptoms: SymptomResult,
        context: Optional[ConversationContext] = None,
        multimodal_context: Optional[MultimodalContext] = None,
    ) -> StrategyResult:
        pass


class BaseResponseGenerator(ABC):
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def generate(
        self,
        user_input: UserInput,
        strategy: StrategyResult,
        context: Optional[ConversationContext] = None,
        rag_context: Optional[List[str]] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> GeneratedResponse:
        pass


class BaseSafetyGate(ABC):
    INTERFACE_VERSION = "1.0.0"

    @abstractmethod
    def verify(
        self,
        user_input: UserInput,
        candidate_response: GeneratedResponse,
        risk: RiskResult
    ) -> SafetyResult:
        pass
