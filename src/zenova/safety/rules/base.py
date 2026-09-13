"""Abstract base class for modular safety policy rules."""
from abc import ABC, abstractmethod
from typing import List, Optional
from zenova.schemas.safety import SafetyViolation, SafetyPolicy, SafetySeverity
from zenova.schemas.standard import UserInput, GeneratedResponse, RiskResult, MultimodalContext


class BaseSafetyRule(ABC):
    """Contract for individual safety policy verifiers."""
    policy: SafetyPolicy
    default_severity: SafetySeverity = SafetySeverity.HIGH

    @abstractmethod
    def check(
        self,
        candidate_text: str,
        user_input: Optional[UserInput] = None,
        risk: Optional[RiskResult] = None,
        multimodal_context: Optional[MultimodalContext] = None
    ) -> List[SafetyViolation]:
        """Check candidate text for policy violations. Returns empty list if clean."""
        pass
