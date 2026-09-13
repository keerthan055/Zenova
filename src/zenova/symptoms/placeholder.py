"""Explicit placeholder implementation for Symptom Identification."""
from typing import Optional
from datetime import datetime, timezone
from zenova.core.interfaces import BaseSymptomAnalyzer
from zenova.schemas.standard import (
    UserInput,
    SymptomResult,
    SymptomSeverity,
    ConversationContext
)


class SymptomPlaceholderAnalyzer(BaseSymptomAnalyzer):
    """Placeholder symptom analyzer explicitly flagged as non-ML baseline."""
    MODULE_NAME = "SymptomPlaceholderAnalyzer"
    VERSION = "placeholder-v0.1.0"

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> SymptomResult:
        return SymptomResult(
            is_placeholder=True,
            module_version=self.VERSION,
            signals=[],
            aggregate_severity=SymptomSeverity.NONE,
            disclaimer="Placeholder implementation; no clinical symptom identification performed.",
            notes="Placeholder mode active.",
            timestamp=datetime.now(timezone.utc)
        )
