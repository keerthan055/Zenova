"""Integration adapter connecting SymptomInferenceEngine with BaseSymptomAnalyzer."""
from typing import Optional
from datetime import datetime, timezone

from zenova.core.interfaces import BaseSymptomAnalyzer
from zenova.symptoms.inference import SymptomInferenceEngine
from zenova.schemas.standard import (
    UserInput,
    SymptomResult,
    SymptomSignal,
    SymptomSeverity,
    ConversationContext
)


class SymptomTransformerAnalyzer(BaseSymptomAnalyzer):
    """Production symptom analyzer fulfilling the BaseSymptomAnalyzer interface contract."""

    MODULE_NAME = "SymptomTransformerAnalyzer"
    VERSION = "transformer-v1.0.0"

    def __init__(self, model_dir: str = "models/symptoms"):
        self.engine = SymptomInferenceEngine(model_dir=model_dir)

    def analyze(self, user_input: UserInput, context: Optional[ConversationContext] = None) -> SymptomResult:
        res = self.engine.predict(user_input.text)

        signals_list = []
        for s in res["signals"]:
            signals_list.append(
                SymptomSignal(
                    marker_name=s["marker_name"],
                    severity=SymptomSeverity(s["severity"]),
                    confidence=s["confidence"],
                    evidence_spans=s["evidence_spans"],
                    clinical_disclaimer=s["clinical_disclaimer"]
                )
            )

        return SymptomResult(
            is_placeholder=False,
            module_version=self.VERSION,
            signals=signals_list,
            aggregate_severity=SymptomSeverity(res["aggregate_severity"]),
            disclaimer=res["disclaimer"],
            notes=res["notes"],
            timestamp=datetime.now(timezone.utc)
        )
