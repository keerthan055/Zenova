"""Unit tests for ZENOVA Pydantic schemas."""
import pytest
from pydantic import ValidationError
from datetime import datetime

from zenova.schemas.emotion import EmotionAnalysisResult, EmotionCategory
from zenova.schemas.symptoms import SymptomAnalysisResult, SymptomSignal, SymptomSeverity
from zenova.schemas.risk import CrisisRiskAssessment, RiskLevel, CrisisCategory
from zenova.schemas.baseline import UserBaselineProfile, BaselineDeviationReport
from zenova.schemas.strategy import SupportStrategy, DialogStage, StrategyPrediction
from zenova.schemas.safety import SafetyGateResult, SafetyAction
from zenova.schemas.pipeline import ZenovaUserTurn, ZenovaPipelineOutput


def test_emotion_schema_valid():
    res = EmotionAnalysisResult(
        primary_emotion=EmotionCategory.ANXIETY,
        confidence=0.85,
        probabilities={"anxiety": 0.85, "fear": 0.15},
        valence=-0.4,
        arousal=0.6
    )
    assert res.primary_emotion == EmotionCategory.ANXIETY
    assert res.confidence == 0.85
    assert res.valence == -0.4
    assert res.arousal == 0.6


def test_emotion_schema_invalid_probability():
    with pytest.raises(ValidationError):
        EmotionAnalysisResult(
            primary_emotion=EmotionCategory.ANXIETY,
            confidence=1.5,  # Invalid: > 1.0
            probabilities={"anxiety": 1.5}
        )


def test_crisis_risk_assessment_bypass_flag():
    low_risk = CrisisRiskAssessment(
        risk_level=RiskLevel.LOW,
        confidence=0.99,
        requires_immediate_escalation=False
    )
    assert not low_risk.is_high_risk

    high_risk = CrisisRiskAssessment(
        risk_level=RiskLevel.HIGH,
        confidence=0.85,
        requires_immediate_escalation=True
    )
    assert high_risk.is_high_risk


def test_symptom_schema():
    sig = SymptomSignal(
        marker_name="sleep_disturbance",
        severity=SymptomSeverity.MILD,
        confidence=0.78,
        evidence_spans=["barely slept 3 hours"]
    )
    res = SymptomAnalysisResult(signals=[sig], aggregate_severity=SymptomSeverity.MILD)
    assert len(res.signals) == 1
    assert res.signals[0].marker_name == "sleep_disturbance"
    assert "not constitute clinical diagnosis" in sig.clinical_disclaimer


def test_strategy_enum_hill_skills():
    expected = {
        "Question",
        "Restatement or Paraphrasing",
        "Reflection of feelings",
        "Affirmation and Reassurance",
        "Self-disclosure",
        "Providing Suggestions",
        "Information",
        "Others"
    }
    actual = {s.value for s in SupportStrategy}
    assert expected == actual


def test_pipeline_output_serialization(sample_user_turn, sample_emotion_result):
    output = ZenovaPipelineOutput(
        session_id=sample_user_turn.session_id,
        turn_id=sample_user_turn.turn_id,
        user_input=sample_user_turn.text,
        emotion=sample_emotion_result,
        final_response="It sounds like you have been carrying a heavy weight with your coursework.",
        escalated_to_human=False
    )
    json_str = output.model_dump_json()
    assert "sess_test_123" in json_str
    assert "sadness" in json_str
