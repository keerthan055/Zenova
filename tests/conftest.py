"""Pytest fixtures and configuration for ZENOVA tests."""
import sys
from pathlib import Path
import pytest

# Ensure src is in python path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from zenova.schemas.pipeline import ZenovaUserTurn
from zenova.schemas.emotion import EmotionAnalysisResult, EmotionCategory
from zenova.schemas.risk import CrisisRiskAssessment, RiskLevel, CrisisCategory


@pytest.fixture
def sample_user_turn() -> ZenovaUserTurn:
    return ZenovaUserTurn(
        session_id="sess_test_123",
        user_id="user_test_456",
        turn_id=1,
        text="I have been feeling really exhausted and unmotivated with my coursework lately."
    )


@pytest.fixture
def sample_emotion_result() -> EmotionAnalysisResult:
    return EmotionAnalysisResult(
        primary_emotion=EmotionCategory.SADNESS,
        confidence=0.88,
        probabilities={"sadness": 0.88, "anxiety": 0.10, "neutral": 0.02},
        valence=-0.6,
        arousal=-0.3,
        dominance=-0.4
    )


@pytest.fixture
def sample_crisis_assessment() -> CrisisRiskAssessment:
    return CrisisRiskAssessment(
        risk_level=RiskLevel.CRITICAL,
        crisis_category=CrisisCategory.SUICIDAL_IDEATION,
        confidence=0.95,
        trigger_cues=["want to end it all"],
        requires_immediate_escalation=True,
        escalation_action="provide_immediate_crisis_hotline"
    )
