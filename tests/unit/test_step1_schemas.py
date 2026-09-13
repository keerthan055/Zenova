"""Unit tests for standard Step 1 schemas."""
import pytest
from datetime import datetime, timezone
from zenova.schemas.standard import (
    UserInput,
    ConversationTurn,
    ConversationContext,
    EmotionResult,
    SymptomResult,
    RiskResult,
    BaselineResult,
    StrategyResult,
    GeneratedResponse,
    SafetyResult,
    EscalationEvent,
    EmotionCategory,
    RiskLevel,
    CrisisCategory,
    SupportStrategy,
    SpeakerRole,
    SafetyAction,
    EscalationStatus
)


def test_user_input_schema():
    inp = UserInput(
        session_id="s1",
        user_id="u1",
        text="Feeling anxious about work"
    )
    assert inp.session_id == "s1"
    assert inp.text == "Feeling anxious about work"
    assert inp.modality == "text"


def test_emotion_result_schema():
    res = EmotionResult(
        is_placeholder=True,
        module_version="placeholder-v0.1.0",
        primary_emotion=EmotionCategory.ANXIETY,
        confidence=0.8,
        probabilities={"anxiety": 0.8, "fear": 0.2},
        valence=-0.5,
        arousal=0.7
    )
    assert res.is_placeholder is True
    assert res.primary_emotion == EmotionCategory.ANXIETY


def test_risk_result_schema():
    res = RiskResult(
        is_placeholder=False,
        risk_level=RiskLevel.HIGH,
        crisis_category=CrisisCategory.SUICIDAL_IDEATION,
        confidence=0.9,
        requires_immediate_escalation=True,
        trigger_cues=["suicidal thought"]
    )
    assert res.is_high_risk is True
    assert res.requires_immediate_escalation is True


def test_escalation_event_schema():
    event = EscalationEvent(
        event_id="esc_123",
        session_id="s1",
        user_id="u1",
        risk_level=RiskLevel.CRITICAL,
        crisis_category=CrisisCategory.SELF_HARM,
        trigger_cues=["hurt myself"]
    )
    assert event.status == EscalationStatus.PENDING
    assert event.event_id == "esc_123"


def test_conversation_turn_and_context():
    turn = ConversationTurn(
        turn_id=1,
        session_id="s1",
        speaker=SpeakerRole.USER,
        content="Hello Zenova"
    )
    ctx = ConversationContext(
        session_id="s1",
        user_id="u1",
        turns=[turn]
    )
    assert len(ctx.turns) == 1
    assert ctx.turns[0].speaker == SpeakerRole.USER
