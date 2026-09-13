"""Unit tests for ZENOVA Step 9: Multimodal Context Engine."""
import pytest
from datetime import datetime, timezone

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSignal,
    SymptomSeverity,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    BaselineResult,
    BehavioralResult,
    VoiceResult,
    StrategyResult,
    SupportStrategy,
    DialogStage
)
from zenova.schemas.context import (
    MultimodalContext,
    PrivacyLevel,
    ValenceTrend,
    UserFeedback
)
from zenova.context.engine import MultimodalContextEngine
from zenova.context.confidence import ConfidencePropagator
from zenova.context.trajectory import TrajectoryAnalyzer
from zenova.context.privacy import PrivacyEngine
from zenova.context.placeholder import ContextPlaceholderEngine
from zenova.strategy.placeholder import StrategyPlaceholderPlanner


@pytest.fixture
def sample_user_input():
    return UserInput(
        session_id="sess_unit_test",
        user_id="user_12345",
        text="I have been feeling quite down lately, but today had some quiet moments."
    )


@pytest.fixture
def sample_emotion():
    return EmotionResult(
        primary_emotion=EmotionCategory.SADNESS,
        confidence=0.82,
        valence=-0.45,
        arousal=0.30,
        dominance=-0.20,
        probabilities={"sadness": 0.82, "neutral": 0.18}
    )


@pytest.fixture
def sample_symptoms():
    return SymptomResult(
        signals=[
            SymptomSignal(
                marker_name="depressed mood",
                confidence=0.85,
                severity=SymptomSeverity.MODERATE,
                evidence_spans=["feeling quite down lately"]
            )
        ]
    )


@pytest.fixture
def sample_risk():
    return RiskResult(
        risk_level=RiskLevel.LOW,
        confidence=0.92,
        crisis_category=CrisisCategory.NONE,
        is_high_risk=False,
        requires_escalation=False
    )


@pytest.fixture
def sample_baseline():
    return BaselineResult(
        user_id="user_12345",
        status="stable",
        confidence=0.75,
        total_observations=8,
        features={
            "anxiety_score": {
                "current_value": 0.45,
                "baseline_mean": 0.42,
                "baseline_std": 0.08,
                "deviation": 0.375,
                "interpretation": "within_normal_baseline"
            }
        }
    )


def test_multimodal_context_schema_validation(sample_user_input, sample_emotion, sample_symptoms, sample_risk, sample_baseline):
    """Test full 9-block MultimodalContext instantiation and validation."""
    engine = MultimodalContextEngine()
    ctx = engine.build_context(
        user_input=sample_user_input,
        emotion=sample_emotion,
        symptoms=sample_symptoms,
        risk=sample_risk,
        baseline=sample_baseline
    )

    assert isinstance(ctx, MultimodalContext)
    # Check 9 blocks exist
    assert ctx.conversation.session_id == "sess_unit_test"
    assert ctx.emotion.primary_emotion == "sadness"
    assert ctx.symptoms.signal_count == 1
    assert ctx.risk.risk_level == "low"
    assert ctx.baseline.status == "stable"
    assert ctx.behavior.is_available is False
    assert ctx.voice.is_available is False
    assert ctx.history.turn_count == 0
    assert ctx.metadata.schema_version == "1.0.0"
    assert ctx.metadata.engine_version.startswith("zenova-context")
    assert len(ctx.metadata.context_hash) == 64


def test_missing_data_explicit_representation(sample_user_input, sample_emotion):
    """Verify that omitted modalities are marked is_available=False with clear reasons without fabrication."""
    engine = MultimodalContextEngine()
    ctx = engine.build_context(
        user_input=sample_user_input,
        emotion=sample_emotion,
        symptoms=None,
        risk=None,
        baseline=None,
        behavior=None,
        voice=None
    )

    # Behavior
    assert ctx.behavior.is_available is False
    assert ctx.behavior.reason == "passive_sensing_omitted"
    assert ctx.behavior.bdi == 0.0
    assert ctx.behavior.domains == {}

    # Voice
    assert ctx.voice.is_available is False
    assert ctx.voice.reason == "audio_omitted"
    assert ctx.voice.transcription is None
    assert ctx.voice.acoustic_summary == {}

    # Symptoms & Risk
    assert ctx.symptoms.is_available is False
    assert ctx.symptoms.reason == "symptom_analyzer_omitted"
    assert ctx.risk.is_available is False
    assert ctx.risk.reason == "risk_analyzer_omitted"

    # Metadata missing/available lists
    assert "behavior" in ctx.metadata.missing_modalities
    assert "voice" in ctx.metadata.missing_modalities
    assert "symptoms" in ctx.metadata.missing_modalities
    assert "emotion" in ctx.metadata.available_modalities


def test_deterministic_context_hashing(sample_user_input, sample_emotion, sample_risk):
    """Test that identical inputs produce identical context hashes and different inputs produce different hashes."""
    engine = MultimodalContextEngine()

    ctx1 = engine.build_context(user_input=sample_user_input, emotion=sample_emotion, risk=sample_risk)
    ctx2 = engine.build_context(user_input=sample_user_input, emotion=sample_emotion, risk=sample_risk)

    assert ctx1.metadata.context_hash == ctx2.metadata.context_hash
    assert ctx1.metadata.context_id == ctx2.metadata.context_id

    # Modify input text
    diff_input = UserInput(session_id="sess_unit_test", user_id="user_12345", text="Completely different text.")
    ctx3 = engine.build_context(user_input=diff_input, emotion=sample_emotion, risk=sample_risk)

    assert ctx3.metadata.context_hash != ctx1.metadata.context_hash


def test_confidence_propagation():
    """Verify weighted average, min floor, and harmonic mean confidence propagation."""
    propagator = ConfidencePropagator()
    mod_confs = {
        "risk": 0.95,
        "emotion": 0.80,
        "symptoms": 0.70,
        "voice": 0.60
    }
    result = propagator.compute_propagation(mod_confs)

    assert 0.0 <= result["overall_context_confidence"] <= 1.0
    assert result["min_modality_confidence"] == 0.60
    assert result["harmonic_mean_confidence"] < result["overall_context_confidence"]
    assert "risk" in result["active_weights"]
    assert sum(result["active_weights"].values()) == pytest.approx(1.0, abs=0.01)


def test_affective_trajectory_improving():
    """Test trajectory analyzer detects improving emotional trend."""
    turns = [
        {"turn_id": 1, "speaker": "user", "emotion": {"valence": -0.70, "arousal": 0.6}},
        {"turn_id": 2, "speaker": "assistant", "strategy": {"selected_strategy": "Reflection of feelings"}},
        {"turn_id": 3, "speaker": "user", "emotion": {"valence": -0.30, "arousal": 0.4}},
        {"turn_id": 4, "speaker": "assistant", "strategy": {"selected_strategy": "Affirmation and Reassurance"}},
        {"turn_id": 5, "speaker": "user", "emotion": {"valence": 0.10, "arousal": 0.3}},
    ]

    res = TrajectoryAnalyzer.analyze_trajectory(turns, current_valence=0.35)
    assert res["valence_trend"] == ValenceTrend.IMPROVING
    assert len(res["previous_outcomes"]) == 2
    assert res["previous_outcomes"][0]["outcome_score"] > 0


def test_affective_trajectory_deteriorating():
    """Test trajectory analyzer detects deteriorating emotional trend."""
    turns = [
        {"turn_id": 1, "speaker": "user", "emotion": {"valence": 0.20, "arousal": 0.2}},
        {"turn_id": 2, "speaker": "assistant", "strategy": {"selected_strategy": "Question"}},
        {"turn_id": 3, "speaker": "user", "emotion": {"valence": -0.30, "arousal": 0.5}},
        {"turn_id": 4, "speaker": "assistant", "strategy": {"selected_strategy": "Question"}},
        {"turn_id": 5, "speaker": "user", "emotion": {"valence": -0.75, "arousal": 0.7}},
    ]

    res = TrajectoryAnalyzer.analyze_trajectory(turns, current_valence=-0.85)
    assert res["valence_trend"] == ValenceTrend.DETERIORATING


def test_privacy_pii_scrubbing():
    """Test PII detection and scrubbing for email, phone, SSN, and card."""
    privacy = PrivacyEngine()
    sensitive_text = (
        "Contact me at alice@example.com or call 555-123-4567. "
        "My SSN is 123-45-6789 and card is 4111-2222-3333-4444."
    )
    scrubbed, count = privacy.scrub_text(sensitive_text)

    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_CARD]" in scrubbed
    assert "alice@example.com" not in scrubbed
    assert "555-123-4567" not in scrubbed
    assert count == 4


def test_privacy_anonymization_pseudonyms(sample_user_input, sample_emotion):
    """Test ANONYMIZED privacy level pseudonymizes user_id and session_id with salted HMAC."""
    engine = MultimodalContextEngine()
    ctx = engine.build_context(
        user_input=sample_user_input,
        emotion=sample_emotion,
        privacy_level=PrivacyLevel.ANONYMIZED
    )

    assert ctx.conversation.user_id.startswith("anon_")
    assert ctx.conversation.session_id.startswith("anon_")
    assert ctx.metadata.privacy["privacy_level"] == "ANONYMIZED"
    assert ctx.metadata.privacy["is_redacted"] is True


def test_verbal_masking_discrepancy_detection(sample_user_input):
    """Test that conflicting voice affect and text affect flags discrepancy."""
    engine = MultimodalContextEngine()

    # Cheerful words (valence +0.6)
    text_emo = EmotionResult(
        primary_emotion=EmotionCategory.JOY,
        confidence=0.85,
        valence=0.60,
        arousal=0.50
    )

    # Distressed / sad voice acoustics (valence -0.5, high arousal 0.7)
    voice_res = VoiceResult(
        is_available=True,
        confidence=0.80,
        valence=-0.50,
        arousal=0.70,
        primary_emotion=EmotionCategory.SADNESS
    )

    ctx = engine.build_context(
        user_input=sample_user_input,
        emotion=text_emo,
        voice=voice_res
    )

    assert ctx.emotion.is_discrepancy_detected is True


def test_strategy_planner_consumes_context(sample_user_input, sample_emotion, sample_symptoms):
    """Verify that StrategyPlaceholderPlanner adapts its behavior when provided with MultimodalContext."""
    planner = StrategyPlaceholderPlanner()
    engine = MultimodalContextEngine()

    # Case 1: Prior turn asked a question -> planner should transition to reflection
    history_turns = [
        {"turn_id": 1, "speaker": "user", "emotion": {"valence": -0.4}},
        {"turn_id": 2, "speaker": "assistant", "strategy": {"selected_strategy": "Question", "stage": "Exploration"}},
    ]
    ctx = engine.build_context(
        user_input=sample_user_input,
        emotion=sample_emotion,
        symptoms=sample_symptoms,
        history=history_turns,
        dialog_stage=DialogStage.COMFORTING
    )

    strat = planner.predict_strategy(
        sample_user_input,
        sample_emotion,
        sample_symptoms,
        multimodal_context=ctx
    )

    assert strat.selected_strategy == SupportStrategy.REFLECTION_OF_FEELINGS
    assert "Progressing from exploration to emotional reflection" in strat.rationale


def test_to_conversation_context_conversion(sample_user_input, sample_emotion):
    """Test backwards-compatibility converter from MultimodalContext to ConversationContext."""
    engine = MultimodalContextEngine()
    ctx = engine.build_context(user_input=sample_user_input, emotion=sample_emotion)

    legacy_conv = MultimodalContextEngine.to_conversation_context(ctx)
    assert legacy_conv.session_id == sample_user_input.session_id
    assert legacy_conv.user_id == sample_user_input.user_id
    assert legacy_conv.metadata["context_id"] == ctx.metadata.context_id


def test_context_placeholder_engine(sample_user_input):
    """Verify fallback ContextPlaceholderEngine builds valid context with placeholder version."""
    placeholder = ContextPlaceholderEngine()
    ctx = placeholder.build_context(user_input=sample_user_input)

    assert isinstance(ctx, MultimodalContext)
    assert ctx.metadata.engine_version == "placeholder-v0.1.0"
