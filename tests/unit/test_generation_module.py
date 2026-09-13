"""Unit tests for strategy-controlled response generation module."""
import pytest
from zenova.schemas.standard import (
    UserInput,
    StrategyResult,
    SupportStrategy,
    DialogStage,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSignal,
    SymptomSeverity,
    GeneratedResponse
)
from zenova.generation.prompt import StrategyPromptBuilder, STRATEGY_DIRECTIVES
from zenova.generation.validator import ResponseValidator
from zenova.generation.providers import (
    LocalFallbackProvider,
    get_llm_provider,
    list_available_providers
)
from zenova.generation.generator import StrategyControlledGenerator


def test_strategy_directives_completeness():
    """Verify all 8 canonical ESConv strategies have comprehensive behavioral directives."""
    assert len(STRATEGY_DIRECTIVES) == 8
    for strat in SupportStrategy:
        assert strat.value in STRATEGY_DIRECTIVES
        data = STRATEGY_DIRECTIVES[strat.value]
        assert "objective" in data
        assert len(data["dos"]) >= 2
        assert len(data["donts"]) >= 2


def test_prompt_builder_reflection_strategy_control():
    """Verify reflection strategy explicitly forbids advice and solutions in prompt directives."""
    builder = StrategyPromptBuilder()
    user_in = UserInput(session_id="s1", user_id="u1", text="I feel like giving up on my exams.")
    strat_res = StrategyResult(
        selected_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
        confidence=0.88,
        stage=DialogStage.COMFORTING,
        rationale="Comforting stage: emotional validation required."
    )

    prompts = builder.build_prompt(user_input=user_in, strategy=strat_res)
    sys_p = prompts["system_prompt"]
    user_p = prompts["user_prompt"]

    # System prompt guardrails
    assert "NON-DIAGNOSTIC MANDATE" in sys_p
    assert "NON-THERAPIST MANDATE" in sys_p
    assert "STRATEGY FIDELITY" in sys_p

    # User prompt directives
    assert "STRATEGY: Reflection of feelings" in user_p
    assert "DO NOT offer coping advice, solutions, or actionable steps" in user_p
    assert "Seeker: I feel like giving up on my exams." in user_p


def test_prompt_builder_question_strategy_control():
    """Verify question strategy explicitly directs inquiry and forbids unsolicited advice."""
    builder = StrategyPromptBuilder()
    user_in = UserInput(session_id="s1", user_id="u1", text="I am overwhelmed.")
    strat_res = StrategyResult(
        selected_strategy=SupportStrategy.QUESTION,
        confidence=0.92,
        stage=DialogStage.EXPLORATION
    )

    prompts = builder.build_prompt(user_input=user_in, strategy=strat_res)
    user_p = prompts["user_prompt"]

    assert "STRATEGY: Question" in user_p
    assert "DO NOT offer advice, solutions, or action items" in user_p


def test_prompt_builder_multimodal_signals_and_rag():
    """Verify builder correctly embeds emotion, symptoms, and RAG grounding."""
    from zenova.schemas.standard import MultimodalContext
    from zenova.context.engine import MultimodalContextEngine

    engine = MultimodalContextEngine()
    user_in = UserInput(session_id="s1", user_id="u1", text="I cannot sleep and feel constant panic.")
    emo_res = EmotionResult(
        primary_emotion=EmotionCategory.ANXIETY,
        confidence=0.90,
        valence=-0.7,
        arousal=0.8,
        dominance=-0.5
    )
    sym_res = SymptomResult(
        signals=[
            SymptomSignal(marker_name="sleep_disturbance", severity=SymptomSeverity.MODERATE, confidence=0.85),
            SymptomSignal(marker_name="anxiety_panic", severity=SymptomSeverity.MODERATE, confidence=0.88)
        ],
        clinical_disclaimer="Non-diagnostic."
    )

    mm_ctx = engine.build_context(user_input=user_in, emotion=emo_res, symptoms=sym_res)
    strat_res = StrategyResult(selected_strategy=SupportStrategy.PROVIDING_SUGGESTIONS, confidence=0.8)

    rag = ["Box Breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s.", "Sleep hygiene: Keep room cool and dark."]
    builder = StrategyPromptBuilder()
    prompts = builder.build_prompt(
        user_input=user_in,
        strategy=strat_res,
        rag_context=rag,
        multimodal_context=mm_ctx
    )
    user_p = prompts["user_prompt"]

    assert "Inferred Emotion: anxiety" in user_p
    assert "sleep_disturbance" in user_p
    assert "Box Breathing" in user_p


def test_local_fallback_provider_all_strategies():
    """Verify local offline provider produces coherent responses for every strategy."""
    prov = LocalFallbackProvider()
    for strat in SupportStrategy:
        prompt = f"STRATEGY: {strat.value}\nSeeker: I am going through a difficult time."
        resp = prov.generate(prompt)
        assert resp.text is not None
        assert len(resp.text) > 20
        assert resp.provider == "local"
        assert resp.tokens_used is not None and resp.tokens_used > 0


def test_response_validator_blocks_diagnosis():
    validator = ResponseValidator()
    # Forbidden diagnosis
    bad_text = "Based on what you said, you have clinical depression and need medication."
    res = validator.validate(bad_text, SupportStrategy.REFLECTION_OF_FEELINGS)
    assert res.is_valid is False
    assert "PROHIBITED_MEDICAL_DIAGNOSIS" in res.violations


def test_response_validator_blocks_therapist_claim():
    validator = ResponseValidator()
    # Forbidden role claim
    bad_text = "As your therapist, I want you to talk about your childhood trauma."
    res = validator.validate(bad_text, SupportStrategy.QUESTION)
    assert res.is_valid is False
    assert "FALSE_THERAPIST_CLAIM" in res.violations


def test_response_validator_accepts_clean_response():
    validator = ResponseValidator()
    good_text = "It sounds like you are carrying a tremendous amount of weight right now, and that feels deeply exhausting."
    res = validator.validate(good_text, SupportStrategy.REFLECTION_OF_FEELINGS)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_response_validator_sanitizes_logs():
    validator = ResponseValidator()
    log_snippet = validator.sanitize_log_content("User told me their confidential secret password and medical issues", max_words=4)
    assert "User told me their..." in log_snippet


def test_strategy_controlled_generator_lifecycle():
    """Verify end-to-end strategy-controlled response generation with local provider."""
    generator = StrategyControlledGenerator(provider_name="local")
    user_in = UserInput(session_id="gen-test-session", user_id="gen-test-user", text="I feel completely stuck at work.")
    strat_res = StrategyResult(
        selected_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
        confidence=0.84,
        stage=DialogStage.COMFORTING,
        rationale="Validation of emotional exhaustion."
    )

    response = generator.generate(user_input=user_in, strategy=strat_res)
    assert isinstance(response, GeneratedResponse)
    assert response.is_placeholder is False
    assert response.strategy_applied == SupportStrategy.REFLECTION_OF_FEELINGS
    assert len(response.response_text) > 15
    assert response.validation_passed is True
    assert response.latency_ms >= 0.0


def test_provider_factory_fallback():
    """Verify requesting cloud provider without credentials gracefully falls back to local."""
    prov = get_llm_provider("openai")
    # If no OPENAI_API_KEY, should be LocalFallbackProvider
    assert isinstance(prov, LocalFallbackProvider)

    providers = list_available_providers()
    assert len(providers) >= 4
    names = [p["name"] for p in providers]
    assert "local" in names
    assert "openai" in names
    assert "anthropic" in names
    assert "gemini" in names
