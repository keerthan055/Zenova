"""Unit tests verifying placeholder modules explicitly identify themselves as placeholders."""
from zenova.schemas.standard import UserInput, EmotionCategory, RiskLevel, SupportStrategy
from zenova.emotion.placeholder import EmotionPlaceholderAnalyzer
from zenova.symptoms.placeholder import SymptomPlaceholderAnalyzer
from zenova.risk.placeholder import RiskPlaceholderAnalyzer
from zenova.baseline.placeholder import BaselinePlaceholderEngine
from zenova.behavior.placeholder import BehaviorPlaceholderAnalyzer
from zenova.voice.placeholder import VoicePlaceholderAnalyzer
from zenova.context.placeholder import ContextPlaceholderEngine
from zenova.strategy.placeholder import StrategyPlaceholderPlanner
from zenova.generation.placeholder import ResponsePlaceholderGenerator
from zenova.safety.placeholder import SafetyPlaceholderGate


def test_emotion_placeholder():
    analyzer = EmotionPlaceholderAnalyzer()
    inp = UserInput(session_id="s1", user_id="u1", text="Hello")
    res = analyzer.analyze(inp)
    assert res.is_placeholder is True
    assert res.module_version.startswith("placeholder")
    assert res.primary_emotion == EmotionCategory.NEUTRAL


def test_symptom_placeholder():
    analyzer = SymptomPlaceholderAnalyzer()
    inp = UserInput(session_id="s1", user_id="u1", text="I feel bad")
    res = analyzer.analyze(inp)
    assert res.is_placeholder is True
    assert "Placeholder" in res.disclaimer


def test_risk_placeholder_normal_and_crisis():
    analyzer = RiskPlaceholderAnalyzer()
    normal_inp = UserInput(session_id="s1", user_id="u1", text="I am having a regular day")
    normal_res = analyzer.analyze(normal_inp)
    assert normal_res.is_placeholder is True
    assert not normal_res.is_high_risk

    crisis_inp = UserInput(session_id="s1", user_id="u1", text="I want to hurt myself")
    crisis_res = analyzer.analyze(crisis_inp)
    assert crisis_res.is_placeholder is True
    assert crisis_res.is_high_risk is True
    assert crisis_res.risk_level == RiskLevel.CRITICAL


def test_strategy_placeholder():
    planner = StrategyPlaceholderPlanner()
    inp = UserInput(session_id="s1", user_id="u1", text="Help")
    emotion = EmotionPlaceholderAnalyzer().analyze(inp)
    symptoms = SymptomPlaceholderAnalyzer().analyze(inp)
    res = planner.predict_strategy(inp, emotion, symptoms)
    assert res.is_placeholder is True
    assert res.selected_strategy == SupportStrategy.QUESTION


def test_response_generator_placeholder():
    gen = ResponsePlaceholderGenerator()
    inp = UserInput(session_id="s1", user_id="u1", text="Help")
    strat = StrategyPlaceholderPlanner().predict_strategy(
        inp,
        EmotionPlaceholderAnalyzer().analyze(inp),
        SymptomPlaceholderAnalyzer().analyze(inp)
    )
    res = gen.generate(inp, strat)
    assert res.is_placeholder is True
    assert len(res.response_text) > 0


def test_safety_gate_placeholder():
    gate = SafetyPlaceholderGate()
    inp = UserInput(session_id="s1", user_id="u1", text="Help")
    gen_res = ResponsePlaceholderGenerator().generate(
        inp,
        StrategyPlaceholderPlanner().predict_strategy(
            inp,
            EmotionPlaceholderAnalyzer().analyze(inp),
            SymptomPlaceholderAnalyzer().analyze(inp)
        )
    )
    risk_low = RiskPlaceholderAnalyzer().analyze(inp)
    safety_res = gate.verify(inp, gen_res, risk_low)
    assert safety_res.is_placeholder is True
    assert safety_res.is_safe is True


def test_behavior_placeholder():
    analyzer = BehaviorPlaceholderAnalyzer()
    inp = UserInput(session_id="s1", user_id="u1", text="Hello")
    res = analyzer.analyze(inp)
    assert res.is_placeholder is True
    assert res.is_available is False
    assert "Placeholder" in res.anomaly_notes


def test_voice_placeholder():
    analyzer = VoicePlaceholderAnalyzer()
    inp = UserInput(session_id="s1", user_id="u1", text="Hello")
    res = analyzer.analyze(inp)
    assert res.is_placeholder is True
    assert res.is_available is False
    assert res.primary_emotion == EmotionCategory.NEUTRAL


def test_context_placeholder():
    engine = ContextPlaceholderEngine()
    inp = UserInput(session_id="s1", user_id="u1", text="Hello")
    res = engine.build_context(inp)
    assert res.metadata.engine_version.startswith("placeholder")
    assert res.conversation.session_id == "s1"



