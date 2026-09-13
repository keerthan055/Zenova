"""Integration test for ZENOVA module interface contracts and high-risk bypass."""
import pytest
from typing import List, Dict, Optional, Any

from zenova.core.interfaces import (
    BaseAnalyzer,
    BaseStrategyPlanner,
    BaseSafetyGate,
    BaseResponseGenerator
)
from zenova.schemas.pipeline import ZenovaUserTurn, ZenovaPipelineOutput
from zenova.schemas.emotion import EmotionAnalysisResult, EmotionCategory
from zenova.schemas.risk import CrisisRiskAssessment, RiskLevel, CrisisCategory
from zenova.schemas.strategy import SupportStrategy, DialogStage, StrategyPrediction
from zenova.schemas.safety import SafetyGateResult, SafetyAction
from zenova.schemas.generation import GenerationResult


class MockEmotionAnalyzer(BaseAnalyzer):
    def analyze(self, user_turn: ZenovaUserTurn, context: Optional[Any] = None) -> EmotionAnalysisResult:
        return EmotionAnalysisResult(
            primary_emotion=EmotionCategory.SADNESS,
            confidence=0.90,
            probabilities={"sadness": 0.90, "neutral": 0.10}
        )


class MockRiskAnalyzer(BaseAnalyzer):
    def analyze(self, user_turn: ZenovaUserTurn, context: Optional[Any] = None) -> CrisisRiskAssessment:
        if "hurt myself" in user_turn.text.lower() or "end it all" in user_turn.text.lower():
            return CrisisRiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                crisis_category=CrisisCategory.SUICIDAL_IDEATION,
                confidence=0.99,
                requires_immediate_escalation=True,
                escalation_action="crisis_hotline_override"
            )
        return CrisisRiskAssessment(
            risk_level=RiskLevel.LOW,
            crisis_category=CrisisCategory.NONE,
            confidence=0.95,
            requires_immediate_escalation=False
        )


class MockStrategyPlanner(BaseStrategyPlanner):
    def predict_strategy(
        self,
        dialogue_history: List[Dict[str, str]],
        current_turn: str,
        seeker_state: Optional[Any] = None
    ) -> StrategyPrediction:
        return StrategyPrediction(
            selected_strategy=SupportStrategy.REFLECTION_OF_FEELINGS,
            confidence=0.82,
            stage=DialogStage.EXPLORATION,
            rationale="Seeker is expressing deep sadness; reflection helps them feel heard."
        )


class MockSafetyGate(BaseSafetyGate):
    def verify(self, candidate_response: str, risk_assessment: Any, user_turn: Any) -> SafetyGateResult:
        if risk_assessment.is_high_risk:
            return SafetyGateResult(
                is_safe=False,
                action=SafetyAction.BLOCK_AND_ESCALATE,
                explanation="Critical risk detected; bypass standard generation with crisis resources."
            )
        return SafetyGateResult(
            is_safe=True,
            action=SafetyAction.ALLOW,
            modified_text=candidate_response
        )


def run_test_pipeline(user_turn: ZenovaUserTurn) -> ZenovaPipelineOutput:
    emotion_analyzer = MockEmotionAnalyzer()
    risk_analyzer = MockRiskAnalyzer()
    strategy_planner = MockStrategyPlanner()
    safety_gate = MockSafetyGate()

    # 1. Run parallel analyzers
    emotion_res = emotion_analyzer.analyze(user_turn)
    risk_res = risk_analyzer.analyze(user_turn)

    # 2. Check high-risk bypass
    if risk_res.is_high_risk:
        safety_res = safety_gate.verify("", risk_res, user_turn)
        crisis_response = (
            "I'm hearing how much pain you're in, and your safety is the most important thing. "
            "Please connect with trained support right away: Call or text 988 (USA/Canada) or visit https://findahelpline.com."
        )
        return ZenovaPipelineOutput(
            session_id=user_turn.session_id,
            turn_id=user_turn.turn_id,
            user_input=user_turn.text,
            emotion=emotion_res,
            risk=risk_res,
            safety=safety_res,
            final_response=crisis_response,
            escalated_to_human=True,
            escalation_reason=risk_res.escalation_action
        )

    # 3. Standard pipeline: Strategy selection -> Safe Generation
    strategy_res = strategy_planner.predict_strategy([], user_turn.text)
    candidate = "It sounds like you are carrying a great deal of exhaustion right now."
    safety_res = safety_gate.verify(candidate, risk_res, user_turn)

    return ZenovaPipelineOutput(
        session_id=user_turn.session_id,
        turn_id=user_turn.turn_id,
        user_input=user_turn.text,
        emotion=emotion_res,
        risk=risk_res,
        strategy=strategy_res,
        safety=safety_res,
        final_response=safety_res.modified_text or candidate,
        escalated_to_human=False
    )


def test_standard_pipeline_flow(sample_user_turn):
    output = run_test_pipeline(sample_user_turn)
    assert not output.escalated_to_human
    assert output.strategy is not None
    assert output.strategy.selected_strategy == SupportStrategy.REFLECTION_OF_FEELINGS
    assert output.emotion.primary_emotion == EmotionCategory.SADNESS


def test_high_risk_bypass_flow():
    crisis_turn = ZenovaUserTurn(
        session_id="crisis_sess_1",
        user_id="user_crisis",
        turn_id=1,
        text="I can't take this anymore, I just want to hurt myself and end it all."
    )
    output = run_test_pipeline(crisis_turn)
    assert output.escalated_to_human is True
    assert output.risk.is_high_risk is True
    assert "988" in output.final_response
    assert output.safety.action == SafetyAction.BLOCK_AND_ESCALATE
