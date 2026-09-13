"""Unit tests for ZENOVA End-to-End Orchestration Engine and Failure Handling."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from zenova.schemas.standard import (
    UserInput,
    EmotionResult,
    EmotionCategory,
    SymptomResult,
    SymptomSeverity,
    RiskResult,
    RiskLevel,
    CrisisCategory,
    SupportStrategy,
    GeneratedResponse,
    SafetyResult,
    SafetyAction
)
from zenova.schemas.orchestration import (
    PipelineStatus,
    SpanStatus,
    PipelineTrace
)
from zenova.orchestration.engine import ZenovaOrchestrationEngine
from zenova.orchestration.tracing.tracer import PipelineTracer
from zenova.orchestration.tracing.buffer import InMemoryTraceBuffer, get_trace_buffer
from zenova.orchestration.fallbacks import (
    get_degraded_emotion_result,
    get_degraded_symptom_result,
    get_safe_fallback_risk_result,
    get_safe_fallback_response
)
from zenova.models.registry import ModelRegistry


import uuid


@pytest.mark.asyncio
async def test_orchestration_nominal_turn():
    """Verify nominal conversational turn runs end-to-end and produces an execution trace."""
    engine = ZenovaOrchestrationEngine()
    sess_id = f"unit_sess_nom_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_nominal",
        text="I felt quite productive today after completing my tasks."
    )

    result = await engine.process_turn(user_in)

    assert result["session_id"] == sess_id
    assert result["turn_id"] >= 1
    assert result["user_input"] == "I felt quite productive today after completing my tasks."
    assert isinstance(result["response"], str) and len(result["response"]) > 0
    assert result["escalated_to_human"] is False
    assert result["risk"]["risk_level"] in ["low", "moderate"]

    # Verify execution trace
    assert "trace_id" in result
    assert result["pipeline_status"] in ["nominal", "degraded"]
    assert "trace" in result
    trace = result["trace"]
    assert len(trace["spans"]) >= 4
    span_names = [s["span_name"] for s in trace["spans"]]
    assert "input_processing" in span_names
    assert "analytical_layer" in span_names
    assert "intervention" in span_names
    assert "persistence" in span_names


@pytest.mark.asyncio
async def test_orchestration_crisis_bypass_and_escalation():
    """Verify acute crisis triggers bypass, human escalation alert, and crisis response."""
    engine = ZenovaOrchestrationEngine()
    sess_id = f"unit_sess_crisis_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_crisis",
        text="I cannot take this anymore, I want to kill myself tonight."
    )

    result = await engine.process_turn(user_in)

    assert result["escalated_to_human"] is True
    assert result["escalation_id"] is not None
    assert result["risk"]["risk_level"] in ["high", "critical"]
    assert "988" in result["response"]
    assert "Crisis Text Line" in result["response"]
    assert result["pipeline_status"] == PipelineStatus.CRISIS_BYPASS.value


@pytest.mark.asyncio
async def test_orchestration_emotion_model_failure_fallback():
    """Verify graceful degraded fallback when emotion model throws an error."""
    mock_registry = ModelRegistry()
    mock_emotion = MagicMock()
    mock_emotion.analyze.side_effect = RuntimeError("Emotion inference CUDA crash")
    mock_registry.register_module("emotion", mock_emotion)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_em_fail_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_em_fail",
        text="I am feeling a bit tired."
    )

    result = await engine.process_turn(user_in)

    assert result["response"] is not None
    assert "emotion" in result["degraded_modules"]
    assert result["emotion"]["primary_emotion"] == EmotionCategory.NEUTRAL.value
    assert result["emotion"]["confidence"] == 0.0


@pytest.mark.asyncio
async def test_orchestration_symptom_model_failure_fallback():
    """Verify graceful degraded fallback when symptom model throws an error."""
    mock_registry = ModelRegistry()
    mock_symptom = MagicMock()
    mock_symptom.analyze.side_effect = Exception("Symptom model weights corrupted")
    mock_registry.register_module("symptom", mock_symptom)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_sy_fail_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_sy_fail",
        text="Everything feels hard lately."
    )

    result = await engine.process_turn(user_in)

    assert result["response"] is not None
    assert "symptom" in result["degraded_modules"]
    assert result["symptoms"]["signals"] == []
    assert result["symptoms"]["aggregate_severity"] == SymptomSeverity.NONE.value


@pytest.mark.asyncio
async def test_orchestration_risk_model_failure_with_acute_cue():
    """Verify conservative heuristic fallback flags acute crisis when risk model fails."""
    mock_registry = ModelRegistry()
    mock_risk = MagicMock()
    mock_risk.analyze.side_effect = TimeoutError("Risk service timeout")
    mock_registry.register_module("risk", mock_risk)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_rf_acute_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_risk_fail_acute",
        text="I have decided to overdose and end it all."
    )

    result = await engine.process_turn(user_in)

    assert "risk" in result["degraded_modules"]
    assert result["risk"]["risk_level"] == RiskLevel.CRITICAL.value
    assert result["risk"]["crisis_category"] == CrisisCategory.SUICIDAL_IDEATION.value
    assert result["escalated_to_human"] is True
    assert "988" in result["response"]


@pytest.mark.asyncio
async def test_orchestration_risk_model_failure_normal_cue():
    """Verify conservative heuristic fallback defaults safely on non-acute message."""
    mock_registry = ModelRegistry()
    mock_risk = MagicMock()
    mock_risk.analyze.side_effect = TimeoutError("Risk service timeout")
    mock_registry.register_module("risk", mock_risk)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_rf_mild_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_risk_fail_mild",
        text="I am studying for my exams tomorrow."
    )

    result = await engine.process_turn(user_in)

    assert "risk" in result["degraded_modules"]
    assert result["risk"]["risk_level"] == RiskLevel.LOW.value
    assert result["escalated_to_human"] is False


@pytest.mark.asyncio
async def test_orchestration_llm_failure_serves_clinical_template():
    """Verify system falls back to deterministic clinical template when LLM fails."""
    mock_registry = ModelRegistry()
    mock_generator = MagicMock()
    mock_generator.generate.side_effect = RuntimeError("API key quota exhausted")
    mock_registry.register_module("generator", mock_generator)

    # Ensure risk model returns low risk so we exercise normal support path
    mock_risk = MagicMock()
    mock_risk.analyze.return_value = RiskResult(
        risk_level=RiskLevel.LOW,
        crisis_category=CrisisCategory.NONE,
        confidence=0.95
    )
    mock_registry.register_module("risk", mock_risk)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_gf_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_gen_fail",
        text="I feel sad about my family situation."
    )

    result = await engine.process_turn(user_in)

    assert "generator" in result["degraded_modules"]
    assert result["pipeline_status"] == PipelineStatus.SAFE_FALLBACK.value
    assert len(result["response"]) > 20
    # Response was served safely and verified by SafetyGate
    assert result["safety"]["is_safe"] is True


@pytest.mark.asyncio
async def test_orchestration_rag_failure_ungrounded_continuation():
    """Verify pipeline continues safely without hallucinated citations when RAG fails."""
    mock_registry = ModelRegistry()
    mock_rag = MagicMock()
    mock_rag.query.side_effect = ConnectionError("Vector DB unreachable")
    mock_registry.register_module("rag", mock_rag)

    engine = ZenovaOrchestrationEngine(model_registry=mock_registry)
    sess_id = f"unit_sess_rag_fail_{uuid.uuid4().hex[:8]}"
    user_in = UserInput(
        session_id=sess_id,
        user_id="unit_user_rag_fail",
        text="What are some symptoms of clinical anxiety?"
    )

    result = await engine.process_turn(user_in)

    assert "rag" in result["degraded_modules"]
    assert result["response"] is not None
    assert result["rag"] is None


def test_in_memory_trace_buffer():
    """Verify circular buffer stores, retrieves, and evicts oldest traces."""
    buf = InMemoryTraceBuffer(maxlen=3)
    buf.clear()

    trace1 = PipelineTrace(trace_id="t1", session_id="s1", turn_id=1, user_id="u1", status=PipelineStatus.NOMINAL)
    trace2 = PipelineTrace(trace_id="t2", session_id="s1", turn_id=2, user_id="u1", status=PipelineStatus.DEGRADED)
    trace3 = PipelineTrace(trace_id="t3", session_id="s2", turn_id=1, user_id="u2", status=PipelineStatus.CRISIS_BYPASS)
    trace4 = PipelineTrace(trace_id="t4", session_id="s2", turn_id=2, user_id="u2", status=PipelineStatus.NOMINAL)

    buf.add_trace(trace1)
    buf.add_trace(trace2)
    buf.add_trace(trace3)
    assert len(buf) == 3
    assert buf.get_trace("t1") is not None

    # Adding 4th should evict t1
    buf.add_trace(trace4)
    assert len(buf) == 3
    assert buf.get_trace("t1") is None
    assert buf.get_trace("t4") is not None

    # Filter by user_id
    u2_traces = buf.list_traces(user_id="u2")
    assert len(u2_traces) == 2
    assert u2_traces[0].trace_id == "t4"
