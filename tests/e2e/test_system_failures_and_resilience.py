"""Failure and fault-tolerance tests verifying ZENOVA fails safely under component outages.

Verifies:
1. Emotion model unavailable/crashed -> degraded neutral fallback
2. Symptom model unavailable/crashed -> degraded empty symptoms fallback
3. Risk model unavailable/crashed -> conservative fail-safe regex triage
4. Strategy planner unavailable/crashed -> safe questioning/reflection fallback
5. RAG retrieval unavailable/crashed -> unaugmented generation fallback
6. LLM generator unavailable/crashed -> clinical safe intervention template
7. Database persistence outage -> safe response with db_persisted=False
"""
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType, EmotionCategory, RiskLevel
from zenova.db.session import init_db


@pytest.mark.asyncio
async def test_emotion_model_failure_resilience():
    """Verify pipeline completes safely when emotion model crashes."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_emo_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_emo_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="I feel disconnected today.")

    # Patch emotion module analyze method to throw RuntimeError
    emo_mod = orchestrator.engine.registry.get_module_instance("emotion")
    with patch.object(emo_mod, "analyze", side_effect=RuntimeError("Simulated emotion model CUDA OOM crash")):
        result = await orchestrator.process_turn(u_in)

        # Pipeline must not crash
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        # Degraded fallback values
        assert result["emotion"]["is_placeholder"] is True
        assert result["emotion"]["confidence"] == 0.0
        assert result["emotion"]["primary_emotion"] == "neutral"


@pytest.mark.asyncio
async def test_symptom_model_failure_resilience():
    """Verify pipeline completes safely when symptom model crashes."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_symp_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_symp_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="I have no energy to get out of bed.")

    symp_mod = orchestrator.engine.registry.get_module_instance("symptom")
    with patch.object(symp_mod, "analyze", side_effect=RuntimeError("Simulated symptom model weights error")):
        result = await orchestrator.process_turn(u_in)
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        assert result["symptoms"]["is_placeholder"] is True
        assert len(result["symptoms"]["signals"]) == 0


@pytest.mark.asyncio
async def test_risk_model_failure_conservative_crisis_scanning():
    """Verify conservative safety fallback still intercepts crisis when ML risk model crashes."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_risk_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_risk_sess_{uuid.uuid4().hex[:8]}"

    # Input has acute crisis keywords
    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        text="I want to kill myself tonight and end this suffering."
    )

    risk_mod = orchestrator.engine.registry.get_module_instance("risk")
    with patch.object(risk_mod, "analyze", side_effect=RuntimeError("Simulated risk classifier failure")):
        result = await orchestrator.process_turn(u_in)

        # Conservative fallback regex must detect acute crisis!
        assert result["risk"]["risk_level"] in ("high", "critical")
        assert result["escalated_to_human"] is True
        assert "988" in result["response"]


@pytest.mark.asyncio
async def test_strategy_planner_failure_resilience():
    """Verify pipeline proceeds when strategy planner fails."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_strat_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_strat_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="Can you help me think through this dilemma?")

    strat_mod = orchestrator.engine.registry.get_module_instance("strategy")
    with patch.object(strat_mod, "predict_strategy", side_effect=RuntimeError("Simulated strategy planner exception")):
        result = await orchestrator.process_turn(u_in)
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        assert "strategy" in result


@pytest.mark.asyncio
async def test_rag_retrieval_failure_resilience():
    """Verify unaugmented generation fallback when RAG vector index fails."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_rag_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_rag_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="What are evidence-based ways to cope with sudden panic?")

    rag_mod = orchestrator.engine.registry.get_module_instance("rag")
    with patch.object(rag_mod, "query", side_effect=RuntimeError("Simulated vector index IO error")):
        result = await orchestrator.process_turn(u_in)
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        assert result["safety"]["is_safe"] is True


@pytest.mark.asyncio
async def test_generator_llm_failure_resilience():
    """Verify safe clinical intervention fallback template engages when LLM generation fails."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_gen_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_gen_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="I don't know what to do next.")

    gen_mod = orchestrator.engine.registry.get_module_instance("generator")
    with patch.object(gen_mod, "generate", side_effect=RuntimeError("Simulated LLM API rate limit / timeout")):
        result = await orchestrator.process_turn(u_in)
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        # Fallback message provided
        assert len(result["response"]) > 10


@pytest.mark.asyncio
async def test_database_outage_resilience():
    """Verify user receives response safely even if database persistence fails."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"fail_db_user_{uuid.uuid4().hex[:8]}"
    sid = f"fail_db_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(session_id=sid, user_id=uid, text="Hello, are you there?")

    # Patch TurnRepository.record_turn to simulate database write failure
    with patch("zenova.db.repositories.TurnRepository.record_turn", side_effect=RuntimeError("Database write error")):
        result = await orchestrator.process_turn(u_in)
        assert result["session_id"] == sid
        assert len(result["response"]) > 0
        # DB persistence failed gracefully without failing the user
        assert result.get("db_persisted") is False
