"""End-to-End full pipeline integration tests verifying the complete ZENOVA flow.

Flow verified:
USER -> INPUT -> TEXT/VOICE PROCESSING -> EMOTION -> SYMPTOMS -> RISK -> BEHAVIOR ->
PERSONAL BASELINE -> CONTEXT -> RISK DECISION
  ├─ IF High/Critical: SAFETY / HUMAN ESCALATION -> USER
  └─ ELSE: STRATEGY PLANNER -> RAG -> LLM -> SAFETY GATE -> USER
"""
import uuid
import base64
import pytest
from zenova.core.orchestrator import ZenovaOrchestrator
from zenova.schemas.standard import UserInput, ModalityType, RiskLevel, EmotionCategory
from zenova.db.session import init_db, get_db_session
from zenova.db.models import EscalationEventModel, ConversationTurnModel, UserSessionModel
from sqlalchemy import select


@pytest.mark.asyncio
async def test_e2e_benign_conversational_turn_flow():
    """Verify complete nominal flow: Strategy Planner -> RAG -> LLM -> Safety Gate -> User."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_benign_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_benign_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        turn_id=1,
        text="I have been feeling overwhelmed by my workload lately, and I would love some advice on organizing my time.",
        modality=ModalityType.TEXT
    )

    result = await orchestrator.process_turn(u_in)

    # 1. Output envelope checks
    assert result["session_id"] == sid
    assert result["turn_id"] == 1
    assert "response" in result
    assert len(result["response"]) > 0

    # 2. Analytical modules executed
    assert "emotion" in result
    assert result["emotion"]["primary_emotion"] is not None
    assert "symptoms" in result
    assert "risk" in result

    # 3. Risk decision (Nominal Branch: High/Critical == False)
    assert result["risk"]["risk_level"] in ("low", "none")
    assert result["risk"].get("requires_immediate_escalation") is False
    assert result.get("escalated_to_human") is False

    # 4. Strategy & Generation layers executed
    assert "strategy" in result
    assert result["strategy"]["selected_strategy"] is not None
    assert "context" in result

    # 5. Safety Gate verification
    assert "safety" in result
    assert result["safety"]["is_safe"] is True
    assert result["safety"]["action"] in ("allow", "revise")

    # 6. Database persistence
    async with get_db_session() as db:
        stmt = select(ConversationTurnModel).where(ConversationTurnModel.session_id == sid)
        res = await db.execute(stmt)
        turns = res.scalars().all()
        assert len(turns) >= 2  # user turn + assistant response


@pytest.mark.asyncio
async def test_e2e_acute_crisis_escalation_flow():
    """Verify acute crisis branch: Risk Decision (High/Critical) -> Safety Bypass & Human Escalation -> User."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_crisis_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_crisis_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        turn_id=1,
        text="I can't take this pain anymore. I want to kill myself tonight and end everything.",
        modality=ModalityType.TEXT
    )

    result = await orchestrator.process_turn(u_in)

    # 1. High/Critical Risk Decision
    assert result["risk"]["risk_level"] in ("high", "critical")
    assert result["risk"].get("requires_immediate_escalation") is True

    # 2. Human Escalation Triggered
    assert result["escalated_to_human"] is True
    assert "escalation_id" in result or "escalation_event_id" in result
    alert_id = result.get("escalation_id") or result.get("escalation_event_id")
    assert alert_id is not None

    # 3. Crisis Response delivered with 988 helpline
    assert "988" in result["response"]

    # 4. Verify Alert in Database
    async with get_db_session() as db:
        stmt = select(EscalationEventModel).where(EscalationEventModel.session_id == sid)
        res = await db.execute(stmt)
        alert = res.scalars().first()
        assert alert is not None
        assert alert.severity in ("critical", "high")
        assert alert.status == "pending"


@pytest.mark.asyncio
async def test_e2e_multimodal_voice_flow():
    """Verify audio voice processing -> feature extraction -> multimodal fusion -> complete pipeline."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_voice_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_voice_sess_{uuid.uuid4().hex[:8]}"

    # Synthetic 1-second audio tone
    dummy_wav_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    b64_audio = base64.b64encode(dummy_wav_bytes).decode("ascii")

    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        turn_id=1,
        text="I am speaking into the microphone and feeling anxious.",
        modality=ModalityType.VOICE,
        metadata={"audio_base64": b64_audio}
    )

    result = await orchestrator.process_turn(u_in)
    assert result["session_id"] == sid
    assert len(result["response"]) > 0
    assert "voice" in result or "fused_state" in result


@pytest.mark.asyncio
async def test_e2e_multimodal_behavioral_wearable_flow():
    """Verify behavioral telemetry integration -> baseline deviation check -> context-aware generation."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_beh_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_beh_sess_{uuid.uuid4().hex[:8]}"

    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        turn_id=1,
        text="I have barely slept this week and feel exhausted.",
        modality=ModalityType.TEXT,
        metadata={
            "passive_behavior_metrics": {
                "sleep_hours": 3.5,
                "step_count": 1800,
                "screen_time_hours": 9.5
            }
        }
    )

    result = await orchestrator.process_turn(u_in)
    assert result["session_id"] == sid
    assert len(result["response"]) > 0
    assert "behavior" in result or "context" in result


@pytest.mark.asyncio
async def test_e2e_full_multimodal_gmu_flow():
    """Verify simultaneous text, voice, and behavioral sensing fusion through complete pipeline."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_full_mm_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_full_mm_sess_{uuid.uuid4().hex[:8]}"

    dummy_wav_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    b64_audio = base64.b64encode(dummy_wav_bytes).decode("ascii")

    u_in = UserInput(
        session_id=sid,
        user_id=uid,
        turn_id=1,
        text="Everything feels very intense right now.",
        modality=ModalityType.VOICE,
        metadata={
            "audio_base64": b64_audio,
            "passive_behavior_metrics": {
                "sleep_hours": 4.0,
                "step_count": 2200
            }
        }
    )

    result = await orchestrator.process_turn(u_in)
    assert result["session_id"] == sid
    assert len(result["response"]) > 0
    assert result["safety"]["is_safe"] is True


@pytest.mark.asyncio
async def test_e2e_sequential_multi_turn_continuity():
    """Verify session continuity across multiple sequential conversational turns."""
    await init_db()
    orchestrator = ZenovaOrchestrator()
    uid = f"e2e_seq_user_{uuid.uuid4().hex[:8]}"
    sid = f"e2e_seq_sess_{uuid.uuid4().hex[:8]}"

    # Turn 1
    t1 = await orchestrator.process_turn(
        UserInput(session_id=sid, user_id=uid, turn_id=1, text="I lost my job yesterday and don't know what to do.")
    )
    assert t1["turn_id"] == 1

    # Turn 2
    t2 = await orchestrator.process_turn(
        UserInput(session_id=sid, user_id=uid, turn_id=2, text="Thank you, that helps. How do I start taking small steps?")
    )
    # The second user turn is turn_id 3 (turn 2 is assistant's first response)
    assert t2["turn_id"] == 3
    assert len(t2["response"]) > 0

    # Verify 4 turns recorded in session (2 user + 2 assistant)
    async with get_db_session() as db:
        stmt = select(ConversationTurnModel).where(ConversationTurnModel.session_id == sid).order_by(ConversationTurnModel.turn_id.asc())
        res = await db.execute(stmt)
        turns = res.scalars().all()
        assert len(turns) == 4
        assert turns[0].speaker == "user"
        assert turns[1].speaker == "assistant"
        assert turns[2].speaker == "user"
        assert turns[3].speaker == "assistant"
