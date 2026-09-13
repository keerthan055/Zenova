import pytest
import uuid
from zenova.db.session import init_db, get_db_session
from zenova.db.repositories import SessionRepository, TurnRepository, EscalationRepository


@pytest.mark.asyncio
async def test_db_lifecycle_and_repositories():
    await init_db()

    test_sess_id = f"test_s_{uuid.uuid4().hex[:8]}"
    test_user_id = f"test_u_{uuid.uuid4().hex[:8]}"
    test_event_id = f"esc_test_{uuid.uuid4().hex[:8]}"

    async with get_db_session() as db:
        session_repo = SessionRepository(db)
        turn_repo = TurnRepository(db)
        esc_repo = EscalationRepository(db)

        # 1. Get or create session
        sess = await session_repo.get_or_create(test_sess_id, test_user_id)
        assert sess.session_id == test_sess_id
        assert sess.user_id == test_user_id

        # 2. Record turn
        turn = await turn_repo.record_turn(
            session_id=test_sess_id,
            turn_id=1,
            speaker="user",
            content="I feel exhausted"
        )
        assert turn.turn_id == 1
        assert turn.content == "I feel exhausted"

        # 3. Fetch history
        history = await turn_repo.get_history(test_sess_id)
        assert len(history) >= 1

        # 4. Create and acknowledge escalation event
        event = await esc_repo.create_event(
            event_id=test_event_id,
            session_id=test_sess_id,
            user_id=test_user_id,
            risk_level="critical",
            crisis_category="suicidal_ideation",
            trigger_cues=["test cue"]
        )
        assert event.event_id == test_event_id
        assert event.status == "pending"

        ack_event = await esc_repo.acknowledge_event(test_event_id, clinician_id="dr_smith", notes="Reviewed")
        assert ack_event.status == "acknowledged"
        assert ack_event.assigned_clinician_id == "dr_smith"
