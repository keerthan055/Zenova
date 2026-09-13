"""Unit tests for user-facing preferences, check-ins, data export, and purge repository service."""
import uuid
import pytest
from datetime import datetime, timezone
from zenova.db.session import init_db, get_db_session
from zenova.db.repositories import UserRepository, SessionRepository, TurnRepository
from zenova.schemas.user import (
    USER_DISCLAIMER_NOTICE,
    PrivacyLevel,
    CommunicationStyle,
    WellbeingCheckinRequest,
    UserPreferencesUpdateRequest,
)


@pytest.mark.asyncio
async def test_user_preferences_defaults_and_updates():
    """Verify default preferences initialization and selective updates."""
    await init_db()
    uid = f"test_user_prefs_{uuid.uuid4().hex[:8]}"

    async with get_db_session() as db:
        repo = UserRepository(db)
        # Default initialization
        prefs = await repo.get_or_create_preferences(uid)
        assert prefs.user_id == uid
        assert prefs.save_history is True
        assert prefs.enable_voice is True
        assert prefs.enable_wearables is False
        assert prefs.privacy_level == "standard"
        assert prefs.communication_style == "warm_empathic"

        # Update preferences
        updated = await repo.update_preferences(
            uid,
            {
                "save_history": False,
                "enable_wearables": True,
                "privacy_level": "anonymized",
                "communication_style": "reflective"
            }
        )
        assert updated.save_history is False
        assert updated.enable_wearables is True
        assert updated.privacy_level == "anonymized"
        assert updated.communication_style == "reflective"


@pytest.mark.asyncio
async def test_record_and_query_checkin_history():
    """Verify wellbeing check-in persistence and chronological query."""
    await init_db()
    uid = f"test_user_checkin_{uuid.uuid4().hex[:8]}"
    chk1_id = f"chk_{uuid.uuid4().hex[:8]}"
    chk2_id = f"chk_{uuid.uuid4().hex[:8]}"

    async with get_db_session() as db:
        repo = UserRepository(db)

        # Record multiple check-ins
        chk1 = await repo.record_checkin(
            checkin_id=chk1_id,
            user_id=uid,
            mood_score=8,
            valence=0.5,
            sleep_hours=8.0,
            stress_level=2,
            energy_level=4,
            notes="Felt productive today."
        )
        assert chk1.mood_score == 8

        chk2 = await repo.record_checkin(
            checkin_id=chk2_id,
            user_id=uid,
            mood_score=5,
            valence=0.0,
            sleep_hours=6.5,
            stress_level=3,
            energy_level=3,
            notes="A bit tired."
        )
        assert chk2.mood_score == 5

        # Query history
        history = await repo.get_checkin_history(uid, limit=10)
        assert len(history) >= 2
        # Verify newest first
        assert history[0].checkin_id == chk2_id
        assert history[1].checkin_id == chk1_id


@pytest.mark.asyncio
async def test_export_user_data():
    """Verify GDPR/HIPAA portable data export aggregation."""
    await init_db()
    uid = f"test_user_export_{uuid.uuid4().hex[:8]}"
    sid = f"sess_export_{uuid.uuid4().hex[:8]}"
    chk_id = f"chk_{uuid.uuid4().hex[:8]}"

    async with get_db_session() as db:
        u_repo = UserRepository(db)
        s_repo = SessionRepository(db)
        t_repo = TurnRepository(db)

        # Create session and turn
        await s_repo.get_or_create(session_id=sid, user_id=uid)
        await t_repo.record_turn(
            session_id=sid,
            turn_id=1,
            speaker="user",
            content="I am reflecting on my progress."
        )

        # Create check-in
        await u_repo.record_checkin(
            checkin_id=chk_id,
            user_id=uid,
            mood_score=7,
            valence=0.3,
            sleep_hours=7.0
        )

        # Export
        export = await u_repo.export_user_data(uid)
        assert export["user_id"] == uid
        assert "export_id" in export
        assert "preferences" in export
        assert export["preferences"]["save_history"] is True
        assert len(export["checkins"]) >= 1
        assert export["sessions_count"] >= 1
        assert export["sessions"][0]["session_id"] == sid
        assert len(export["sessions"][0]["turns"]) >= 1


@pytest.mark.asyncio
async def test_purge_user_data_right_to_be_forgotten():
    """Verify permanent deletion of user sessions, turns, and check-ins."""
    await init_db()
    uid = f"test_user_purge_{uuid.uuid4().hex[:8]}"
    sid = f"sess_purge_{uuid.uuid4().hex[:8]}"
    chk_id = f"chk_{uuid.uuid4().hex[:8]}"

    async with get_db_session() as db:
        u_repo = UserRepository(db)
        s_repo = SessionRepository(db)
        t_repo = TurnRepository(db)

        # Ingest data
        await s_repo.get_or_create(session_id=sid, user_id=uid)
        await t_repo.record_turn(session_id=sid, turn_id=1, speaker="user", content="Temporary message")
        await u_repo.record_checkin(checkin_id=chk_id, user_id=uid, mood_score=6)

        # Confirm data exists
        sessions_before = await u_repo.list_user_sessions(uid)
        checkins_before = await u_repo.get_checkin_history(uid)
        assert len(sessions_before) >= 1
        assert len(checkins_before) >= 1

        # Purge
        res = await u_repo.purge_user_data(uid)
        assert res["deleted_sessions"] >= 1
        assert res["deleted_turns"] >= 1
        assert res["deleted_checkins"] >= 1

        # Confirm deleted
        sessions_after = await u_repo.list_user_sessions(uid)
        checkins_after = await u_repo.get_checkin_history(uid)
        assert len(sessions_after) == 0
        assert len(checkins_after) == 0


def test_disclaimer_notice_text():
    """Verify non-medical disclaimer notice is comprehensive and references crisis resources."""
    assert "NOT a licensed healthcare provider" in USER_DISCLAIMER_NOTICE
    assert "988" in USER_DISCLAIMER_NOTICE
    assert "medical diagnoses" in USER_DISCLAIMER_NOTICE
