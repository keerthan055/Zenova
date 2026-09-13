"""Unit tests for Clinician Dashboard Module Registry, Service, RBAC, and Explainability."""
import pytest
from fastapi import HTTPException

from zenova.db.session import init_db, get_db_session
from zenova.db.repositories import TurnRepository, SessionRepository, DashboardRepository
from zenova.schemas.escalation import UserRole
from zenova.schemas.dashboard import DashboardModuleDescriptor, CLINICAL_DISCLAIMER_NOTICE
from zenova.dashboard.registry import DashboardModuleRegistry
from zenova.dashboard.service import ClinicianDashboardService


def test_dashboard_module_registry():
    registry = DashboardModuleRegistry()
    modules = registry.list_modules()
    assert len(modules) >= 10

    # Filter by category
    overview_mods = registry.list_modules(category="overview")
    assert len(overview_mods) >= 3

    # Register custom module
    custom = DashboardModuleDescriptor(
        module_id="eeg_frontal_asymmetry",
        name="Frontal EEG Asymmetry Tracker",
        category="sensor",
        description="Tracks frontal alpha asymmetry biomarker."
    )
    registry.register(custom)
    assert registry.get_module("eeg_frontal_asymmetry") is not None
    assert len(registry.list_modules(category="sensor")) >= 3

    # Unregister
    removed = registry.unregister("eeg_frontal_asymmetry")
    assert removed is not None
    assert registry.get_module("eeg_frontal_asymmetry") is None


@pytest.mark.asyncio
async def test_dashboard_service_patient_forbidden():
    await init_db()
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        with pytest.raises(HTTPException) as exc_info:
            await service.get_overview(actor_role=UserRole.PATIENT, actor_id="patient_123")
        assert exc_info.value.status_code == 403
        assert "Patients are strictly prohibited" in exc_info.value.detail


@pytest.mark.asyncio
async def test_dashboard_service_overview_success():
    await init_db()
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        overview = await service.get_overview(actor_role=UserRole.CLINICIAN, actor_id="dr_smith")
        assert overview.system_status["status"] == "operational"
        assert overview.system_status["database_connected"] is True
        assert "active_alerts_count" in overview.model_dump()
        assert "risk_distribution" in overview.model_dump()
        assert CLINICAL_DISCLAIMER_NOTICE == overview.clinical_notice


@pytest.mark.asyncio
async def test_dashboard_service_patient_masking_for_auditor():
    import uuid
    p_uid = f"real-patient-9999-{uuid.uuid4().hex[:6]}"
    p_sid = f"auditor-test-sess-{uuid.uuid4().hex[:6]}"
    await init_db()
    async with get_db_session() as session:
        sess_repo = SessionRepository(session)
        await sess_repo.get_or_create(session_id=p_sid, user_id=p_uid)

        service = ClinicianDashboardService(session)
        # Clinician sees unmasked
        clinician_res = await service.list_patients(actor_role=UserRole.CLINICIAN, actor_id="dr_smith")
        uids_clinician = [p.user_id for p in clinician_res.patients]
        assert any(p_uid in uid for uid in uids_clinician)

        # Auditor sees masked
        auditor_res = await service.list_patients(actor_role=UserRole.AUDITOR, actor_id="auditor_1")
        uids_auditor = [p.user_id for p in auditor_res.patients]
        assert not any(p_uid == uid for uid in uids_auditor)
        assert any("usr_***" in uid for uid in uids_auditor)


@pytest.mark.asyncio
async def test_dashboard_service_explainability_cards():
    await init_db()
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        res = await service.get_patient_explainability(
            user_id="explain-test-patient",
            actor_role=UserRole.CLINICIAN,
            actor_id="dr_smith"
        )
        assert len(res.model_cards) >= 7
        names = [c.model_name for c in res.model_cards]
        assert "Emotion Transformer" in names
        assert "Multi-Task Crisis & Suicide Risk Transformer" in names
        assert "Response Safety Gate" in names

        for card in res.model_cards:
            assert card.confidence > 0.0
            assert len(card.limitations) > 10
            assert "DO NOT constitute psychiatric diagnoses" in card.clinical_notice


@pytest.mark.asyncio
async def test_dashboard_service_timeline_transcript_masking():
    await init_db()
    async with get_db_session() as session:
        sess_repo = SessionRepository(session)
        await sess_repo.get_or_create(session_id="timeline-mask-sess", user_id="patient-secret-turn")

        turn_repo = TurnRepository(session)
        await turn_repo.record_turn(
            session_id="timeline-mask-sess",
            turn_id=1,
            speaker="user",
            content="Extremely sensitive psychiatric disclosure text."
        )

        service = ClinicianDashboardService(session)

        # Clinician sees unmasked content
        c_timeline = await service.get_patient_timeline(
            user_id="patient-secret-turn",
            actor_role=UserRole.CLINICIAN,
            actor_id="dr_smith"
        )
        conv_c = [e for e in c_timeline.events if e.event_type == "conversation"]
        assert len(conv_c) > 0
        assert "Extremely sensitive" in conv_c[0].description

        # Auditor sees protected text
        a_timeline = await service.get_patient_timeline(
            user_id="patient-secret-turn",
            actor_role=UserRole.AUDITOR,
            actor_id="auditor_1"
        )
        conv_a = [e for e in a_timeline.events if e.event_type == "conversation"]
        assert len(conv_a) > 0
        assert conv_a[0].description == "[CONFIDENTIAL_TRANSCRIPT_PROTECTED]"


@pytest.mark.asyncio
async def test_dashboard_service_access_audit_logs():
    await init_db()
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        # Perform access
        await service.get_overview(actor_role=UserRole.CLINICIAN, actor_id="dr_audited")

        # Query audit logs as supervisor
        logs_res = await service.get_audit_logs(actor_role=UserRole.TRIAGE_SUPERVISOR, actor_id="sup_1")
        assert logs_res.count > 0
        matching = [l for l in logs_res.logs if l.actor_id == "dr_audited" and l.action == "view_overview"]
        assert len(matching) >= 1
