"""User-facing application routes and Single Page Web Interface for ZENOVA."""
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import HTMLResponse, JSONResponse

from zenova.core.logging import get_logger
from zenova.db.session import get_db_session
from zenova.db.repositories import UserRepository, SessionRepository, TurnRepository
from zenova.db.models import UserModel
from zenova.auth.dependencies import get_optional_user, enforce_user_ownership
from zenova.schemas.user import (
    USER_DISCLAIMER_NOTICE,
    PrivacyLevel,
    CommunicationStyle,
    UserPreferences,
    UserPreferencesUpdateRequest,
    WellbeingCheckinRequest,
    WellbeingCheckinResponse,
    CheckinHistoryItem,
    CheckinHistoryResponse,
    SupportResource,
    SupportResourcesResponse,
    SessionSummaryItem,
    UserSessionsResponse,
    UserDataExportResponse,
    DataPurgeResponse,
)

logger = get_logger("zenova.api.routes.user")

router = APIRouter(tags=["User Application"])

VERIFIED_SUPPORT_HOTLINES: List[SupportResource] = [
    SupportResource(
        name="988 Suicide & Crisis Lifeline",
        category="Crisis & Suicide Prevention",
        description="Free, confidential 24/7 support for anyone experiencing mental health crisis or emotional distress.",
        phone="988",
        text_sms="988",
        website="https://988lifeline.org",
        hours="24/7/365",
        availability="Free & Confidential",
        country="United States & Canada"
    ),
    SupportResource(
        name="Crisis Text Line",
        category="Crisis Support via SMS",
        description="Connect with a trained crisis counselor 24/7 via text message.",
        phone=None,
        text_sms="HOME to 741741",
        website="https://www.crisistextline.org",
        hours="24/7/365",
        availability="Free & Confidential",
        country="US, UK, Canada, Ireland"
    ),
    SupportResource(
        name="The Trevor Project",
        category="LGBTQ+ Youth Crisis Support",
        description="Confidential suicide prevention and crisis intervention services for LGBTQ young people.",
        phone="1-866-488-7386",
        text_sms="START to 678-678",
        website="https://www.thetrevorproject.org",
        hours="24/7/365",
        availability="Free & Confidential",
        country="United States"
    ),
    SupportResource(
        name="Veterans Crisis Line",
        category="Veterans & Military Service",
        description="Immediate crisis support for veterans, service members, and their families.",
        phone="988 (Press 1)",
        text_sms="838255",
        website="https://www.veteranscrisisline.net",
        hours="24/7/365",
        availability="Free & Confidential",
        country="United States"
    ),
    SupportResource(
        name="National Maternal Mental Health Hotline",
        category="Maternal Health & Postpartum",
        description="24/7, free, confidential support before, during, and after pregnancy.",
        phone="1-833-852-6262",
        text_sms="1-833-852-6262",
        website="https://mchb.hrsa.gov/national-maternal-mental-health-hotline",
        hours="24/7/365",
        availability="Free & Confidential",
        country="United States"
    ),
    SupportResource(
        name="Befrienders Worldwide & IASP",
        category="International Crisis Directory",
        description="Global directory of free emotional support helplines across over 40 countries.",
        phone=None,
        text_sms=None,
        website="https://www.befrienders.org",
        hours="Varies by country",
        availability="Free & Confidential",
        country="Global / International"
    ),
]

GROUNDING_TECHNIQUES: List[Dict[str, Any]] = [
    {
        "id": "54321_grounding",
        "title": "5-4-3-2-1 Sensory Grounding",
        "description": "An evidence-based mindfulness technique to anchor you in the present moment during moments of high anxiety.",
        "steps": [
            "Acknowledge 5 things you can SEE around you (e.g. a shadow, a pattern on the wall, a pencil).",
            "Acknowledge 4 things you can TOUCH or feel (e.g. feet on the floor, warmth of your hands, texture of your shirt).",
            "Acknowledge 3 things you can HEAR (e.g. traffic outside, clock ticking, refrigerator hum).",
            "Acknowledge 2 things you can SMELL (e.g. fresh air, coffee, paper).",
            "Acknowledge 1 thing you can TASTE or one positive thought about yourself."
        ]
    },
    {
        "id": "box_breathing",
        "title": "Box Breathing (4-4-4-4)",
        "description": "Regulates the autonomic nervous system to reduce acute physiological stress.",
        "steps": [
            "Inhale slowly through your nose for 4 seconds.",
            "Gently hold your breath for 4 seconds.",
            "Exhale smoothly through your mouth for 4 seconds.",
            "Hold empty lungs for 4 seconds before repeating 4 times."
        ]
    }
]


# ==============================================================================
# REST API Endpoints
# ==============================================================================

@router.get("/api/v1/user/resources", response_model=SupportResourcesResponse)
async def get_support_resources() -> SupportResourcesResponse:
    """Retrieve verified crisis hotlines, supportive organizations, and grounding exercises."""
    return SupportResourcesResponse(
        hotlines=VERIFIED_SUPPORT_HOTLINES,
        grounding_techniques=GROUNDING_TECHNIQUES,
        disclaimer=USER_DISCLAIMER_NOTICE
    )


@router.get("/api/v1/user/preferences/{user_id}", response_model=UserPreferences)
async def get_user_preferences(
    user_id: str = Path(..., min_length=1),
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> UserPreferences:
    """Retrieve user settings for data privacy, storage, and communication preferences."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)

    async with get_db_session() as db:
        repo = UserRepository(db)
        model = await repo.get_or_create_preferences(user_id)
        return UserPreferences(
            user_id=model.user_id,
            save_history=model.save_history,
            enable_voice=model.enable_voice,
            enable_wearables=model.enable_wearables,
            privacy_level=PrivacyLevel(model.privacy_level),
            preferred_language=model.preferred_language,
            communication_style=CommunicationStyle(model.communication_style),
            allow_clinician_sharing=model.allow_clinician_sharing,
            updated_at=model.updated_at.isoformat() if model.updated_at else datetime.now(timezone.utc).isoformat()
        )


@router.put("/api/v1/user/preferences/{user_id}", response_model=UserPreferences)
async def update_user_preferences(
    user_id: str,
    updates: UserPreferencesUpdateRequest,
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> UserPreferences:
    """Update user privacy and data retention preferences."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)

    async with get_db_session() as db:
        repo = UserRepository(db)
        update_dict = updates.model_dump(exclude_unset=True)
        # Convert enums to string values for database storage
        if "privacy_level" in update_dict and update_dict["privacy_level"] is not None:
            update_dict["privacy_level"] = update_dict["privacy_level"].value
        if "communication_style" in update_dict and update_dict["communication_style"] is not None:
            update_dict["communication_style"] = update_dict["communication_style"].value

        model = await repo.update_preferences(user_id, update_dict)
        return UserPreferences(
            user_id=model.user_id,
            save_history=model.save_history,
            enable_voice=model.enable_voice,
            enable_wearables=model.enable_wearables,
            privacy_level=PrivacyLevel(model.privacy_level),
            preferred_language=model.preferred_language,
            communication_style=CommunicationStyle(model.communication_style),
            allow_clinician_sharing=model.allow_clinician_sharing,
            updated_at=model.updated_at.isoformat() if model.updated_at else datetime.now(timezone.utc).isoformat()
        )


@router.post("/api/v1/user/checkin", response_model=WellbeingCheckinResponse)
async def submit_wellbeing_checkin(
    checkin: WellbeingCheckinRequest,
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> WellbeingCheckinResponse:
    """Record a daily or periodic wellbeing check-in to track longitudinal state."""
    if auth_user:
        enforce_user_ownership(checkin.user_id, auth_user)

    checkin_id = f"chk_{uuid.uuid4().hex[:12]}"
    async with get_db_session() as db:
        repo = UserRepository(db)
        model = await repo.record_checkin(
            checkin_id=checkin_id,
            user_id=checkin.user_id,
            mood_score=checkin.mood_score,
            valence=checkin.valence,
            sleep_hours=checkin.sleep_hours,
            stress_level=checkin.stress_level,
            energy_level=checkin.energy_level,
            notes=checkin.notes
        )

        # Supportive non-diagnostic reflection message based on mood score
        if checkin.mood_score >= 8:
            msg = "Thank you for checking in. It sounds like you are feeling grounded and vital today."
        elif checkin.mood_score >= 5:
            msg = "Thank you for taking a moment for yourself today. Checking in regularly helps build awareness."
        else:
            msg = "Thank you for sharing. It takes courage to acknowledge when things feel difficult. Remember to be gentle with yourself today."

        return WellbeingCheckinResponse(
            checkin_id=model.checkin_id,
            user_id=model.user_id,
            mood_score=model.mood_score,
            valence=model.valence,
            sleep_hours=model.sleep_hours,
            stress_level=model.stress_level,
            energy_level=model.energy_level,
            notes=model.notes,
            created_at=model.created_at.isoformat() if model.created_at else datetime.now(timezone.utc).isoformat(),
            feedback_message=msg
        )


@router.get("/api/v1/user/checkins/{user_id}", response_model=CheckinHistoryResponse)
async def get_checkin_history(
    user_id: str,
    limit: int = Query(default=30, ge=1, le=100),
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> CheckinHistoryResponse:
    """Retrieve historical check-in timeline and longitudinal averages."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)

    async with get_db_session() as db:
        repo = UserRepository(db)
        records = await repo.get_checkin_history(user_id, limit=limit)
        if not records:
            return CheckinHistoryResponse(
                user_id=user_id,
                total_checkins=0,
                average_mood=0.0,
                average_sleep_hours=0.0,
                average_stress=0.0,
                checkins=[]
            )

        avg_mood = sum(r.mood_score for r in records) / len(records)
        avg_sleep = sum(r.sleep_hours for r in records) / len(records)
        avg_stress = sum(r.stress_level for r in records) / len(records)

        items = [
            CheckinHistoryItem(
                checkin_id=r.checkin_id,
                mood_score=r.mood_score,
                valence=r.valence,
                sleep_hours=r.sleep_hours,
                stress_level=r.stress_level,
                energy_level=r.energy_level,
                notes=r.notes,
                created_at=r.created_at.isoformat() if r.created_at else ""
            )
            for r in records
        ]

        return CheckinHistoryResponse(
            user_id=user_id,
            total_checkins=len(records),
            average_mood=round(avg_mood, 2),
            average_sleep_hours=round(avg_sleep, 2),
            average_stress=round(avg_stress, 2),
            checkins=items
        )


@router.get("/api/v1/user/sessions/{user_id}", response_model=UserSessionsResponse)
async def list_user_sessions(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> UserSessionsResponse:
    """Retrieve user's past conversational session threads."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)

    async with get_db_session() as db:
        repo = UserRepository(db)
        sessions = await repo.list_user_sessions(user_id, limit=limit)
        items = [
            SessionSummaryItem(
                session_id=s["session_id"],
                user_id=s["user_id"],
                turn_count=s["turn_count"],
                first_message_preview=s["first_message_preview"],
                created_at=s["created_at"],
                updated_at=s["updated_at"]
            )
            for s in sessions
        ]
        return UserSessionsResponse(
            user_id=user_id,
            total_sessions=len(items),
            sessions=items
        )


@router.post("/api/v1/user/export/{user_id}", response_model=UserDataExportResponse)
async def export_user_data(
    user_id: str,
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> UserDataExportResponse:
    """Export all stored conversational turns, check-ins, and preferences in JSON format."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)

    async with get_db_session() as db:
        repo = UserRepository(db)
        data = await repo.export_user_data(user_id)
        return UserDataExportResponse(
            export_id=data["export_id"],
            user_id=data["user_id"],
            exported_at=data["exported_at"],
            preferences=data["preferences"],
            checkins=data["checkins"],
            sessions_count=data["sessions_count"],
            sessions=data["sessions"],
            observations_count=data["observations_count"],
            disclaimer=USER_DISCLAIMER_NOTICE
        )


@router.delete("/api/v1/user/data/{user_id}", response_model=DataPurgeResponse)
async def purge_user_data(
    user_id: str,
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> DataPurgeResponse:
    """Permanently delete all personal user records (GDPR Right to be Forgotten)."""
    if auth_user:
        enforce_user_ownership(user_id, auth_user)
    async with get_db_session() as db:
        repo = UserRepository(db)
        counts = await repo.purge_user_data(user_id)
        logger.info(f"User {user_id} purged personal data: {counts}")
        return DataPurgeResponse(
            user_id=user_id,
            purged_at=datetime.now(timezone.utc).isoformat(),
            deleted_sessions=counts["deleted_sessions"],
            deleted_turns=counts["deleted_turns"],
            deleted_checkins=counts["deleted_checkins"],
            deleted_observations=counts["deleted_observations"]
        )


# ==============================================================================
# User-Facing Responsive Web Application (GET /app)
# ==============================================================================

@router.get("/app", response_class=HTMLResponse, summary="ZENOVA User-Facing Web Application")
async def get_user_application() -> HTMLResponse:
    """Serve the complete, accessible, responsive single-page Web Application for users."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA — Supportive AI Companion</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%230284c7'><circle cx='12' cy='12' r='10'/></svg>">
    <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-surface: #ffffff;
            --bg-sidebar: #f1f5f9;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #0284c7;
            --primary-hover: #0369a1;
            --primary-light: #e0f2fe;
            --accent: #0d9488;
            --border: #e2e8f0;
            --danger: #ef4444;
            --danger-light: #fee2e2;
            --warning: #f59e0b;
            --warning-light: #fef3c7;
            --success: #10b981;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        [data-theme="dark"] {
            --bg-primary: #0b0f19;
            --bg-surface: #131b2e;
            --bg-sidebar: #0f172a;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --primary-hover: #0ea5e9;
            --primary-light: #1e293b;
            --accent: #2dd4bf;
            --border: #1e293b;
            --danger-light: #450a0a;
            --warning-light: #451a03;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Disclaimer Banner */
        .disclaimer-banner {
            background-color: var(--warning-light);
            border-bottom: 1px solid var(--warning);
            color: #92400e;
            padding: 8px 16px;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        [data-theme="dark"] .disclaimer-banner {
            color: #fde68a;
        }
        .disclaimer-banner strong {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .crisis-callout-btn {
            background-color: var(--danger);
            color: #ffffff;
            border: none;
            padding: 4px 12px;
            border-radius: var(--radius-sm);
            font-weight: 600;
            font-size: 12px;
            cursor: pointer;
            text-decoration: none;
            white-space: nowrap;
        }
        .crisis-callout-btn:hover {
            opacity: 0.9;
        }

        /* App Header */
        header.app-header {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 40;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-logo {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 18px;
        }
        .brand-text h1 {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .brand-text p {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Navigation Pills */
        nav.nav-tabs {
            display: flex;
            align-items: center;
            gap: 6px;
            background-color: var(--bg-sidebar);
            padding: 4px;
            border-radius: var(--radius-md);
        }
        .nav-btn {
            background: none;
            border: none;
            padding: 6px 14px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .nav-btn.active {
            background-color: var(--bg-surface);
            color: var(--primary);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .nav-btn:hover:not(.active) {
            color: var(--text-main);
        }

        /* Controls / User profile pill */
        .header-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .theme-toggle-btn, .user-id-pill {
            background-color: var(--bg-sidebar);
            border: 1px solid var(--border);
            padding: 6px 10px;
            border-radius: var(--radius-sm);
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
        }

        /* Main Workspace */
        main.app-container {
            flex: 1;
            display: flex;
            overflow: hidden;
            position: relative;
        }

        /* Tab Views */
        .tab-view {
            display: none;
            width: 100%;
            height: calc(100vh - 110px);
        }
        .tab-view.active {
            display: flex;
        }

        /* ====================================================================
           1. CONVERSATION VIEW
           ==================================================================== */
        .sidebar {
            width: 280px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 16px;
            gap: 14px;
        }
        .new-chat-btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: background-color 0.2s;
        }
        .new-chat-btn:hover {
            background-color: var(--primary-hover);
        }
        .sessions-list-header {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }
        .sessions-list {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .session-card {
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .session-card:hover {
            border-color: var(--primary);
        }
        .session-card.active {
            border-color: var(--primary);
            background-color: var(--primary-light);
        }
        .session-card-title {
            font-size: 13px;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .session-card-meta {
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }

        /* Chat Panel */
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background-color: var(--bg-surface);
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message-bubble {
            max-width: 75%;
            padding: 14px 18px;
            border-radius: var(--radius-lg);
            font-size: 14px;
            line-height: 1.5;
            position: relative;
        }
        .message-user {
            align-self: flex-end;
            background-color: var(--primary);
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message-assistant {
            align-self: flex-start;
            background-color: var(--bg-sidebar);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-bottom-left-radius: 4px;
        }
        .message-time {
            font-size: 10px;
            margin-top: 6px;
            opacity: 0.7;
            text-align: right;
        }

        /* Crisis Escalation Card (Triggered by backend safety workflow) */
        .crisis-intervention-card {
            align-self: center;
            width: 90%;
            max-width: 600px;
            background-color: var(--danger-light);
            border: 2px solid var(--danger);
            border-radius: var(--radius-lg);
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            animation: fadeIn 0.3s ease;
        }
        .crisis-intervention-card h3 {
            color: var(--danger);
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
        }
        .crisis-intervention-card p {
            font-size: 13px;
            color: var(--text-main);
            line-height: 1.4;
        }
        .crisis-action-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .crisis-action-btn {
            background-color: var(--danger);
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: var(--radius-sm);
            font-weight: 700;
            font-size: 13px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .crisis-action-btn.secondary {
            background-color: transparent;
            border: 1px solid var(--danger);
            color: var(--danger);
        }

        /* Input Controls */
        .chat-input-bar {
            padding: 16px 24px;
            border-top: 1px solid var(--border);
            background-color: var(--bg-surface);
            display: flex;
            align-items: flex-end;
            gap: 10px;
        }
        .chat-textarea {
            flex: 1;
            background-color: var(--bg-sidebar);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            font-size: 14px;
            color: var(--text-main);
            resize: none;
            height: 48px;
            max-height: 140px;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-textarea:focus {
            border-color: var(--primary);
        }
        .icon-btn {
            width: 44px;
            height: 44px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            background-color: var(--bg-sidebar);
            color: var(--text-main);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }
        .icon-btn:hover {
            border-color: var(--primary);
            color: var(--primary);
        }
        .icon-btn.recording {
            background-color: var(--danger);
            color: white;
            animation: pulse 1.5s infinite;
            border-color: var(--danger);
        }
        .send-btn {
            background-color: var(--primary);
            color: white;
            border: none;
        }
        .send-btn:hover {
            background-color: var(--primary-hover);
            color: white;
        }

        /* Typing indicator */
        .typing-indicator {
            display: none;
            align-self: flex-start;
            padding: 8px 16px;
            background-color: var(--bg-sidebar);
            border-radius: var(--radius-md);
            font-size: 12px;
            color: var(--text-muted);
            font-style: italic;
        }

        /* ====================================================================
           2. WELLBEING CHECK-IN VIEW
           ==================================================================== */
        .checkin-container {
            max-width: 800px;
            margin: 24px auto;
            padding: 0 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
            width: 100%;
        }
        .card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .card-header h2 {
            font-size: 18px;
            font-weight: 700;
        }
        .card-header p {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .form-group label {
            font-size: 13px;
            font-weight: 600;
        }
        .slider-wrapper {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .slider-wrapper input[type="range"] {
            flex: 1;
            accent-color: var(--primary);
        }
        .slider-val {
            min-width: 32px;
            font-weight: 700;
            color: var(--primary);
            font-size: 16px;
            text-align: right;
        }
        .pill-selector {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .pill-choice {
            padding: 8px 16px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            background-color: var(--bg-sidebar);
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .pill-choice.selected {
            background-color: var(--primary-light);
            border-color: var(--primary);
            color: var(--primary);
            font-weight: 600;
        }
        .submit-btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: var(--radius-md);
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
            align-self: flex-start;
        }
        .submit-btn:hover {
            background-color: var(--primary-hover);
        }
        .metrics-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
        }
        .metric-card {
            background-color: var(--bg-sidebar);
            padding: 14px;
            border-radius: var(--radius-md);
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .metric-card span {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        .metric-card strong {
            font-size: 20px;
            color: var(--primary);
        }

        /* ====================================================================
           3. SUPPORT RESOURCES VIEW
           ==================================================================== */
        .resources-container {
            max-width: 900px;
            margin: 24px auto;
            padding: 0 20px;
            overflow-y: auto;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        .resource-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }
        .resource-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 18px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .resource-card h3 {
            font-size: 16px;
            color: var(--primary);
        }
        .resource-badge {
            font-size: 11px;
            background-color: var(--primary-light);
            color: var(--primary);
            padding: 2px 8px;
            border-radius: var(--radius-sm);
            width: fit-content;
            font-weight: 600;
        }
        .resource-card p {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.4;
        }
        .contact-box {
            background-color: var(--bg-sidebar);
            padding: 10px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        /* Grounding exercises */
        .exercise-step {
            padding: 10px 14px;
            background-color: var(--bg-sidebar);
            border-left: 3px solid var(--accent);
            border-radius: var(--radius-sm);
            font-size: 13px;
            margin-bottom: 8px;
        }

        /* ====================================================================
           4. PRIVACY & DATA GOVERNANCE VIEW
           ==================================================================== */
        .settings-container {
            max-width: 700px;
            margin: 24px auto;
            padding: 0 20px;
            overflow-y: auto;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .setting-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        .setting-desc h4 {
            font-size: 14px;
            margin-bottom: 4px;
        }
        .setting-desc p {
            font-size: 12px;
            color: var(--text-muted);
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: var(--border);
            transition: .3s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: var(--primary);
        }
        input:checked + .slider:before {
            transform: translateX(20px);
        }
        .danger-zone {
            border: 1px solid var(--danger);
            background-color: var(--danger-light);
            border-radius: var(--radius-lg);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .danger-zone h3 {
            color: var(--danger);
            font-size: 16px;
        }
        .danger-btn {
            background-color: var(--danger);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: var(--radius-sm);
            font-weight: 700;
            font-size: 13px;
            cursor: pointer;
            align-self: flex-start;
        }

        /* Roadmap Placeholders */
        .roadmap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 8px;
        }
        .roadmap-card {
            background-color: var(--bg-sidebar);
            border: 1px dashed var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .roadmap-card span {
            font-size: 10px;
            color: var(--accent);
            text-transform: uppercase;
            font-weight: 700;
        }
        .roadmap-card h4 {
            font-size: 14px;
        }
        .roadmap-card p {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* First-Time Onboarding Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal-card {
            background: var(--bg-surface);
            border-radius: var(--radius-lg);
            max-width: 520px;
            width: 90%;
            padding: 28px;
            box-shadow: var(--shadow-lg);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .modal-card h2 {
            font-size: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .modal-card p {
            font-size: 13px;
            line-height: 1.5;
            color: var(--text-muted);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        /* Responsive Mobile layout */
        @media (max-width: 768px) {
            .sidebar {
                display: none; /* Collapsed on mobile; could toggle */
            }
            .nav-tabs span {
                display: none; /* Hide text, keep icons */
            }
            .brand-text p {
                display: none;
            }
        }
    </style>
</head>
<body>

    <!-- Top Non-Medical Disclaimer Banner -->
    <aside class="disclaimer-banner" role="complementary" aria-label="Medical Disclaimer">
        <span>
            <strong>⚠️ Non-Medical AI Companion:</strong>
            ZENOVA provides supportive emotional dialogue and does not give medical diagnoses or treatments.
        </span>
        <a href="tel:988" class="crisis-callout-btn" aria-label="Call 988 Crisis Lifeline">Emergency? Call 988</a>
    </aside>

    <!-- App Header -->
    <header class="app-header">
        <div class="brand">
            <div class="brand-logo" aria-hidden="true">Z</div>
            <div class="brand-text">
                <h1>ZENOVA</h1>
                <p>Supportive Wellbeing Companion</p>
            </div>
        </div>

        <nav class="nav-tabs" role="tablist" aria-label="Main Navigation">
            <button class="nav-btn active" role="tab" aria-selected="true" data-tab="conversation">
                💬 <span>Conversation</span>
            </button>
            <button class="nav-btn" role="tab" aria-selected="false" data-tab="checkin">
                🌱 <span>Wellbeing Check-in</span>
            </button>
            <button class="nav-btn" role="tab" aria-selected="false" data-tab="resources">
                🛡️ <span>Support Resources</span>
            </button>
            <button class="nav-btn" role="tab" aria-selected="false" data-tab="settings">
                ⚙️ <span>Privacy & Data</span>
            </button>
        </nav>

        <div class="header-controls">
            <button id="theme-toggle" class="theme-toggle-btn" aria-label="Toggle dark mode">🌙</button>
            <div id="user-pill" class="user-id-pill" title="Current User ID">user_guest</div>
            <button id="logout-btn" class="theme-toggle-btn" onclick="handleLogout()" title="Sign out of ZENOVA" style="color: var(--danger); font-weight: 600;">Sign Out</button>
        </div>
    </header>

    <!-- Main Workspace Area -->
    <main class="app-container">

        <!-- ==============================================================
             1. CONVERSATION VIEW
             ============================================================== -->
        <section id="tab-conversation" class="tab-view active" role="tabpanel">
            <!-- Sidebar: Sessions History -->
            <aside class="sidebar">
                <button id="new-session-btn" class="new-chat-btn" aria-label="Start new conversation">
                    ➕ New Conversation
                </button>
                <div class="sessions-list-header">Previous Sessions</div>
                <div id="sessions-list-container" class="sessions-list">
                    <!-- Session cards dynamically populated -->
                </div>
            </aside>

            <!-- Chat Panel -->
            <section class="chat-panel">
                <div id="chat-messages" class="chat-messages" role="log" aria-live="polite">
                    <!-- Initial Welcome Bubble -->
                    <div class="message-bubble message-assistant">
                        Hello. I'm ZENOVA, your supportive AI companion for emotional reflection and wellbeing.
                        I'm here to listen, validate what you're experiencing, and explore gentle coping strategies with you.
                        How are you feeling right now?
                        <div class="message-time">Just now</div>
                    </div>
                </div>

                <div id="typing-indicator" class="typing-indicator" aria-hidden="true">
                    ZENOVA is listening and reflecting...
                </div>

                <!-- Chat Input Area -->
                <footer class="chat-input-bar">
                    <button id="mic-btn" class="icon-btn" aria-label="Toggle voice input" title="Record Voice Audio">
                        🎤
                    </button>
                    <textarea id="chat-input" class="chat-textarea" placeholder="Share what is on your mind... (Press Enter to send)" rows="1" aria-label="Message input"></textarea>
                    <button id="send-btn" class="icon-btn send-btn" aria-label="Send message" title="Send">
                        ➤
                    </button>
                </footer>
            </section>
        </section>

        <!-- ==============================================================
             2. WELLBEING CHECK-IN VIEW
             ============================================================== -->
        <section id="tab-checkin" class="tab-view" role="tabpanel">
            <div class="checkin-container">
                <div class="card">
                    <div class="card-header">
                        <h2>Daily Wellbeing Check-in</h2>
                        <p>Track your emotional valence, sleep, and energy levels over time to build self-awareness.</p>
                    </div>

                    <div class="form-group">
                        <label>How would you describe your overall mood today? (1: Very Low — 10: Flourishing)</label>
                        <div class="slider-wrapper">
                            <input type="range" id="mood-slider" min="1" max="10" value="6">
                            <span id="mood-val" class="slider-val">6</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Hours of Sleep (past 24h)</label>
                        <div class="slider-wrapper">
                            <input type="range" id="sleep-slider" min="0" max="14" step="0.5" value="7.5">
                            <span id="sleep-val" class="slider-val">7.5h</span>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Subjective Stress Level</label>
                        <div id="stress-pills" class="pill-selector">
                            <div class="pill-choice" data-val="1">1: Very Calm</div>
                            <div class="pill-choice" data-val="2">2: Mild</div>
                            <div class="pill-choice selected" data-val="3">3: Moderate</div>
                            <div class="pill-choice" data-val="4">4: Elevated</div>
                            <div class="pill-choice" data-val="5">5: Overwhelming</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Subjective Energy Level</label>
                        <div id="energy-pills" class="pill-selector">
                            <div class="pill-choice" data-val="1">1: Exhausted</div>
                            <div class="pill-choice" data-val="2">2: Low</div>
                            <div class="pill-choice selected" data-val="3">3: Moderate</div>
                            <div class="pill-choice" data-val="4">4: Energetic</div>
                            <div class="pill-choice" data-val="5">5: Vital</div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Reflective Note (Optional)</label>
                        <textarea id="checkin-note" class="chat-textarea" placeholder="Any specific thought, context, or gratitude you would like to log?" rows="2"></textarea>
                    </div>

                    <button id="submit-checkin-btn" class="submit-btn">Record Check-in</button>
                    <div id="checkin-feedback" style="display:none; font-size:13px; color:var(--success); font-weight:600;"></div>
                </div>

                <!-- Longitudinal Summary -->
                <div class="card">
                    <div class="card-header">
                        <h2>Your Wellbeing Trends</h2>
                        <p>Averages across your recent check-in entries.</p>
                    </div>
                    <div class="metrics-summary-grid">
                        <div class="metric-card">
                            <span>Total Check-ins</span>
                            <strong id="stat-total">0</strong>
                        </div>
                        <div class="metric-card">
                            <span>Average Mood</span>
                            <strong id="stat-avg-mood">—</strong>
                        </div>
                        <div class="metric-card">
                            <span>Average Sleep</span>
                            <strong id="stat-avg-sleep">—</strong>
                        </div>
                        <div class="metric-card">
                            <span>Average Stress</span>
                            <strong id="stat-avg-stress">—</strong>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==============================================================
             3. SUPPORT RESOURCES VIEW
             ============================================================== -->
        <section id="tab-resources" class="tab-view" role="tabpanel">
            <div class="resources-container">
                <div class="card" style="border-left: 4px solid var(--danger);">
                    <h2>Immediate Crisis Support (Available 24/7/365)</h2>
                    <p>If you or someone you know is struggling or in crisis, help is available right now. These services are free, confidential, and staffed by trained professionals.</p>
                </div>

                <div id="resources-grid" class="resource-grid">
                    <!-- Populated dynamically from /api/v1/user/resources -->
                </div>

                <div class="card">
                    <h2>Self-Care & Grounding Exercises</h2>
                    <p>Interactive tools you can practice whenever you feel overwhelmed.</p>
                    <div id="grounding-container">
                        <!-- Populated dynamically -->
                    </div>
                </div>
            </div>
        </section>

        <!-- ==============================================================
             4. PRIVACY & DATA GOVERNANCE VIEW
             ============================================================== -->
        <section id="tab-settings" class="tab-view" role="tabpanel">
            <div class="settings-container">
                <div class="card">
                    <div class="card-header">
                        <h2>Privacy & Data Ownership</h2>
                        <p>You have full ownership of your data in ZENOVA. Toggle features on or off at any time.</p>
                    </div>

                    <div class="setting-row">
                        <div class="setting-desc">
                            <h4>Save Conversation History</h4>
                            <p>Store your chat history so you can resume previous conversations.</p>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-save-history" checked>
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div class="setting-row">
                        <div class="setting-desc">
                            <h4>Enable Voice Input Feature Extraction</h4>
                            <p>Allow acoustic speech feature analysis when using the microphone.</p>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-enable-voice" checked>
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div class="setting-row">
                        <div class="setting-desc">
                            <h4>Wearable & Behavioral Integration</h4>
                            <p>Allow passive sleep and heart-rate telemetry integration.</p>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-enable-wearables">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div class="setting-row">
                        <div class="setting-desc">
                            <h4>Data Export (GDPR / HIPAA Portability)</h4>
                            <p>Download a copy of all your stored sessions, check-ins, and settings.</p>
                        </div>
                        <button id="export-data-btn" class="submit-btn" style="padding: 8px 16px;">Export JSON</button>
                    </div>
                </div>

                <!-- Extensibility Roadmap Placeholders -->
                <div class="card">
                    <div class="card-header">
                        <h2>Future Roadmap & Integrations</h2>
                        <p>ZENOVA is designed to grow with extensible integrations.</p>
                    </div>
                    <div class="roadmap-grid">
                        <div class="roadmap-card">
                            <span>Mobile App</span>
                            <h4>Progressive Web App (PWA)</h4>
                            <p>Installable on iOS & Android home screens with offline caching capabilities.</p>
                        </div>
                        <div class="roadmap-card">
                            <span>Wearables</span>
                            <h4>Apple Health & Fitbit</h4>
                            <p>Passive sleep quality, HRV, and circadian rhythm alignment.</p>
                        </div>
                        <div class="roadmap-card">
                            <span>Multilingual</span>
                            <h4>Global Language Support</h4>
                            <p>Real-time localization in Spanish, French, German, and Mandarin.</p>
                        </div>
                        <div class="roadmap-card">
                            <span>Clinician Link</span>
                            <h4>Care Team Coordination</h4>
                            <p>Securely share longitudinal baseline summaries with your licensed therapist.</p>
                        </div>
                    </div>
                </div>

                <!-- Danger Zone: Right to be Forgotten -->
                <div class="danger-zone">
                    <h3>Right to be Forgotten (Purge All Data)</h3>
                    <p>Permanently delete all your conversation history, check-in logs, and observations. This action is irreversible.</p>
                    <button id="purge-data-btn" class="danger-btn">Delete All My Data</button>
                </div>
            </div>
        </section>

    </main>

    <!-- Onboarding Disclaimer Modal -->
    <div id="onboarding-modal" class="modal-overlay">
        <div class="modal-card">
            <h2>🛡️ Welcome to ZENOVA</h2>
            <p><strong>Please read before continuing:</strong></p>
            <p>ZENOVA is an AI companion created for supportive dialogue, emotional validation, and personal wellbeing tracking.</p>
            <p><strong>What ZENOVA is NOT:</strong></p>
            <ul style="font-size: 13px; color: var(--text-muted); margin-left: 20px; line-height: 1.6;">
                <li>ZENOVA is NOT a medical doctor, psychiatrist, or licensed therapist.</li>
                <li>ZENOVA does NOT diagnose psychiatric or physical medical conditions.</li>
                <li>ZENOVA cannot prescribe or manage medical prescriptions.</li>
                <li>ZENOVA cannot handle emergency life-threatening crises.</li>
            </ul>
            <p>If you are in distress or need crisis care, the 988 Lifeline is free, confidential, and available 24/7.</p>
            <button id="acknowledge-btn" class="submit-btn" style="width: 100%; text-align: center;">I Understand & Agree</button>
        </div>
    </div>

    <script>
        // ====================================================================
        // ZENOVA Client Application Logic (Zero Frontend ML)
        // ====================================================================
        let userId = "user_" + (localStorage.getItem("zenova_uid") || Math.random().toString(36).substring(2, 8));
        localStorage.setItem("zenova_uid", userId.replace("user_", ""));
        document.getElementById("user-pill").textContent = userId;

        function authFetch(url, options = {}) {
            const token = localStorage.getItem("zenova_token");
            const headers = options.headers ? { ...options.headers } : {};
            if (token && !headers["Authorization"]) {
                headers["Authorization"] = `Bearer ${token}`;
            }
            return fetch(url, {
                ...options,
                credentials: "include",
                headers: headers
            });
        }

        let activeSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
        let selectedStress = 3;
        let selectedEnergy = 3;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        // Onboarding Disclaimer Acknowledgement
        if (!localStorage.getItem("zenova_disclaimer_acknowledged")) {
            document.getElementById("onboarding-modal").style.display = "flex";
        }
        document.getElementById("acknowledge-btn").addEventListener("click", () => {
            localStorage.setItem("zenova_disclaimer_acknowledged", "true");
            document.getElementById("onboarding-modal").style.display = "none";
        });

        // Theme Toggle
        const themeToggle = document.getElementById("theme-toggle");
        themeToggle.addEventListener("click", () => {
            const current = document.documentElement.getAttribute("data-theme");
            const next = current === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            themeToggle.textContent = next === "dark" ? "☀️" : "🌙";
        });

        // Navigation Tabs
        const navButtons = document.querySelectorAll(".nav-btn");
        const tabViews = document.querySelectorAll(".tab-view");
        navButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                navButtons.forEach(b => {
                    b.classList.remove("active");
                    b.setAttribute("aria-selected", "false");
                });
                tabViews.forEach(t => t.classList.remove("active"));
                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");
                const targetTab = btn.getAttribute("data-tab");
                document.getElementById("tab-" + targetTab).classList.add("active");
                if (targetTab === "checkin") loadCheckinStats();
                if (targetTab === "resources") loadResources();
                if (targetTab === "settings") loadPreferences();
            });
        });

        // ====================================================================
        // Conversation Handlers
        // ====================================================================
        const chatInput = document.getElementById("chat-input");
        const sendBtn = document.getElementById("send-btn");
        const chatMessages = document.getElementById("chat-messages");
        const typingIndicator = document.getElementById("typing-indicator");
        const micBtn = document.getElementById("mic-btn");

        async function sendMessage(audioBase64 = null) {
            const text = chatInput.value.trim();
            if (!text && !audioBase64) return;

            // Render user bubble
            if (text) {
                appendBubble(text, "user");
                chatInput.value = "";
            } else if (audioBase64) {
                appendBubble("🎤 [Voice Message Sent]", "user");
            }

            typingIndicator.style.display = "block";
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const payload = {
                    session_id: activeSessionId,
                    user_id: userId,
                    text: text || "...",
                    audio_base64: audioBase64,
                    metadata: { client: "zenova_web_app" }
                };

                const res = await authFetch("/api/v1/orchestrator/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                typingIndicator.style.display = "none";

                if (!res.ok) {
                    appendBubble("I apologize, but I encountered a momentary connection issue. Please try again.", "assistant");
                    return;
                }

                const data = await res.json();
                appendBubble(data.response, "assistant");

                // Check for safety escalation workflow
                if (data.escalated_to_human || (data.safety && data.safety.action === "block_and_escalate")) {
                    renderCrisisBanner();
                }

                loadSessionsList();
            } catch (err) {
                typingIndicator.style.display = "none";
                appendBubble("Network connection error. Please ensure the server is accessible.", "assistant");
            }
        }

        function appendBubble(content, role) {
            const bubble = document.createElement("div");
            bubble.className = "message-bubble message-" + role;
            bubble.textContent = content;
            const time = document.createElement("div");
            time.className = "message-time";
            time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
            bubble.appendChild(time);
            chatMessages.appendChild(bubble);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function renderCrisisBanner() {
            const card = document.createElement("div");
            card.className = "crisis-intervention-card";
            card.innerHTML = `
                <h3>⚠️ Immediate Support Resources</h3>
                <p>It sounds like you may be carrying an overwhelming amount of pain or thoughts of harm right now. Your safety and life matter deeply. Please reach out to someone who can help keep you safe:</p>
                <div class="crisis-action-row">
                    <a href="tel:988" class="crisis-action-btn">📞 Call 988 Lifeline</a>
                    <a href="sms:988" class="crisis-action-btn">💬 Text 988</a>
                    <a href="sms:741741?body=HOME" class="crisis-action-btn secondary">Text HOME to 741741</a>
                </div>
            `;
            chatMessages.appendChild(card);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        sendBtn.addEventListener("click", () => sendMessage());
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Voice Input (MediaRecorder)
        micBtn.addEventListener("click", async () => {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        const blob = new Blob(audioChunks, { type: "audio/webm" });
                        const reader = new FileReader();
                        reader.readAsDataURL(blob);
                        reader.onloadend = () => {
                            const base64Audio = reader.result.split(",")[1];
                            sendMessage(base64Audio);
                        };
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.classList.add("recording");
                } catch (err) {
                    alert("Microphone access unavailable or denied. You can continue typing in the text box.");
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.classList.remove("recording");
            }
        });

        // New Session
        document.getElementById("new-session-btn").addEventListener("click", () => {
            activeSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
            chatMessages.innerHTML = `
                <div class="message-bubble message-assistant">
                    Started a fresh session. How are you feeling right now?
                    <div class="message-time">Just now</div>
                </div>
            `;
            loadSessionsList();
        });

        // Load Sessions List
        async function loadSessionsList() {
            try {
                const res = await authFetch("/api/v1/user/sessions/" + userId);
                if (!res.ok) return;
                const data = await res.json();
                const container = document.getElementById("sessions-list-container");
                container.innerHTML = "";
                data.sessions.forEach(s => {
                    const card = document.createElement("div");
                    card.className = "session-card" + (s.session_id === activeSessionId ? " active" : "");
                    card.innerHTML = `
                        <div class="session-card-title">${s.first_message_preview || 'Conversation'}</div>
                        <div class="session-card-meta">
                            <span>${s.turn_count} turns</span>
                            <span>${s.created_at ? new Date(s.created_at).toLocaleDateString() : ''}</span>
                        </div>
                    `;
                    card.addEventListener("click", () => switchSession(s.session_id));
                    container.appendChild(card);
                });
            } catch (err) {
                console.error("Error loading sessions:", err);
            }
        }

        async function switchSession(sid) {
            activeSessionId = sid;
            loadSessionsList();
            try {
                const res = await authFetch("/api/v1/conversation/" + sid + "/history");
                if (!res.ok) return;
                const data = await res.json();
                chatMessages.innerHTML = "";
                data.turns.forEach(t => appendBubble(t.content, t.speaker));
            } catch (err) {
                console.error("Error loading history:", err);
            }
        }

        // ====================================================================
        // Wellbeing Check-in Logic
        // ====================================================================
        const moodSlider = document.getElementById("mood-slider");
        const moodVal = document.getElementById("mood-val");
        moodSlider.addEventListener("input", () => moodVal.textContent = moodSlider.value);

        const sleepSlider = document.getElementById("sleep-slider");
        const sleepVal = document.getElementById("sleep-val");
        sleepSlider.addEventListener("input", () => sleepVal.textContent = sleepSlider.value + "h");

        function setupPills(containerId, onSelect) {
            const pills = document.querySelectorAll("#" + containerId + " .pill-choice");
            pills.forEach(p => {
                p.addEventListener("click", () => {
                    pills.forEach(x => x.classList.remove("selected"));
                    p.classList.add("selected");
                    onSelect(parseInt(p.getAttribute("data-val")));
                });
            });
        }
        setupPills("stress-pills", val => selectedStress = val);
        setupPills("energy-pills", val => selectedEnergy = val);

        document.getElementById("submit-checkin-btn").addEventListener("click", async () => {
            const mood = parseInt(moodSlider.value);
            const sleep = parseFloat(sleepSlider.value);
            const notes = document.getElementById("checkin-note").value.trim();
            const valence = (mood - 5.5) / 4.5; // map 1-10 to approx -1.0 to 1.0

            try {
                const res = await authFetch("/api/v1/user/checkin", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: userId,
                        mood_score: mood,
                        valence: valence,
                        sleep_hours: sleep,
                        stress_level: selectedStress,
                        energy_level: selectedEnergy,
                        notes: notes || null
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    const fb = document.getElementById("checkin-feedback");
                    fb.textContent = "✓ Check-in saved! " + data.feedback_message;
                    fb.style.display = "block";
                    loadCheckinStats();
                }
            } catch (err) {
                alert("Could not save check-in. Please try again.");
            }
        });

        async function loadCheckinStats() {
            try {
                const res = await authFetch("/api/v1/user/checkins/" + userId);
                if (!res.ok) return;
                const data = await res.json();
                document.getElementById("stat-total").textContent = data.total_checkins;
                document.getElementById("stat-avg-mood").textContent = data.total_checkins ? data.average_mood + "/10" : "—";
                document.getElementById("stat-avg-sleep").textContent = data.total_checkins ? data.average_sleep_hours + "h" : "—";
                document.getElementById("stat-avg-stress").textContent = data.total_checkins ? data.average_stress + "/5" : "—";
            } catch (err) {
                console.error("Check-in stats error:", err);
            }
        }

        // ====================================================================
        // Support Resources Logic
        // ====================================================================
        async function loadResources() {
            try {
                const res = await authFetch("/api/v1/user/resources");
                if (!res.ok) return;
                const data = await res.json();
                const grid = document.getElementById("resources-grid");
                grid.innerHTML = "";
                data.hotlines.forEach(h => {
                    const card = document.createElement("div");
                    card.className = "resource-card";
                    card.innerHTML = `
                        <div class="resource-badge">${h.category}</div>
                        <h3>${h.name}</h3>
                        <p>${h.description}</p>
                        <div class="contact-box">
                            ${h.phone ? `<div><strong>Phone:</strong> <a href="tel:${h.phone.replace(/[^0-9]/g, '')}">${h.phone}</a></div>` : ''}
                            ${h.text_sms ? `<div><strong>Text SMS:</strong> ${h.text_sms}</div>` : ''}
                            ${h.website ? `<div><strong>Website:</strong> <a href="${h.website}" target="_blank" rel="noopener">${h.website.replace('https://', '')}</a></div>` : ''}
                            <div><small style="color:var(--text-muted)">${h.availability} • ${h.hours}</small></div>
                        </div>
                    `;
                    grid.appendChild(card);
                });

                const grounding = document.getElementById("grounding-container");
                grounding.innerHTML = "";
                data.grounding_techniques.forEach(g => {
                    const block = document.createElement("div");
                    block.style.marginTop = "14px";
                    block.innerHTML = `
                        <h4 style="margin-bottom:6px;">${g.title}</h4>
                        <p style="font-size:13px; color:var(--text-muted); margin-bottom:10px;">${g.description}</p>
                        ${g.steps.map(s => `<div class="exercise-step">${s}</div>`).join("")}
                    `;
                    grounding.appendChild(block);
                });
            } catch (err) {
                console.error("Resources error:", err);
            }
        }

        // ====================================================================
        // Privacy & Data Governance Logic
        // ====================================================================
        async function loadPreferences() {
            try {
                const res = await authFetch("/api/v1/user/preferences/" + userId);
                if (!res.ok) return;
                const data = await res.json();
                document.getElementById("toggle-save-history").checked = data.save_history;
                document.getElementById("toggle-enable-voice").checked = data.enable_voice;
                document.getElementById("toggle-enable-wearables").checked = data.enable_wearables;
            } catch (err) {
                console.error("Preferences error:", err);
            }
        }

        async function savePreferences() {
            const body = {
                save_history: document.getElementById("toggle-save-history").checked,
                enable_voice: document.getElementById("toggle-enable-voice").checked,
                enable_wearables: document.getElementById("toggle-enable-wearables").checked
            };
            try {
                await authFetch("/api/v1/user/preferences/" + userId, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
            } catch (err) {
                console.error("Error saving preferences:", err);
            }
        }

        document.getElementById("toggle-save-history").addEventListener("change", savePreferences);
        document.getElementById("toggle-enable-voice").addEventListener("change", savePreferences);
        document.getElementById("toggle-enable-wearables").addEventListener("change", savePreferences);

        // Export Data
        document.getElementById("export-data-btn").addEventListener("click", async () => {
            try {
                const res = await authFetch("/api/v1/user/export/" + userId, { method: "POST" });
                if (!res.ok) return alert("Export failed.");
                const data = await res.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `zenova_export_${userId}.json`;
                a.click();
                URL.revokeObjectURL(url);
            } catch (err) {
                alert("Error generating export archive.");
            }
        });

        // Purge Data (Right to be Forgotten)
        document.getElementById("purge-data-btn").addEventListener("click", async () => {
            const confirmMsg = "Are you sure you want to permanently delete all your conversation history, check-ins, and data?\\n\\nThis action cannot be undone.";
            if (confirm(confirmMsg)) {
                try {
                    const res = await authFetch("/api/v1/user/data/" + userId, { method: "DELETE" });
                    if (res.ok) {
                        alert("All personal data has been permanently deleted from ZENOVA.");
                        window.location.reload();
                    } else {
                        alert("Failed to delete data. Please try again.");
                    }
                } catch (err) {
                    alert("Error executing purge request.");
                }
            }
        });

        // Authentication state checking and session binding
        async function checkAuthAndInit() {
            try {
                const res = await authFetch('/api/v1/auth/me');
                if (res.ok) {
                    const user = await res.json();
                    userId = user.id;
                    localStorage.setItem('zenova_user', JSON.stringify(user));
                    localStorage.setItem('zenova_uid', user.id);
                    const pill = document.getElementById("user-pill");
                    if (pill) {
                        pill.textContent = user.display_name || user.email.split('@')[0];
                        pill.title = `Signed in as ${user.email} (${user.role})`;
                    }
                } else {
                    // Session not active or token invalid -> clear and redirect
                    localStorage.removeItem('zenova_token');
                    localStorage.removeItem('zenova_user');
                    window.location.href = '/login?redirect=/app';
                    return;
                }
            } catch (err) {
                console.warn("Auth check error:", err);
                const token = localStorage.getItem('zenova_token');
                const cachedUser = localStorage.getItem('zenova_user');
                if (token && cachedUser) {
                    try {
                        const u = JSON.parse(cachedUser);
                        userId = u.id;
                    } catch (e) {
                        window.location.href = '/login?redirect=/app';
                        return;
                    }
                } else {
                    window.location.href = '/login?redirect=/app';
                    return;
                }
            }
            loadSessionsList();
            loadPreferences();
        }

        async function handleLogout() {
            try {
                await authFetch('/api/v1/auth/logout', { method: 'POST' });
            } catch (err) {
                console.warn("Logout error:", err);
            } finally {
                localStorage.removeItem('zenova_token');
                localStorage.removeItem('zenova_user');
                window.location.href = '/login';
            }
        }

        // Initial setup
        window.addEventListener("DOMContentLoaded", () => {
            checkAuthAndInit();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
