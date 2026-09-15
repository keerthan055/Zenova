"""FastAPI routes and interactive Web UI for the Clinician Dashboard."""
from typing import Optional, Dict, Any
from fastapi import APIRouter, Header, Depends, Query, Request, HTTPException
from fastapi.responses import HTMLResponse

from zenova.schemas.escalation import UserRole
from zenova.schemas.dashboard import (
    DashboardOverviewResponse,
    PatientSummary,
    PatientListResponse,
    PatientTrendsResponse,
    PatientTimelineResponse,
    PatientConversationsResponse,
    PatientExplainabilityResponse,
    DashboardModulesResponse,
    DashboardAuditLogsResponse,
    CLINICAL_DISCLAIMER_NOTICE
)
from zenova.dashboard.service import ClinicianDashboardService
from zenova.dashboard.registry import DashboardModuleRegistry
from zenova.db.session import get_db_session
from zenova.core.logging import get_logger
from zenova.db.models import UserModel
from zenova.auth.dependencies import get_optional_user

logger = get_logger("zenova.api.routes.dashboard")

router = APIRouter(tags=["Clinician Dashboard"])

_registry = DashboardModuleRegistry()


def get_user_context(
    x_user_role: str = Header(default="clinician", alias="X-User-Role"),
    x_user_id: str = Header(default="dr_smith", alias="X-User-ID"),
    auth_user: Optional[UserModel] = Depends(get_optional_user)
) -> tuple[UserRole, str]:
    """Validate caller RBAC role and ID with authenticated session precedence."""
    if auth_user is not None:
        if auth_user.role not in ("clinician", "admin"):
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: User role '{auth_user.role}' is not authorized to access clinical decision-support."
            )
        role = UserRole.CLINICIAN if auth_user.role == "clinician" else UserRole.SYSTEM_ADMIN
        return role, auth_user.id

    try:
        role = UserRole(x_user_role.lower())
    except ValueError:
        raise HTTPException(status_code=403, detail=f"Invalid or unrecognized user role: '{x_user_role}'")
    return role, x_user_id


@router.get("/api/v1/dashboard/modules", response_model=DashboardModulesResponse)
async def list_dashboard_modules(
    category: Optional[str] = Query(default=None),
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Enumerate all extensible dashboard widgets and analytical modules."""
    role, user_id = user_ctx
    if role == UserRole.PATIENT:
        raise HTTPException(status_code=403, detail="Patients are prohibited from accessing dashboard modules.")
    modules = _registry.list_modules(category=category)
    return DashboardModulesResponse(count=len(modules), modules=modules)


@router.get("/api/v1/dashboard/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview(
    request: Request,
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Retrieve population-level overview metrics, alerts, risk distribution, and health."""
    role, user_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_overview(actor_role=role, actor_id=user_id, ip_address=client_ip)


@router.get("/api/v1/dashboard/users", response_model=PatientListResponse)
async def list_dashboard_users(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """List active patients for clinician selection and triage."""
    role, user_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.list_patients(actor_role=role, actor_id=user_id, limit=limit, ip_address=client_ip)


@router.get("/api/v1/dashboard/users/{user_id}/summary", response_model=PatientSummary)
async def get_patient_summary(
    user_id: str,
    request: Request,
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Fetch individual patient profile and clinical baseline summary."""
    role, actor_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_patient_summary(user_id=user_id, actor_role=role, actor_id=actor_id, ip_address=client_ip)


@router.get("/api/v1/dashboard/users/{user_id}/trends", response_model=PatientTrendsResponse)
async def get_patient_trends(
    user_id: str,
    request: Request,
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Fetch longitudinal time-series trajectories for an individual patient."""
    role, actor_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_patient_trends(user_id=user_id, actor_role=role, actor_id=actor_id, ip_address=client_ip)


@router.get("/api/v1/dashboard/users/{user_id}/timeline", response_model=PatientTimelineResponse)
async def get_patient_timeline(
    user_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Fetch unified chronological longitudinal timeline."""
    role, actor_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_patient_timeline(user_id=user_id, actor_role=role, actor_id=actor_id, limit=limit, ip_address=client_ip)


@router.get("/api/v1/dashboard/users/{user_id}/conversations", response_model=PatientConversationsResponse)
async def get_patient_conversations(
    user_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Fetch conversation turns and turn-level ML explainability cards."""
    role, actor_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_patient_conversations(user_id=user_id, actor_role=role, actor_id=actor_id, limit=limit, ip_address=client_ip)


@router.get("/api/v1/dashboard/users/{user_id}/explainability", response_model=PatientExplainabilityResponse)
async def get_patient_explainability(
    user_id: str,
    request: Request,
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Fetch comprehensive ML explainability cards for all active models."""
    role, actor_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_patient_explainability(user_id=user_id, actor_role=role, actor_id=actor_id, ip_address=client_ip)


@router.get("/api/v1/dashboard/system-status")
async def get_dashboard_system_status(
    request: Request,
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
) -> Dict[str, Any]:
    """Inspect model registry status, active providers, and hardware devices."""
    role, user_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_system_status(actor_role=role, actor_id=user_id, ip_address=client_ip)


@router.get("/api/v1/dashboard/audit-logs", response_model=DashboardAuditLogsResponse)
async def get_dashboard_audit_logs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_ctx: tuple[UserRole, str] = Depends(get_user_context)
):
    """Retrieve compliance audit trail (restricted to auditor & supervisor)."""
    role, user_id = user_ctx
    client_ip = request.client.host if request.client else "127.0.0.1"
    async with get_db_session() as session:
        service = ClinicianDashboardService(session)
        return await service.get_audit_logs(actor_role=role, actor_id=user_id, limit=limit, offset=offset, ip_address=client_ip)


@router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_ui():
    """Serves the self-contained Clinician Dashboard Web UI application."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZENOVA Clinician Decision-Support Portal</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-surface: #1e293b;
            --bg-card: #283548;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-teal: #0d9488;
            --accent-teal-light: #14b8a6;
            --accent-blue: #3b82f6;
            --critical-red: #ef4444;
            --warning-amber: #f59e0b;
            --safe-green: #10b981;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-family);
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
        }

        /* Top Header */
        header {
            background-color: var(--bg-surface);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-badge {
            background: linear-gradient(135deg, #0d9488, #3b82f6);
            color: #ffffff;
            font-weight: 800;
            font-size: 16px;
            padding: 6px 12px;
            border-radius: 6px;
            letter-spacing: 1px;
        }
        .header-title h1 {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }
        .header-title p {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .role-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-card);
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        .role-selector label { font-size: 12px; color: var(--text-secondary); font-weight: 600; }
        .role-selector select {
            background: transparent;
            color: var(--text-primary);
            border: none;
            outline: none;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--safe-green);
            background: rgba(16, 185, 129, 0.1);
            padding: 6px 10px;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--safe-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--safe-green);
        }

        /* Disclaimer Banner */
        .clinical-banner {
            background: rgba(245, 158, 11, 0.12);
            border-bottom: 1px solid rgba(245, 158, 11, 0.3);
            color: #fef3c7;
            padding: 8px 24px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .banner-icon { font-weight: bold; color: var(--warning-amber); font-size: 13px; }

        /* Main Layout */
        .app-container {
            padding: 20px 24px;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Tabs */
        .tab-nav {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 600;
            padding: 10px 18px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            color: var(--text-primary);
        }
        .tab-btn.active {
            color: var(--accent-teal-light);
            border-bottom-color: var(--accent-teal-light);
        }

        /* Card Component */
        .card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .card-header h2 {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-tile {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }
        .metric-tile .title { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; }
        .metric-tile .value { font-size: 28px; font-weight: 800; margin: 4px 0; }
        .metric-tile .subtext { font-size: 11px; color: var(--text-secondary); }

        /* Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            color: var(--text-secondary);
            font-weight: 600;
            background: rgba(255, 255, 255, 0.02);
        }
        tr:hover {
            background: rgba(255, 255, 255, 0.03);
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-high { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-medium { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-info { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }

        /* Timeline Items */
        .timeline-container {
            position: relative;
            padding-left: 24px;
            border-left: 2px solid var(--border-color);
            margin-left: 8px;
        }
        .timeline-item {
            position: relative;
            margin-bottom: 24px;
        }
        .timeline-dot {
            position: absolute;
            left: -31px;
            top: 4px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--accent-teal-light);
            border: 2px solid var(--bg-surface);
        }
        .timeline-dot.critical { background-color: var(--critical-red); }
        .timeline-dot.warning { background-color: var(--warning-amber); }
        .timeline-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px 16px;
        }
        .timeline-card .header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .timeline-card .time { font-size: 11px; color: var(--text-secondary); }

        /* Explainability Cards */
        .explain-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 16px;
        }
        .explain-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .explain-card h3 { font-size: 14px; font-weight: 700; color: var(--accent-teal-light); margin-bottom: 4px; }
        .explain-card .version { font-size: 11px; color: var(--text-secondary); margin-bottom: 12px; }
        .explain-card .signals {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
            padding: 8px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
            color: #cbd5e1;
        }
        .explain-card .limitations {
            font-size: 11px;
            color: #94a3b8;
            border-top: 1px solid var(--border-color);
            padding-top: 10px;
            margin-top: 10px;
        }

        /* Controls */
        select, button {
            font-family: inherit;
        }
        .btn-action {
            background: var(--accent-teal);
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }
        .btn-action:hover { background: var(--accent-teal-light); }

        .hidden { display: none !important; }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="header-brand">
            <div class="logo-badge">ZENOVA</div>
            <div class="header-title">
                <h1>Clinician Decision-Support Portal</h1>
                <p>Multi-Modal Mental Health & Crisis Monitoring Engine</p>
            </div>
        </div>
        <div class="header-controls">
            <div class="role-selector">
                <label for="role-select">Current Role:</label>
                <select id="role-select" onchange="handleRoleChange()">
                    <option value="clinician" selected>Clinician (Dr. Smith)</option>
                    <option value="triage_supervisor">Triage Supervisor</option>
                    <option value="auditor">Compliance Auditor</option>
                    <option value="system_admin">System Admin</option>
                </select>
            </div>
            <div class="status-indicator">
                <div class="pulse-dot"></div>
                <span>Pipeline Operational</span>
            </div>
        </div>
    </header>

    <!-- Clinical Disclaimer Banner -->
    <div class="clinical-banner">
        <span class="banner-icon">⚠️</span>
        <span><strong>CLINICAL NOTICE:</strong> Algorithmic predictions represent statistical pattern proxies and interaction strategies. They <strong>DO NOT</strong> constitute psychiatric diagnoses, DSM-5 medical evaluations, or emergency dispatch directives. Professional clinical judgment always supersedes automated outputs.</span>
    </div>

    <!-- Main Content -->
    <div class="app-container">
        <!-- Tab Navigation -->
        <nav class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
            <button class="tab-btn" onclick="switchTab('patients')">Patient Analysis</button>
            <button class="tab-btn" onclick="switchTab('timeline')">Longitudinal Timeline</button>
            <button class="tab-btn" onclick="switchTab('explainability')">ML Explainability Inspector</button>
            <button class="tab-btn" onclick="switchTab('audit')">Compliance Audit Trail</button>
        </nav>

        <!-- TAB 1: OVERVIEW -->
        <section id="tab-overview">
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="title">Active Alerts</div>
                    <div class="value" id="val-active-alerts" style="color: var(--critical-red);">-</div>
                    <div class="subtext">Pending / In-Review clinical alerts</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Critical / High Severity</div>
                    <div class="value" id="val-critical-alerts" style="color: var(--warning-amber);">-</div>
                    <div class="subtext">Immediate attention warranted</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Active Patients</div>
                    <div class="value" id="val-active-patients" style="color: var(--safe-green);">-</div>
                    <div class="subtext">In longitudinal dialogue</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Total Sessions</div>
                    <div class="value" id="val-total-sessions" style="color: var(--accent-blue);">-</div>
                    <div class="subtext">Logged in ZENOVA DB</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Active Crisis & Escalation Incidents</h2>
                    <button class="btn-action" onclick="loadOverview()">Refresh</button>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Alert ID</th>
                            <th>Patient</th>
                            <th>Severity</th>
                            <th>Trigger Type</th>
                            <th>Status</th>
                            <th>Created At</th>
                        </tr>
                    </thead>
                    <tbody id="overview-alerts-tbody">
                        <tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">Loading alerts...</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Population Risk Distribution</h2>
                </div>
                <div id="risk-distribution-bars" style="display: flex; gap: 12px; flex-wrap: wrap;"></div>
            </div>
        </section>

        <!-- TAB 2: PATIENT ANALYSIS -->
        <section id="tab-patients" class="hidden">
            <div class="card" style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <label for="patient-select" style="font-weight: 700;">Select Patient:</label>
                    <select id="patient-select" onchange="loadSelectedPatient()" style="background: var(--bg-card); color: white; padding: 6px 12px; border-radius: 4px; border: 1px solid var(--border-color);">
                        <option value="">Loading patients...</option>
                    </select>
                </div>
                <button class="btn-action" onclick="loadPatientList()">Reload Patients</button>
            </div>

            <div class="metric-grid" id="patient-summary-grid">
                <div class="metric-tile">
                    <div class="title">Current Triage Risk</div>
                    <div class="value" id="p-risk-badge" style="font-size: 20px;">-</div>
                    <div class="subtext">Multi-Task Risk Transformer</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Baseline Status</div>
                    <div class="value" id="p-baseline-status" style="font-size: 20px;">-</div>
                    <div class="subtext">Personal Longitudinal Profile</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Baseline Deviation</div>
                    <div class="value" id="p-deviation-score" style="font-size: 20px;">-</div>
                    <div class="subtext">Z-Score shift from mean</div>
                </div>
                <div class="metric-tile">
                    <div class="title">Open Alerts</div>
                    <div class="value" id="p-open-alerts" style="font-size: 20px; color: var(--warning-amber);">-</div>
                    <div class="subtext">Requires clinical action</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Turn-by-Turn Conversation & Model Predictions</h2>
                </div>
                <div id="patient-conversations-list" style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="color: var(--text-secondary);">Select a patient to inspect conversation transcripts.</div>
                </div>
            </div>
        </section>

        <!-- TAB 3: TIMELINE -->
        <section id="tab-timeline" class="hidden">
            <div class="card">
                <div class="card-header">
                    <h2>Unified Multi-Modal Event Stream</h2>
                    <span style="font-size: 12px; color: var(--text-secondary);">Conversations, Voice Samples, Behavioral Deviations, Escalations</span>
                </div>
                <div class="timeline-container" id="timeline-feed">
                    <div style="color: var(--text-secondary);">Select a patient to view their timeline.</div>
                </div>
            </div>
        </section>

        <!-- TAB 4: EXPLAINABILITY -->
        <section id="tab-explainability" class="hidden">
            <div class="card">
                <div class="card-header">
                    <h2>Machine Learning Model Transparency & Explainability Inspector</h2>
                </div>
                <div class="explain-grid" id="explainability-cards">
                    <div style="color: var(--text-secondary);">Loading explainability cards...</div>
                </div>
            </div>
        </section>

        <!-- TAB 5: AUDIT TRAIL -->
        <section id="tab-audit" class="hidden">
            <div class="card">
                <div class="card-header">
                    <h2>Dashboard Access & PHI Compliance Log</h2>
                    <button class="btn-action" onclick="loadAuditLogs()">Refresh Audit Trail</button>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Access ID</th>
                            <th>Actor</th>
                            <th>Role</th>
                            <th>Action</th>
                            <th>Endpoint</th>
                            <th>Target User</th>
                            <th>IP Address</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody id="audit-logs-tbody">
                        <tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">Select Refresh to load logs.</td></tr>
                    </tbody>
                </table>
            </div>
        </section>
    </div>

    <!-- JavaScript Application Logic -->
    <script>
        let currentRole = "clinician";
        let currentActorId = "dr_smith";
        let selectedUserId = "";

        function getHeaders() {
            return {
                "X-User-Role": currentRole,
                "X-User-ID": currentActorId,
                "Content-Type": "application/json"
            };
        }

        function handleRoleChange() {
            currentRole = document.getElementById("role-select").value;
            currentActorId = currentRole === "auditor" ? "auditor_1" : (currentRole === "system_admin" ? "admin_ops" : "dr_smith");
            loadOverview();
            loadPatientList();
            if (selectedUserId) {
                loadSelectedPatient();
            }
        }

        function switchTab(tabId) {
            document.querySelectorAll(".tab-nav .tab-btn").forEach(btn => btn.classList.remove("active"));
            event.target.classList.add("active");

            document.querySelectorAll("section[id^='tab-']").forEach(sec => sec.classList.add("hidden"));
            document.getElementById("tab-" + tabId).classList.remove("hidden");

            if (tabId === "overview") loadOverview();
            if (tabId === "patients") loadPatientList();
            if (tabId === "timeline" && selectedUserId) loadTimeline(selectedUserId);
            if (tabId === "explainability") loadExplainability(selectedUserId || "default");
            if (tabId === "audit") loadAuditLogs();
        }

        async function loadOverview() {
            try {
                const res = await fetch("/api/v1/dashboard/overview", { headers: getHeaders() });
                if (!res.ok) {
                    if (res.status === 403) alert("Access Denied: Unprivileged role.");
                    return;
                }
                const data = await res.json();
                document.getElementById("val-active-alerts").innerText = data.active_alerts_count;
                const crit = (data.alerts_by_severity.critical || 0) + (data.alerts_by_severity.high || 0);
                document.getElementById("val-critical-alerts").innerText = crit;
                document.getElementById("val-active-patients").innerText = data.active_users_count;
                document.getElementById("val-total-sessions").innerText = data.total_sessions_count;

                // Alerts table
                const tbody = document.getElementById("overview-alerts-tbody");
                tbody.innerHTML = "";
                if (data.recent_escalations.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='6' style='text-align:center; color:#94a3b8;'>No active escalation incidents.</td></tr>";
                } else {
                    data.recent_escalations.forEach(a => {
                        const sevClass = a.severity === "critical" ? "badge-critical" : (a.severity === "high" ? "badge-high" : "badge-medium");
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td><code>${a.alert_id}</code></td>
                            <td><strong>${a.user_id}</strong></td>
                            <td><span class="badge ${sevClass}">${a.severity}</span></td>
                            <td>${a.trigger_type}</td>
                            <td><span class="badge badge-info">${a.status}</span></td>
                            <td>${new Date(a.created_at).toLocaleString()}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                // Risk distribution
                const riskContainer = document.getElementById("risk-distribution-bars");
                riskContainer.innerHTML = "";
                for (const [risk, count] of Object.entries(data.risk_distribution)) {
                    const d = document.createElement("div");
                    d.style.background = "var(--bg-card)";
                    d.style.padding = "8px 16px";
                    d.style.borderRadius = "4px";
                    d.style.border = "1px solid var(--border-color)";
                    d.innerHTML = `<span style='text-transform: capitalize;'>${risk.replace('_', ' ')}:</span> <strong>${count}</strong>`;
                    riskContainer.appendChild(d);
                }
            } catch (err) {
                console.error("Overview error:", err);
            }
        }

        async function loadPatientList() {
            try {
                const res = await fetch("/api/v1/dashboard/users", { headers: getHeaders() });
                if (!res.ok) return;
                const data = await res.json();
                const sel = document.getElementById("patient-select");
                sel.innerHTML = "";
                if (data.patients.length === 0) {
                    sel.innerHTML = "<option value=''>No active patients found</option>";
                    return;
                }
                data.patients.forEach(p => {
                    const opt = document.createElement("option");
                    opt.value = p.user_id;
                    opt.innerText = `${p.user_id} [Risk: ${p.current_risk_level}]`;
                    sel.appendChild(opt);
                });
                selectedUserId = data.patients[0].user_id;
                loadSelectedPatient();
            } catch (err) {
                console.error("Error loading patients:", err);
            }
        }

        async function loadSelectedPatient() {
            const sel = document.getElementById("patient-select");
            selectedUserId = sel.value;
            if (!selectedUserId) return;

            // Summary
            try {
                const sRes = await fetch(`/api/v1/dashboard/users/${selectedUserId}/summary`, { headers: getHeaders() });
                if (sRes.ok) {
                    const s = await sRes.json();
                    document.getElementById("p-risk-badge").innerText = s.current_risk_level.toUpperCase();
                    document.getElementById("p-baseline-status").innerText = s.baseline_status;
                    document.getElementById("p-deviation-score").innerText = s.baseline_deviation_score.toFixed(2) + " σ";
                    document.getElementById("p-open-alerts").innerText = s.open_alerts_count;
                }

                // Conversations
                const cRes = await fetch(`/api/v1/dashboard/users/${selectedUserId}/conversations`, { headers: getHeaders() });
                if (cRes.ok) {
                    const cData = await cRes.json();
                    const cList = document.getElementById("patient-conversations-list");
                    cList.innerHTML = "";
                    if (cData.turns.length === 0) {
                        cList.innerHTML = "<div style='color: var(--text-secondary);'>No conversation turns recorded.</div>";
                    } else {
                        cData.turns.forEach(t => {
                            const isUser = t.speaker === "user";
                            const d = document.createElement("div");
                            d.style.background = isUser ? "rgba(59, 130, 246, 0.08)" : "rgba(13, 148, 136, 0.08)";
                            d.style.border = "1px solid " + (isUser ? "rgba(59, 130, 246, 0.2)" : "rgba(13, 148, 136, 0.2)");
                            d.style.borderRadius = "6px";
                            d.style.padding = "12px";

                            let metaBadges = "";
                            if (t.emotion) metaBadges += `<span class='badge badge-info'>Emotion: ${t.emotion.primary_emotion || 'none'}</span> `;
                            if (t.risk) metaBadges += `<span class='badge badge-high'>Risk: ${t.risk.risk_level || 'no_risk'}</span> `;
                            if (t.strategy) metaBadges += `<span class='badge badge-low'>Strategy: ${t.strategy.selected_strategy || 'none'}</span> `;

                            d.innerHTML = `
                                <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
                                    <strong>${t.speaker.toUpperCase()} (Turn ${t.turn_id})</strong>
                                    <span style='font-size: 11px; color: var(--text-secondary);'>${t.created_at ? new Date(t.created_at).toLocaleTimeString() : ''}</span>
                                </div>
                                <div style='margin-bottom: 8px;'>${t.content}</div>
                                <div>${metaBadges}</div>
                            `;
                            cList.appendChild(d);
                        });
                    }
                }
            } catch (err) {
                console.error("Error loading patient detail:", err);
            }
        }

        async function loadTimeline(uid) {
            try {
                const res = await fetch(`/api/v1/dashboard/users/${uid}/timeline`, { headers: getHeaders() });
                if (!res.ok) return;
                const data = await res.json();
                const container = document.getElementById("timeline-feed");
                container.innerHTML = "";
                if (data.events.length === 0) {
                    container.innerHTML = "<div style='color: var(--text-secondary);'>No longitudinal events found.</div>";
                    return;
                }
                data.events.forEach(ev => {
                    const item = document.createElement("div");
                    item.className = "timeline-item";
                    const isCrit = ev.severity === "critical" || ev.severity === "high";
                    item.innerHTML = `
                        <div class="timeline-dot ${isCrit ? 'critical' : ''}"></div>
                        <div class="timeline-card">
                            <div class="header">
                                <strong>${ev.title}</strong>
                                <span class="time">${new Date(ev.timestamp).toLocaleString()}</span>
                            </div>
                            <div style="font-size: 13px; margin-top: 4px;">${ev.description}</div>
                        </div>
                    `;
                    container.appendChild(item);
                });
            } catch (err) {
                console.error("Timeline error:", err);
            }
        }

        async function loadExplainability(uid) {
            try {
                const res = await fetch(`/api/v1/dashboard/users/${uid || 'default'}/explainability`, { headers: getHeaders() });
                if (!res.ok) return;
                const data = await res.json();
                const container = document.getElementById("explainability-cards");
                container.innerHTML = "";
                data.model_cards.forEach(c => {
                    const card = document.createElement("div");
                    card.className = "explain-card";
                    card.innerHTML = `
                        <div>
                            <h3>${c.model_name}</h3>
                            <div class="version">Version: <code>${c.model_version}</code> | Modality: <code>${c.modality}</code> | Confidence: <strong>${Math.round(c.confidence * 100)}%</strong></div>
                            <p style="font-size: 13px; margin-bottom: 8px;">${c.prediction_summary}</p>
                            <div class="signals">Signals: ${c.relevant_signals.join(", ")}</div>
                        </div>
                        <div class="limitations">
                            <strong>Limitations:</strong> ${c.limitations}
                        </div>
                    `;
                    container.appendChild(card);
                });
            } catch (err) {
                console.error("Explainability error:", err);
            }
        }

        async function loadAuditLogs() {
            try {
                const res = await fetch("/api/v1/dashboard/audit-logs", { headers: getHeaders() });
                if (!res.ok) {
                    if (res.status === 403) alert("Access Denied: Auditor or Supervisor role required.");
                    return;
                }
                const data = await res.json();
                const tbody = document.getElementById("audit-logs-tbody");
                tbody.innerHTML = "";
                if (data.logs.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='8' style='text-align:center; color:#94a3b8;'>No audit records logged yet.</td></tr>";
                    return;
                }
                data.logs.forEach(l => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><code>${l.access_id}</code></td>
                        <td>${l.actor_id}</td>
                        <td><span class="badge badge-info">${l.actor_role}</span></td>
                        <td><strong>${l.action}</strong></td>
                        <td><code>${l.endpoint}</code></td>
                        <td>${l.target_user_id || '-'}</td>
                        <td>${l.ip_address}</td>
                        <td>${new Date(l.timestamp).toLocaleString()}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error("Audit load error:", err);
            }
        }

        // Initialize on load
        window.addEventListener("DOMContentLoaded", () => {
            loadOverview();
            loadPatientList();
            loadExplainability("default");
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)
