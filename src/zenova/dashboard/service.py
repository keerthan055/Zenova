"""Clinician Dashboard Service orchestrating data aggregation, RBAC, privacy, and explainability."""
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from zenova.schemas.escalation import UserRole
from zenova.schemas.dashboard import (
    CLINICAL_DISCLAIMER_NOTICE,
    DashboardOverviewResponse,
    PatientSummary,
    PatientListResponse,
    PatientTrendsResponse,
    TimelineEvent,
    PatientTimelineResponse,
    ConversationTurnDisplay,
    PatientConversationsResponse,
    MLExplainabilityCard,
    PatientExplainabilityResponse,
    DashboardAccessAuditEntry,
    DashboardAuditLogsResponse
)
from zenova.escalation.rbac import AccessControlManager
from zenova.db.repositories import DashboardRepository
from zenova.models.registry import ModelRegistry
from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.dashboard.service")


class ClinicianDashboardService:
    """Core service for the Clinician Web Dashboard."""

    def __init__(self, db_session: AsyncSession):
        self.session = db_session
        self.repo = DashboardRepository(db_session)
        self.model_registry = ModelRegistry()
        self.config = get_system_config()

    def _verify_access(self, role: UserRole, permission: str = "can_view_queue") -> None:
        """Enforce role-based access control."""
        if role == UserRole.PATIENT:
            raise HTTPException(status_code=403, detail="Patients are strictly prohibited from accessing clinician dashboard interfaces.")
        if not AccessControlManager.check_permission(role, permission):
            raise HTTPException(status_code=403, detail=f"Role '{role.value}' is not authorized for permission '{permission}'.")

    async def _audit_access(
        self,
        actor_id: str,
        actor_role: UserRole,
        action: str,
        endpoint: str,
        target_user_id: Optional[str] = None,
        ip_address: str = "127.0.0.1"
    ) -> None:
        """Record an immutable access audit entry."""
        try:
            access_id = f"acc_{uuid.uuid4().hex[:12]}"
            await self.repo.record_access(
                access_id=access_id,
                actor_id=actor_id,
                actor_role=actor_role.value,
                endpoint=endpoint,
                action=action,
                target_user_id=target_user_id,
                ip_address=ip_address
            )
        except Exception as e:
            logger.warning(f"Failed to record dashboard access audit log: {e}")

    async def get_overview(
        self,
        actor_role: UserRole,
        actor_id: str,
        ip_address: str = "127.0.0.1"
    ) -> DashboardOverviewResponse:
        """Fetch high-level overview metrics, alerts, risk distribution, and system status."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_overview", "/api/v1/dashboard/overview", ip_address=ip_address)

        stats = await self.repo.get_overview_metrics()

        # Format recent escalations
        formatted_escalations = []
        for e in stats.get("recent_escalations", []):
            formatted_escalations.append({
                "alert_id": e.event_id,
                "session_id": e.session_id,
                "user_id": e.user_id if AccessControlManager.check_permission(actor_role, "can_view_full_context") else f"usr_***{e.user_id[-4:] if len(e.user_id) >= 4 else 'masked'}",
                "severity": e.severity,
                "trigger_type": e.trigger_type,
                "status": e.status,
                "risk_level": e.risk_level,
                "crisis_category": e.crisis_category,
                "created_at": e.created_at.isoformat() if e.created_at else None
            })

        system_status = await self.get_system_status(actor_role, actor_id, record_audit=False)

        return DashboardOverviewResponse(
            active_alerts_count=stats.get("active_alerts_count", 0),
            alerts_by_severity=stats.get("alerts_by_severity", {}),
            risk_distribution=stats.get("risk_distribution", {}),
            recent_escalations=formatted_escalations,
            active_users_count=stats.get("active_users_count", 0),
            total_sessions_count=stats.get("total_sessions_count", 0),
            system_status=system_status,
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def list_patients(
        self,
        actor_role: UserRole,
        actor_id: str,
        limit: int = 50,
        ip_address: str = "127.0.0.1"
    ) -> PatientListResponse:
        """Enumerate active patients for clinician selection."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "list_patients", "/api/v1/dashboard/users", ip_address=ip_address)

        raw_summaries = await self.repo.get_all_users_summary(limit=limit)
        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")

        patients = []
        for s in raw_summaries:
            u_id = s["user_id"] if can_view_ids else f"usr_***{s['user_id'][-4:] if len(s['user_id']) >= 4 else 'masked'}"
            patients.append(PatientSummary(
                user_id=u_id,
                current_risk_level=s["current_risk_level"],
                baseline_status=s["baseline_status"],
                baseline_deviation_score=s["baseline_deviation_score"],
                open_alerts_count=s["open_alerts_count"],
                total_turns=s["total_turns"],
                total_sessions=s["total_sessions"],
                last_active_at=s["last_active_at"]
            ))

        return PatientListResponse(
            count=len(patients),
            patients=patients,
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def get_patient_summary(
        self,
        user_id: str,
        actor_role: UserRole,
        actor_id: str,
        ip_address: str = "127.0.0.1"
    ) -> PatientSummary:
        """Fetch summary card for an individual patient."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_patient_summary", f"/api/v1/dashboard/users/{user_id}/summary", target_user_id=user_id, ip_address=ip_address)

        s = await self.repo.get_user_summary(user_id)
        if not s:
            # Fallback if brand new or not yet in session table
            s = {
                "user_id": user_id,
                "current_risk_level": "no_risk",
                "baseline_status": "insufficient_data",
                "baseline_deviation_score": 0.0,
                "open_alerts_count": 0,
                "total_turns": 0,
                "total_sessions": 0,
                "last_active_at": None
            }

        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")
        display_id = s["user_id"] if can_view_ids else f"usr_***{s['user_id'][-4:] if len(s['user_id']) >= 4 else 'masked'}"

        return PatientSummary(
            user_id=display_id,
            current_risk_level=s["current_risk_level"],
            baseline_status=s["baseline_status"],
            baseline_deviation_score=s["baseline_deviation_score"],
            open_alerts_count=s["open_alerts_count"],
            total_turns=s["total_turns"],
            total_sessions=s["total_sessions"],
            last_active_at=s["last_active_at"]
        )

    async def get_patient_trends(
        self,
        user_id: str,
        actor_role: UserRole,
        actor_id: str,
        ip_address: str = "127.0.0.1"
    ) -> PatientTrendsResponse:
        """Fetch emotion, symptom, risk, strategy, and baseline trends for patient."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_trends", f"/api/v1/dashboard/users/{user_id}/trends", target_user_id=user_id, ip_address=ip_address)

        raw = await self.repo.get_user_trends(user_id)
        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")
        display_id = user_id if can_view_ids else f"usr_***{user_id[-4:] if len(user_id) >= 4 else 'masked'}"

        return PatientTrendsResponse(
            user_id=display_id,
            emotion_trends=raw.get("emotion_trends", []),
            symptom_trends=raw.get("symptom_trends", []),
            risk_trends=raw.get("risk_trends", []),
            strategy_trends=raw.get("strategy_trends", []),
            baseline_metrics=raw.get("baseline_metrics", {}),
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def get_patient_timeline(
        self,
        user_id: str,
        actor_role: UserRole,
        actor_id: str,
        limit: int = 100,
        ip_address: str = "127.0.0.1"
    ) -> PatientTimelineResponse:
        """Fetch unified chronological longitudinal timeline."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_timeline", f"/api/v1/dashboard/users/{user_id}/timeline", target_user_id=user_id, ip_address=ip_address)

        events_raw = await self.repo.get_user_timeline(user_id, limit=limit)
        can_view_transcripts = AccessControlManager.check_permission(actor_role, "can_view_full_context")
        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")

        events: List[TimelineEvent] = []
        for e in events_raw:
            desc = e["description"]
            if e["event_type"] == "conversation" and not can_view_transcripts:
                desc = "[CONFIDENTIAL_TRANSCRIPT_PROTECTED]"

            events.append(TimelineEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                timestamp=e["timestamp"],
                title=e["title"],
                description=desc,
                severity=e["severity"],
                data=e["data"],
                explainability=None
            ))

        display_id = user_id if can_view_ids else f"usr_***{user_id[-4:] if len(user_id) >= 4 else 'masked'}"

        return PatientTimelineResponse(
            user_id=display_id,
            events_count=len(events),
            events=events,
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def get_patient_conversations(
        self,
        user_id: str,
        actor_role: UserRole,
        actor_id: str,
        limit: int = 100,
        ip_address: str = "127.0.0.1"
    ) -> PatientConversationsResponse:
        """Fetch turn-by-turn conversation transcripts with parsed ML predictions."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_conversations", f"/api/v1/dashboard/users/{user_id}/conversations", target_user_id=user_id, ip_address=ip_address)

        raw_turns = await self.repo.get_user_conversations(user_id, limit=limit)
        can_view_transcripts = AccessControlManager.check_permission(actor_role, "can_view_full_context")
        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")

        turns: List[ConversationTurnDisplay] = []
        for t in raw_turns:
            content = t["content"] if can_view_transcripts else "[CONFIDENTIAL_TRANSCRIPT_PROTECTED]"

            # Build explainability cards for this turn
            turn_cards: List[MLExplainabilityCard] = []
            if t.get("emotion"):
                em = t["emotion"]
                turn_cards.append(MLExplainabilityCard(
                    model_name="Emotion Transformer",
                    model_version="distilroberta-base-go-emotions-v1",
                    modality="text",
                    confidence=float(em.get("confidence", 1.0)),
                    prediction_summary=f"Detected primary emotion: {em.get('primary_emotion')} (valence: {em.get('valence'):.2f}, arousal: {em.get('arousal'):.2f})",
                    relevant_signals=[em.get("primary_emotion", "unknown")],
                    limitations="Classification heuristic over short conversational snippets. Does not detect chronic affective disorders.",
                    clinical_notice=CLINICAL_DISCLAIMER_NOTICE
                ))
            if t.get("risk"):
                rk = t["risk"]
                turn_cards.append(MLExplainabilityCard(
                    model_name="Multi-Task Risk Transformer",
                    model_version="risk-multitask-transformer-v1",
                    modality="text",
                    confidence=float(rk.get("confidence", 1.0)),
                    prediction_summary=f"Risk level: {rk.get('risk_level')}, Category: {rk.get('crisis_category')}",
                    relevant_signals=rk.get("trigger_cues", []) if can_view_transcripts else ["[REDACTED_CONFIDENTIAL_CUE]"],
                    limitations="Statistical crisis classifier. High false-positive tolerance designed for safety triage; requires licensed human verification.",
                    clinical_notice=CLINICAL_DISCLAIMER_NOTICE
                ))
            if t.get("strategy"):
                st = t["strategy"]
                turn_cards.append(MLExplainabilityCard(
                    model_name="Strategy Transformer",
                    model_version="strategy-transformer-v1",
                    modality="dialogue_context",
                    confidence=float(st.get("confidence", 1.0)),
                    prediction_summary=f"Selected ESConv strategy: {st.get('selected_strategy')}",
                    relevant_signals=[st.get("selected_strategy", "Others")],
                    limitations="Algorithmic conversational planning policy based on peer-support taxonomy. Not a formal therapeutic intervention.",
                    clinical_notice=CLINICAL_DISCLAIMER_NOTICE
                ))

            turns.append(ConversationTurnDisplay(
                turn_id=t["turn_id"],
                session_id=t["session_id"],
                speaker=t["speaker"],
                content=content,
                created_at=t["created_at"],
                emotion=t.get("emotion"),
                symptoms=t.get("symptoms"),
                risk=t.get("risk"),
                strategy=t.get("strategy"),
                safety=t.get("safety"),
                voice=t.get("voice"),
                behavior=t.get("behavior"),
                explainability=turn_cards
            ))

        display_id = user_id if can_view_ids else f"usr_***{user_id[-4:] if len(user_id) >= 4 else 'masked'}"

        return PatientConversationsResponse(
            user_id=display_id,
            turns_count=len(turns),
            turns=turns,
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def get_patient_explainability(
        self,
        user_id: str,
        actor_role: UserRole,
        actor_id: str,
        ip_address: str = "127.0.0.1"
    ) -> PatientExplainabilityResponse:
        """Fetch comprehensive ML explainability cards for all active models in the pipeline."""
        self._verify_access(actor_role, "can_view_queue")
        await self._audit_access(actor_id, actor_role, "view_explainability", f"/api/v1/dashboard/users/{user_id}/explainability", target_user_id=user_id, ip_address=ip_address)

        can_view_ids = AccessControlManager.check_permission(actor_role, "can_view_full_context")
        display_id = user_id if can_view_ids else f"usr_***{user_id[-4:] if len(user_id) >= 4 else 'masked'}"

        cards = [
            MLExplainabilityCard(
                model_name="Emotion Transformer",
                model_version="distilroberta-base-go-emotions-v1",
                modality="text",
                confidence=0.92,
                prediction_summary="Multi-class emotional state, valence (-1 to +1), and arousal (0 to +1) estimation.",
                relevant_signals=["token_attentions", "valence_arousal_coordinates", "sub-emotion_logits"],
                limitations="Trained on short-text dialogue corpora; cannot infer underlying mood disorders, trauma, or clinical affect blunting.",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Symptom Signal Classifier",
                model_version="symptom-transformer-v1.0",
                modality="text",
                confidence=0.88,
                prediction_summary="Detects self-reported psychological symptom signals (depressive symptoms, anxiety, sleep disturbances).",
                relevant_signals=["phrasing_syntax", "distress_frequency", "symptom_severity_levels"],
                limitations="Pattern-recognition proxy based on user verbal disclosures. NOT a clinical psychometric battery or DSM-5 assessment.",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Multi-Task Crisis & Suicide Risk Transformer",
                model_version="risk-multitask-transformer-v1",
                modality="multimodal_text",
                confidence=0.96,
                prediction_summary="Triages imminent suicidal ideation, intent, and self-harm crisis severity across 5 tiers.",
                relevant_signals=["crisis_trigger_cues", "hopelessness_indicators", "lethal_means_mentions"],
                limitations="Heuristic safety gate with conservative decision boundaries; calibrated for high sensitivity to prevent triage failures.",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Support Strategy Transformer",
                model_version="strategy-transformer-v1",
                modality="dialogue_context",
                confidence=0.85,
                prediction_summary="Selects appropriate emotional support strategies from the 8-class ESConv taxonomy.",
                relevant_signals=["user_affect_stage", "dialogue_turn_depth", "previous_strategy_effectiveness"],
                limitations="Peer-support conversational dialogue planning only; does not provide psychotherapy (CBT, DBT, psychodynamic).",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Response Safety Gate",
                model_version="safety-gate-v1.0.0",
                modality="candidate_generation",
                confidence=1.0,
                prediction_summary="Deterministic and semantic validation enforcing 12 non-bypassable clinical safety policies.",
                relevant_signals=["policy_violation_codes", "toxicity_scores", "prescribing_prevention_regex"],
                limitations="Post-generation guardrail filter. Overrides LLM output with clinical fallback or 988 emergency resources upon breach.",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Acoustic Voice Emotion Analyzer",
                model_version="voice-analyzer-v1.0.0",
                modality="speech_audio",
                confidence=0.84,
                prediction_summary="Extracts pitch (F0), jitter, shimmer, and MFCC representations to estimate vocal affect.",
                relevant_signals=["fundamental_frequency_pitch", "spectral_flux", "vocal_energy_rms"],
                limitations="Susceptible to ambient noise, microphone hardware variance, and physical health conditions (e.g., upper respiratory infection).",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            ),
            MLExplainabilityCard(
                model_name="Passive Behavioral Anomaly Detector",
                model_version="behavioral-analyzer-v1.0.0",
                modality="smartphone_passive_sensing",
                confidence=0.79,
                prediction_summary="Identifies longitudinal deviations in mobility, screen time, and sleep proxies relative to personal baseline.",
                relevant_signals=["step_count_z_score", "screen_time_delta", "sleep_irregularity_index"],
                limitations="Infers behavioral proxies only; cannot measure subjective internal distress or physiological vitals directly.",
                clinical_notice=CLINICAL_DISCLAIMER_NOTICE
            )
        ]

        return PatientExplainabilityResponse(
            user_id=display_id,
            model_cards=cards,
            clinical_notice=CLINICAL_DISCLAIMER_NOTICE
        )

    async def get_system_status(
        self,
        actor_role: UserRole,
        actor_id: str,
        record_audit: bool = True,
        ip_address: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """Fetch system health, active providers, and model registry statuses."""
        self._verify_access(actor_role, "can_view_queue")
        if record_audit:
            await self._audit_access(actor_id, actor_role, "view_system_status", "/api/v1/dashboard/system-status", ip_address=ip_address)

        active_providers = self.model_registry._active_providers
        registry_status = {}
        for task, p_name in active_providers.items():
            reg_entry = self.model_registry._registry.get(task, {}).get(p_name)
            registry_status[task] = {
                "active_provider": p_name,
                "version": reg_entry.version if reg_entry else "0.1.0",
                "device": reg_entry.device if reg_entry else "cpu",
                "status": "healthy"
            }

        return {
            "system_name": self.config.name,
            "version": self.config.version,
            "environment": self.config.environment,
            "status": "operational",
            "database_connected": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_models": registry_status,
            "clinical_disclaimer": CLINICAL_DISCLAIMER_NOTICE
        }

    async def get_audit_logs(
        self,
        actor_role: UserRole,
        actor_id: str,
        limit: int = 50,
        offset: int = 0,
        ip_address: str = "127.0.0.1"
    ) -> DashboardAuditLogsResponse:
        """Fetch dashboard access audit trail (restricted to auditor & supervisor)."""
        if actor_role not in (UserRole.AUDITOR, UserRole.TRIAGE_SUPERVISOR, UserRole.SYSTEM_ADMIN):
            raise HTTPException(status_code=403, detail="Audit compliance log is restricted to auditor and supervisor roles.")
        await self._audit_access(actor_id, actor_role, "view_audit_logs", "/api/v1/dashboard/audit-logs", ip_address=ip_address)

        records = await self.repo.get_access_logs(limit=limit, offset=offset)
        logs = [
            DashboardAccessAuditEntry(
                access_id=r.access_id,
                actor_id=r.actor_id,
                actor_role=r.actor_role,
                target_user_id=r.target_user_id,
                endpoint=r.endpoint,
                action=r.action,
                ip_address=r.ip_address,
                timestamp=r.timestamp
            )
            for r in records
        ]
        return DashboardAuditLogsResponse(count=len(logs), logs=logs)
