import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from zenova.db.models import (
    UserSessionModel,
    ConversationTurnModel,
    EscalationEventModel,
    UserBaselineModel,
    UserObservationModel,
    BehavioralObservationModel,
    VoiceObservationModel,
    ContextSnapshotModel,
    SafetyAuditModel,
    EscalationAuditModel,
    DashboardAccessAuditModel,
    ExecutionTraceModel,
    UserPreferencesModel,
    UserCheckinModel
)
from zenova.schemas.orchestration import (
    PipelineTrace,
    TraceSpan,
    PipelineStatus,
    SpanStatus
)
from zenova.schemas.standard import (
    ConversationTurn,
    EscalationEvent,
    SpeakerRole,
    RiskLevel,
    CrisisCategory,
    EscalationStatus
)


class SessionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, session_id: str, user_id: str) -> UserSessionModel:
        stmt = select(UserSessionModel).where(UserSessionModel.session_id == session_id)
        result = await self.session.execute(stmt)
        record = result.scalars().first()
        if not record:
            record = UserSessionModel(session_id=session_id, user_id=user_id)
            self.session.add(record)
            await self.session.flush()
        return record


class TurnRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_turn(
        self,
        session_id: str,
        turn_id: int,
        speaker: str,
        content: str,
        emotion_json: Optional[str] = None,
        symptom_json: Optional[str] = None,
        risk_json: Optional[str] = None,
        behavior_json: Optional[str] = None,
        voice_json: Optional[str] = None,
        strategy_json: Optional[str] = None,
        safety_json: Optional[str] = None,
        context_json: Optional[str] = None,
        feedback_json: Optional[str] = None
    ) -> ConversationTurnModel:
        turn = ConversationTurnModel(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            content=content,
            emotion_json=emotion_json,
            symptom_json=symptom_json,
            risk_json=risk_json,
            behavior_json=behavior_json,
            voice_json=voice_json,
            strategy_json=strategy_json,
            safety_json=safety_json,
            context_json=context_json,
            feedback_json=feedback_json
        )
        self.session.add(turn)
        await self.session.flush()
        return turn

    async def get_history(self, session_id: str, limit: int = 50) -> List[ConversationTurnModel]:
        stmt = (
            select(ConversationTurnModel)
            .where(ConversationTurnModel.session_id == session_id)
            .order_by(ConversationTurnModel.turn_id.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class EscalationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_alert(
        self,
        alert_id: str,
        session_id: str,
        user_id: str,
        severity: str,
        trigger_type: str,
        risk_level: str,
        crisis_category: str,
        trigger_cues: List[str],
        reason_dict: Optional[Dict[str, Any]] = None,
        context_summary_dict: Optional[Dict[str, Any]] = None
    ) -> EscalationEventModel:
        now = datetime.now(timezone.utc)
        alert = EscalationEventModel(
            event_id=alert_id,
            session_id=session_id,
            user_id=user_id,
            severity=severity,
            trigger_type=trigger_type,
            risk_level=risk_level,
            crisis_category=crisis_category,
            trigger_cues_json=json.dumps(trigger_cues),
            reason_json=json.dumps(reason_dict) if reason_dict else None,
            context_summary_json=json.dumps(context_summary_dict) if context_summary_dict else None,
            status="pending",
            created_at=now,
            updated_at=now
        )
        self.session.add(alert)
        await self.session.flush()
        return alert

    async def create_event(
        self,
        event_id: str,
        session_id: str,
        user_id: str,
        risk_level: str,
        crisis_category: str,
        trigger_cues: List[str]
    ) -> EscalationEventModel:
        """Backwards-compatible wrapper around create_alert."""
        return await self.create_alert(
            alert_id=event_id,
            session_id=session_id,
            user_id=user_id,
            severity="critical" if risk_level in ("critical", "high") else "medium",
            trigger_type="crisis_risk",
            risk_level=risk_level,
            crisis_category=crisis_category,
            trigger_cues=trigger_cues
        )

    async def get_alert(self, alert_id: str) -> Optional[EscalationEventModel]:
        stmt = select(EscalationEventModel).where(EscalationEventModel.event_id == alert_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        clinician_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[EscalationEventModel]:
        stmt = select(EscalationEventModel).order_by(EscalationEventModel.created_at.desc())
        if status:
            stmt = stmt.where(EscalationEventModel.status == status)
        if severity:
            stmt = stmt.where(EscalationEventModel.severity == severity)
        if clinician_id:
            stmt = stmt.where(EscalationEventModel.assigned_clinician_id == clinician_id)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_events(self, status: Optional[str] = None, limit: int = 50) -> List[EscalationEventModel]:
        """Backwards-compatible wrapper."""
        return await self.list_alerts(status=status, limit=limit)

    async def acknowledge_alert(
        self,
        alert_id: str,
        clinician_id: str,
        notes: Optional[str] = None
    ) -> Optional[EscalationEventModel]:
        alert = await self.get_alert(alert_id)
        if alert:
            now = datetime.now(timezone.utc)
            alert.status = "acknowledged"
            alert.assigned_clinician_id = clinician_id
            alert.acknowledged_by = clinician_id
            alert.acknowledged_at = now
            alert.updated_at = now
            if notes:
                alert.notes = notes
            await self.session.flush()
        return alert

    async def acknowledge_event(
        self,
        event_id: str,
        clinician_id: str,
        notes: Optional[str] = None
    ) -> Optional[EscalationEventModel]:
        """Backwards-compatible wrapper."""
        return await self.acknowledge_alert(alert_id=event_id, clinician_id=clinician_id, notes=notes)

    async def update_status(
        self,
        alert_id: str,
        new_status: str,
        clinician_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[EscalationEventModel]:
        alert = await self.get_alert(alert_id)
        if alert:
            now = datetime.now(timezone.utc)
            alert.status = new_status
            alert.updated_at = now
            if clinician_id:
                alert.assigned_clinician_id = clinician_id
            if notes:
                alert.notes = notes
            await self.session.flush()
        return alert

    async def resolve_alert(
        self,
        alert_id: str,
        clinician_id: str,
        action_taken: str,
        resolution_notes: str
    ) -> Optional[EscalationEventModel]:
        alert = await self.get_alert(alert_id)
        if alert:
            now = datetime.now(timezone.utc)
            alert.status = "resolved"
            alert.resolved_by = clinician_id
            alert.resolved_at = now
            alert.action_taken = action_taken
            alert.resolution_notes = resolution_notes
            alert.updated_at = now
            await self.session.flush()
        return alert

    async def record_audit(
        self,
        audit_id: str,
        alert_id: str,
        action: str,
        actor_id: str,
        actor_role: str,
        previous_status: Optional[str],
        new_status: str,
        details_dict: Optional[Dict[str, Any]] = None
    ) -> EscalationAuditModel:
        audit_entry = EscalationAuditModel(
            audit_id=audit_id,
            alert_id=alert_id,
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            previous_status=previous_status,
            new_status=new_status,
            details_json=json.dumps(details_dict or {}),
            timestamp=datetime.now(timezone.utc)
        )
        self.session.add(audit_entry)
        await self.session.flush()
        return audit_entry

    async def get_audit_trail(self, alert_id: str) -> List[EscalationAuditModel]:
        stmt = (
            select(EscalationAuditModel)
            .where(EscalationAuditModel.alert_id == alert_id)
            .order_by(EscalationAuditModel.timestamp.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class BaselineRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_baseline(self, user_id: str) -> Optional[UserBaselineModel]:
        stmt = select(UserBaselineModel).where(UserBaselineModel.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def save_baseline(
        self,
        user_id: str,
        status: str,
        confidence: float,
        total_observations: int,
        profile_dict: Dict[str, Any]
    ) -> UserBaselineModel:
        record = await self.get_baseline(user_id)
        if not record:
            record = UserBaselineModel(
                user_id=user_id,
                status=status,
                confidence=confidence,
                total_observations=total_observations,
                baseline_profile_json=json.dumps(profile_dict)
            )
            self.session.add(record)
        else:
            record.status = status
            record.confidence = confidence
            record.total_observations = total_observations
            record.baseline_profile_json = json.dumps(profile_dict)
        await self.session.flush()
        return record

    async def record_observation(
        self,
        user_id: str,
        features_dict: Dict[str, Any],
        session_id: Optional[str] = None,
        turn_id: Optional[int] = None
    ) -> UserObservationModel:
        obs = UserObservationModel(
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            features_json=json.dumps(features_dict)
        )
        self.session.add(obs)
        await self.session.flush()
        return obs

    async def get_observations(self, user_id: str, limit: int = 100) -> List[UserObservationModel]:
        stmt = (
            select(UserObservationModel)
            .where(UserObservationModel.user_id == user_id)
            .order_by(UserObservationModel.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def reset_baseline(self, user_id: str) -> bool:
        stmt_del_obs = delete(UserObservationModel).where(UserObservationModel.user_id == user_id)
        stmt_del_base = delete(UserBaselineModel).where(UserBaselineModel.user_id == user_id)
        await self.session.execute(stmt_del_obs)
        await self.session.execute(stmt_del_base)
        await self.session.flush()
        return True


class BehaviorRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_observation(
        self,
        user_id: str,
        metrics_dict: Dict[str, Any],
        source_device: str = "smartphone_passive",
        timestamp: Optional[datetime] = None
    ) -> BehavioralObservationModel:
        obs = BehavioralObservationModel(
            user_id=user_id,
            source_device=source_device,
            metrics_json=json.dumps(metrics_dict),
            timestamp=timestamp or datetime.now(timezone.utc)
        )
        self.session.add(obs)
        await self.session.flush()
        return obs

    async def get_recent_observations(self, user_id: str, limit: int = 30) -> List[BehavioralObservationModel]:
        stmt = (
            select(BehavioralObservationModel)
            .where(BehavioralObservationModel.user_id == user_id)
            .order_by(BehavioralObservationModel.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def clear_observations(self, user_id: str) -> bool:
        stmt = delete(BehavioralObservationModel).where(BehavioralObservationModel.user_id == user_id)
        await self.session.execute(stmt)
        await self.session.flush()
        return True


class VoiceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_observation(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        duration_seconds: float = 0.0,
        transcription: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 0.0,
        features_dict: Optional[Dict[str, Any]] = None,
    ) -> VoiceObservationModel:
        obs = VoiceObservationModel(
            user_id=user_id,
            session_id=session_id,
            duration_seconds=duration_seconds,
            transcription=transcription,
            emotion=emotion,
            confidence=confidence,
            features_json=json.dumps(features_dict or {})
        )
        self.session.add(obs)
        await self.session.flush()
        return obs

    async def get_recent_observations(self, user_id: str, limit: int = 20) -> List[VoiceObservationModel]:
        stmt = (
            select(VoiceObservationModel)
            .where(VoiceObservationModel.user_id == user_id)
            .order_by(VoiceObservationModel.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def clear_observations(self, user_id: str) -> bool:
        stmt = delete(VoiceObservationModel).where(VoiceObservationModel.user_id == user_id)
        await self.session.execute(stmt)
        await self.session.flush()
        return True


class ContextRepository:
    """Persistence operations for Multimodal Context snapshots and turn feedback."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_snapshot(
        self,
        context_id: str,
        session_id: str,
        turn_id: int,
        context_hash: str,
        context_json: str,
        privacy_level: str = "STANDARD"
    ) -> ContextSnapshotModel:
        stmt = select(ContextSnapshotModel).where(ContextSnapshotModel.context_id == context_id)
        res = await self.session.execute(stmt)
        existing = res.scalars().first()
        if existing:
            existing.context_hash = context_hash
            existing.context_json = context_json
            existing.privacy_level = privacy_level
            existing.turn_id = turn_id
            await self.session.flush()
            return existing

        snapshot = ContextSnapshotModel(
            context_id=context_id,
            session_id=session_id,
            turn_id=turn_id,
            context_hash=context_hash,
            privacy_level=privacy_level,
            context_json=context_json,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(snapshot)
        await self.session.flush()
        return snapshot

    async def get_latest_snapshot(self, session_id: str) -> Optional[ContextSnapshotModel]:
        stmt = (
            select(ContextSnapshotModel)
            .where(ContextSnapshotModel.session_id == session_id)
            .order_by(ContextSnapshotModel.turn_id.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_snapshot_by_id(self, context_id: str) -> Optional[ContextSnapshotModel]:
        stmt = select(ContextSnapshotModel).where(ContextSnapshotModel.context_id == context_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def record_turn_feedback(
        self,
        session_id: str,
        turn_id: int,
        feedback_dict: Dict[str, Any]
    ) -> Optional[ConversationTurnModel]:
        stmt = (
            select(ConversationTurnModel)
            .where(ConversationTurnModel.session_id == session_id, ConversationTurnModel.turn_id == turn_id)
        )
        result = await self.session.execute(stmt)
        turn = result.scalars().first()
        if turn:
            turn.feedback_json = json.dumps(feedback_dict, default=str)
            await self.session.flush()
        return turn


class SafetyAuditRepository:
    """Persistence repository for privacy-sanitized safety audit logs."""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_audit(
        self,
        audit_id: str,
        session_id: str,
        model_version: str,
        action: str,
        is_safe: bool,
        violated_policies: List[str],
        reason_codes: List[str],
        turn_id: Optional[int] = None,
        risk_level: Optional[str] = None,
        latency_ms: float = 0.0
    ) -> SafetyAuditModel:
        entry = SafetyAuditModel(
            audit_id=audit_id,
            session_id=session_id,
            turn_id=turn_id,
            model_version=model_version,
            action=action,
            is_safe=1 if is_safe else 0,
            violated_policies_json=json.dumps(violated_policies),
            reason_codes_json=json.dumps(reason_codes),
            risk_level=risk_level,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_recent_audits(self, limit: int = 50) -> List[SafetyAuditModel]:
        stmt = (
            select(SafetyAuditModel)
            .order_by(SafetyAuditModel.created_at.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_audits_for_session(self, session_id: str) -> List[SafetyAuditModel]:
        stmt = (
            select(SafetyAuditModel)
            .where(SafetyAuditModel.session_id == session_id)
            .order_by(SafetyAuditModel.created_at.asc())
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())


class DashboardRepository:
    """Persistence and aggregation repository for the Clinician Dashboard."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_access(
        self,
        access_id: str,
        actor_id: str,
        actor_role: str,
        endpoint: str,
        action: str,
        target_user_id: Optional[str] = None,
        ip_address: str = "127.0.0.1"
    ) -> DashboardAccessAuditModel:
        entry = DashboardAccessAuditModel(
            access_id=access_id,
            actor_id=actor_id,
            actor_role=actor_role,
            target_user_id=target_user_id,
            endpoint=endpoint,
            action=action,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc)
        )
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_access_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        actor_id: Optional[str] = None,
        target_user_id: Optional[str] = None
    ) -> List[DashboardAccessAuditModel]:
        stmt = select(DashboardAccessAuditModel).order_by(DashboardAccessAuditModel.timestamp.desc())
        if actor_id:
            stmt = stmt.where(DashboardAccessAuditModel.actor_id == actor_id)
        if target_user_id:
            stmt = stmt.where(DashboardAccessAuditModel.target_user_id == target_user_id)
        stmt = stmt.offset(offset).limit(limit)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def get_overview_metrics(self) -> Dict[str, Any]:
        # 1. Active alerts
        stmt_alerts = select(EscalationEventModel).order_by(EscalationEventModel.created_at.desc())
        res_alerts = await self.session.execute(stmt_alerts)
        all_alerts = list(res_alerts.scalars().all())

        active_alerts = [a for a in all_alerts if a.status in ("pending", "acknowledged", "in_review")]
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for a in active_alerts:
            sev = (a.severity or "medium").lower()
            if sev in severity_counts:
                severity_counts[sev] += 1
            else:
                severity_counts["medium"] += 1

        # 2. Risk distribution across recent user turns
        stmt_turns = (
            select(ConversationTurnModel)
            .where(ConversationTurnModel.speaker == "user")
            .order_by(ConversationTurnModel.created_at.desc())
            .limit(200)
        )
        res_turns = await self.session.execute(stmt_turns)
        user_turns = list(res_turns.scalars().all())
        risk_dist = {"no_risk": 0, "low": 0, "moderate": 0, "high": 0, "critical": 0}
        for t in user_turns:
            if t.risk_json:
                try:
                    rdata = json.loads(t.risk_json)
                    rl = rdata.get("risk_level", "no_risk").lower()
                    if rl in risk_dist:
                        risk_dist[rl] += 1
                    else:
                        risk_dist["low"] += 1
                except Exception:
                    pass

        # 3. User counts
        stmt_sessions = select(UserSessionModel)
        res_sessions = await self.session.execute(stmt_sessions)
        sessions = list(res_sessions.scalars().all())
        distinct_users = set(s.user_id for s in sessions)

        return {
            "active_alerts_count": len(active_alerts),
            "alerts_by_severity": severity_counts,
            "risk_distribution": risk_dist,
            "recent_escalations": all_alerts[:5],
            "active_users_count": len(distinct_users),
            "total_sessions_count": len(sessions)
        }

    async def get_all_users_summary(self, limit: int = 50) -> List[Dict[str, Any]]:
        # Get all sessions
        stmt = select(UserSessionModel).order_by(UserSessionModel.updated_at.desc())
        res = await self.session.execute(stmt)
        sessions = list(res.scalars().all())

        user_sessions_map: Dict[str, List[UserSessionModel]] = {}
        for s in sessions:
            user_sessions_map.setdefault(s.user_id, []).append(s)

        summaries = []
        for uid, u_sessions in list(user_sessions_map.items())[:limit]:
            sess_ids = [s.session_id for s in u_sessions]
            stmt_t = (
                select(ConversationTurnModel)
                .where(ConversationTurnModel.session_id.in_(sess_ids))
                .order_by(ConversationTurnModel.created_at.desc())
            )
            res_t = await self.session.execute(stmt_t)
            all_turns = list(res_t.scalars().all())

            latest_risk = "no_risk"
            last_active = None
            if all_turns:
                last_active = all_turns[0].created_at
                for t in all_turns:
                    if t.speaker == "user" and t.risk_json:
                        try:
                            latest_risk = json.loads(t.risk_json).get("risk_level", "no_risk")
                            break
                        except Exception:
                            pass

            stmt_b = select(UserBaselineModel).where(UserBaselineModel.user_id == uid)
            res_b = await self.session.execute(stmt_b)
            base_rec = res_b.scalars().first()
            baseline_status = base_rec.status if base_rec else "insufficient_data"
            dev_score = 0.0
            if base_rec and base_rec.baseline_profile_json:
                try:
                    p = json.loads(base_rec.baseline_profile_json)
                    dev_score = float(p.get("anxiety_score", {}).get("z_score", 0.0) if isinstance(p.get("anxiety_score"), dict) else 0.0)
                except Exception:
                    pass

            stmt_esc = select(EscalationEventModel).where(
                EscalationEventModel.user_id == uid,
                EscalationEventModel.status.in_(("pending", "acknowledged", "in_review"))
            )
            res_esc = await self.session.execute(stmt_esc)
            open_alerts = len(list(res_esc.scalars().all()))

            summaries.append({
                "user_id": uid,
                "current_risk_level": latest_risk,
                "baseline_status": baseline_status,
                "baseline_deviation_score": round(dev_score, 2),
                "open_alerts_count": open_alerts,
                "total_turns": len(all_turns),
                "total_sessions": len(u_sessions),
                "last_active_at": last_active
            })

        return summaries

    async def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        users = await self.get_all_users_summary(limit=500)
        for u in users:
            if u["user_id"] == user_id:
                return u
        return None

    async def get_user_trends(self, user_id: str) -> Dict[str, Any]:
        stmt_s = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        res_s = await self.session.execute(stmt_s)
        sessions = list(res_s.scalars().all())
        sess_ids = [s.session_id for s in sessions]

        if not sess_ids:
            return {
                "emotion_trends": [],
                "symptom_trends": [],
                "risk_trends": [],
                "strategy_trends": [],
                "baseline_metrics": {}
            }

        stmt_t = (
            select(ConversationTurnModel)
            .where(ConversationTurnModel.session_id.in_(sess_ids))
            .order_by(ConversationTurnModel.created_at.asc())
        )
        res_t = await self.session.execute(stmt_t)
        turns = list(res_t.scalars().all())

        emotion_trends = []
        symptom_trends = []
        risk_trends = []
        strategy_trends = []

        for t in turns:
            iso_time = t.created_at.isoformat() if t.created_at else None
            # Emotion
            if t.emotion_json:
                try:
                    ed = json.loads(t.emotion_json)
                    emotion_trends.append({
                        "turn_id": t.turn_id,
                        "timestamp": iso_time,
                        "primary_emotion": ed.get("primary_emotion", "neutral"),
                        "valence": round(float(ed.get("valence", 0.0)), 3),
                        "arousal": round(float(ed.get("arousal", 0.0)), 3),
                        "confidence": round(float(ed.get("confidence", 1.0)), 3)
                    })
                except Exception:
                    pass

            # Symptom
            if t.symptom_json:
                try:
                    sd = json.loads(t.symptom_json)
                    signals = sd.get("signals", [])
                    detected = [s.get("category") for s in signals if s.get("category")]
                    highest_sev = "none"
                    for s in signals:
                        sev = s.get("severity", "none")
                        if sev == "severe":
                            highest_sev = "severe"
                            break
                        elif sev == "moderate" and highest_sev != "severe":
                            highest_sev = "moderate"
                        elif sev == "mild" and highest_sev not in ("severe", "moderate"):
                            highest_sev = "mild"
                    symptom_trends.append({
                        "turn_id": t.turn_id,
                        "timestamp": iso_time,
                        "detected_symptoms": detected,
                        "highest_severity": highest_sev
                    })
                except Exception:
                    pass

            # Risk
            if t.risk_json:
                try:
                    rd = json.loads(t.risk_json)
                    risk_trends.append({
                        "turn_id": t.turn_id,
                        "timestamp": iso_time,
                        "risk_level": rd.get("risk_level", "no_risk"),
                        "crisis_category": rd.get("crisis_category", "none"),
                        "confidence": round(float(rd.get("confidence", 1.0)), 3)
                    })
                except Exception:
                    pass

            # Strategy (assistant turns)
            if t.speaker == "assistant" and t.strategy_json:
                try:
                    std = json.loads(t.strategy_json)
                    strategy_trends.append({
                        "turn_id": t.turn_id,
                        "timestamp": iso_time,
                        "strategy": std.get("selected_strategy", "Others"),
                        "confidence": round(float(std.get("confidence", 1.0)), 3)
                    })
                except Exception:
                    pass

        stmt_b = select(UserBaselineModel).where(UserBaselineModel.user_id == user_id)
        res_b = await self.session.execute(stmt_b)
        base_rec = res_b.scalars().first()
        baseline_metrics = {}
        if base_rec and base_rec.baseline_profile_json:
            try:
                baseline_metrics = json.loads(base_rec.baseline_profile_json)
            except Exception:
                pass

        return {
            "emotion_trends": emotion_trends,
            "symptom_trends": symptom_trends,
            "risk_trends": risk_trends,
            "strategy_trends": strategy_trends,
            "baseline_metrics": baseline_metrics
        }

    async def get_user_timeline(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        events = []

        stmt_s = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        res_s = await self.session.execute(stmt_s)
        sessions = list(res_s.scalars().all())
        sess_ids = [s.session_id for s in sessions]

        if sess_ids:
            stmt_t = select(ConversationTurnModel).where(ConversationTurnModel.session_id.in_(sess_ids))
            res_t = await self.session.execute(stmt_t)
            for t in res_t.scalars().all():
                events.append({
                    "event_id": f"turn_{t.session_id}_{t.turn_id}",
                    "event_type": "conversation",
                    "timestamp": t.created_at or datetime.now(timezone.utc),
                    "title": f"Turn {t.turn_id} ({t.speaker.capitalize()})",
                    "description": t.content[:140] + ("..." if len(t.content) > 140 else ""),
                    "severity": "info",
                    "data": {
                        "session_id": t.session_id,
                        "turn_id": t.turn_id,
                        "speaker": t.speaker,
                        "has_risk": bool(t.risk_json and ('"risk_level": "high"' in t.risk_json or '"risk_level": "critical"' in t.risk_json))
                    }
                })

        stmt_beh = select(BehavioralObservationModel).where(BehavioralObservationModel.user_id == user_id)
        res_beh = await self.session.execute(stmt_beh)
        for b in res_beh.scalars().all():
            m = {}
            if b.metrics_json:
                try:
                    m = json.loads(b.metrics_json)
                except Exception:
                    pass
            events.append({
                "event_id": f"beh_{b.id}",
                "event_type": "behavior",
                "timestamp": b.timestamp or datetime.now(timezone.utc),
                "title": f"Passive Behavioral Observation ({b.source_device})",
                "description": f"Steps: {m.get('step_count', 'N/A')}, Screen: {m.get('screen_time_hours', 'N/A')}h, Sleep: {m.get('sleep_duration_hours', 'N/A')}h",
                "severity": "info",
                "data": m
            })

        stmt_v = select(VoiceObservationModel).where(VoiceObservationModel.user_id == user_id)
        res_v = await self.session.execute(stmt_v)
        for v in res_v.scalars().all():
            events.append({
                "event_id": f"voice_{v.id}",
                "event_type": "voice",
                "timestamp": v.created_at or datetime.now(timezone.utc),
                "title": f"Voice Acoustic Sample ({v.duration_seconds:.1f}s)",
                "description": f"Acoustic Emotion: {v.emotion or 'unspecified'} (confidence: {v.confidence:.2f})",
                "severity": "info",
                "data": {
                    "duration_seconds": v.duration_seconds,
                    "emotion": v.emotion,
                    "confidence": v.confidence,
                    "transcription": v.transcription
                }
            })

        stmt_esc = select(EscalationEventModel).where(EscalationEventModel.user_id == user_id)
        res_esc = await self.session.execute(stmt_esc)
        for e in res_esc.scalars().all():
            events.append({
                "event_id": f"esc_{e.event_id}",
                "event_type": "escalation",
                "timestamp": e.created_at or datetime.now(timezone.utc),
                "title": f"Escalation Alert ({e.severity.upper()}): {e.trigger_type}",
                "description": f"Status: {e.status.upper()}. Risk: {e.risk_level}, Category: {e.crisis_category}",
                "severity": e.severity or "high",
                "data": {
                    "alert_id": e.event_id,
                    "status": e.status,
                    "severity": e.severity,
                    "trigger_type": e.trigger_type,
                    "action_taken": e.action_taken
                }
            })

        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return events[:limit]

    async def get_user_conversations(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        stmt_s = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        res_s = await self.session.execute(stmt_s)
        sessions = list(res_s.scalars().all())
        sess_ids = [s.session_id for s in sessions]

        if not sess_ids:
            return []

        stmt_t = (
            select(ConversationTurnModel)
            .where(ConversationTurnModel.session_id.in_(sess_ids))
            .order_by(ConversationTurnModel.created_at.asc())
            .limit(limit)
        )
        res_t = await self.session.execute(stmt_t)
        turns = list(res_t.scalars().all())

        results = []
        for t in turns:
            results.append({
                "turn_id": t.turn_id,
                "session_id": t.session_id,
                "speaker": t.speaker,
                "content": t.content,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "emotion": json.loads(t.emotion_json) if t.emotion_json else None,
                "symptoms": json.loads(t.symptom_json) if t.symptom_json else None,
                "risk": json.loads(t.risk_json) if t.risk_json else None,
                "strategy": json.loads(t.strategy_json) if t.strategy_json else None,
                "safety": json.loads(t.safety_json) if t.safety_json else None,
                "voice": json.loads(t.voice_json) if t.voice_json else None,
                "behavior": json.loads(t.behavior_json) if t.behavior_json else None,
            })
        return results


class TraceRepository:
    """Repository for recording and querying end-to-end execution traces."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_trace(self, trace: PipelineTrace) -> ExecutionTraceModel:
        """Persist a complete execution trace with spans and degraded modules."""
        record = ExecutionTraceModel(
            trace_id=trace.trace_id,
            session_id=trace.session_id,
            turn_id=trace.turn_id,
            user_id=trace.user_id,
            status=trace.status.value if hasattr(trace.status, "value") else str(trace.status),
            latency_ms=trace.total_duration_ms,
            degraded_modules_json=json.dumps(trace.degraded_modules),
            spans_json=json.dumps([s.model_dump() for s in trace.spans]),
            metadata_json=json.dumps(trace.metadata) if trace.metadata else "{}",
            created_at=datetime.fromisoformat(trace.created_at) if isinstance(trace.created_at, str) and trace.created_at else datetime.now(timezone.utc)
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        """Fetch a single execution trace by trace_id."""
        stmt = select(ExecutionTraceModel).where(ExecutionTraceModel.trace_id == trace_id)
        res = await self.session.execute(stmt)
        record = res.scalars().first()
        if not record:
            return None
        return self._to_schema(record)

    async def list_traces(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[PipelineTrace]:
        """Query execution traces filtered by session, user, or status."""
        stmt = select(ExecutionTraceModel)
        if session_id:
            stmt = stmt.where(ExecutionTraceModel.session_id == session_id)
        if user_id:
            stmt = stmt.where(ExecutionTraceModel.user_id == user_id)
        if status:
            stmt = stmt.where(ExecutionTraceModel.status == status)
        stmt = stmt.order_by(ExecutionTraceModel.created_at.desc()).limit(limit)
        res = await self.session.execute(stmt)
        records = res.scalars().all()
        return [self._to_schema(r) for r in records]

    def _to_schema(self, record: ExecutionTraceModel) -> PipelineTrace:
        spans_raw = json.loads(record.spans_json) if record.spans_json else []
        spans = [TraceSpan(**s) for s in spans_raw]
        degraded = json.loads(record.degraded_modules_json) if record.degraded_modules_json else []
        meta = json.loads(record.metadata_json) if getattr(record, "metadata_json", None) else {}
        return PipelineTrace(
            trace_id=record.trace_id,
            session_id=record.session_id,
            turn_id=record.turn_id,
            user_id=record.user_id,
            status=PipelineStatus(record.status),
            total_duration_ms=record.latency_ms,
            spans=spans,
            degraded_modules=degraded,
            created_at=record.created_at.isoformat() if record.created_at else datetime.now(timezone.utc).isoformat(),
            metadata=meta
        )


class UserRepository:
    """Repository managing user preferences, wellbeing check-ins, data export, and purge."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create_preferences(self, user_id: str) -> UserPreferencesModel:
        """Fetch or initialize default preferences for a user."""
        stmt = select(UserPreferencesModel).where(UserPreferencesModel.user_id == user_id)
        res = await self.session.execute(stmt)
        prefs = res.scalars().first()
        if not prefs:
            prefs = UserPreferencesModel(
                user_id=user_id,
                save_history=True,
                enable_voice=True,
                enable_wearables=False,
                privacy_level="standard",
                preferred_language="en",
                communication_style="warm_empathic",
                allow_clinician_sharing=False,
            )
            self.session.add(prefs)
            await self.session.flush()
        return prefs

    async def update_preferences(self, user_id: str, updates: Dict[str, Any]) -> UserPreferencesModel:
        """Update specific user preferences."""
        prefs = await self.get_or_create_preferences(user_id)
        for key, val in updates.items():
            if val is not None and hasattr(prefs, key):
                setattr(prefs, key, val)
        prefs.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return prefs

    async def record_checkin(
        self,
        checkin_id: str,
        user_id: str,
        mood_score: int,
        valence: float = 0.0,
        sleep_hours: float = 7.0,
        stress_level: int = 3,
        energy_level: int = 3,
        notes: Optional[str] = None
    ) -> UserCheckinModel:
        """Record a structured wellbeing check-in."""
        checkin = UserCheckinModel(
            checkin_id=checkin_id,
            user_id=user_id,
            mood_score=mood_score,
            valence=valence,
            sleep_hours=sleep_hours,
            stress_level=stress_level,
            energy_level=energy_level,
            notes=notes
        )
        self.session.add(checkin)
        await self.session.flush()
        return checkin

    async def get_checkin_history(self, user_id: str, limit: int = 30) -> List[UserCheckinModel]:
        """Fetch recent check-in history sorted newest first."""
        stmt = (
            select(UserCheckinModel)
            .where(UserCheckinModel.user_id == user_id)
            .order_by(UserCheckinModel.created_at.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def list_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List past conversation sessions with turn counts and first-message preview."""
        stmt = (
            select(UserSessionModel)
            .where(UserSessionModel.user_id == user_id)
            .order_by(UserSessionModel.updated_at.desc())
            .limit(limit)
        )
        res = await self.session.execute(stmt)
        sessions = res.scalars().all()
        result = []
        for s in sessions:
            turn_stmt = (
                select(ConversationTurnModel)
                .where(ConversationTurnModel.session_id == s.session_id)
                .order_by(ConversationTurnModel.turn_id.asc())
            )
            t_res = await self.session.execute(turn_stmt)
            turns = t_res.scalars().all()
            preview = turns[0].content[:60] if turns else "New conversation"
            result.append({
                "session_id": s.session_id,
                "user_id": s.user_id,
                "turn_count": len(turns),
                "first_message_preview": preview,
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "updated_at": s.updated_at.isoformat() if s.updated_at else "",
            })
        return result

    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Compile a portable GDPR/HIPAA JSON export package of all user data."""
        prefs = await self.get_or_create_preferences(user_id)
        checkins = await self.get_checkin_history(user_id, limit=500)

        # Retrieve sessions & turns
        stmt = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        s_res = await self.session.execute(stmt)
        sessions = s_res.scalars().all()
        session_data = []
        for s in sessions:
            t_stmt = (
                select(ConversationTurnModel)
                .where(ConversationTurnModel.session_id == s.session_id)
                .order_by(ConversationTurnModel.turn_id.asc())
            )
            t_res = await self.session.execute(t_stmt)
            turns = t_res.scalars().all()
            session_data.append({
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "speaker": t.speaker,
                        "content": t.content,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in turns
                ]
            })

        # Observations count
        obs_stmt = select(UserObservationModel).where(UserObservationModel.user_id == user_id)
        obs_res = await self.session.execute(obs_stmt)
        obs_count = len(obs_res.scalars().all())

        return {
            "export_id": f"export_{user_id}_{int(datetime.now(timezone.utc).timestamp())}",
            "user_id": user_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "preferences": {
                "save_history": prefs.save_history,
                "enable_voice": prefs.enable_voice,
                "enable_wearables": prefs.enable_wearables,
                "privacy_level": prefs.privacy_level,
                "preferred_language": prefs.preferred_language,
                "communication_style": prefs.communication_style,
                "allow_clinician_sharing": prefs.allow_clinician_sharing,
                "updated_at": prefs.updated_at.isoformat() if prefs.updated_at else None
            },
            "checkins": [
                {
                    "checkin_id": c.checkin_id,
                    "mood_score": c.mood_score,
                    "valence": c.valence,
                    "sleep_hours": c.sleep_hours,
                    "stress_level": c.stress_level,
                    "energy_level": c.energy_level,
                    "notes": c.notes,
                    "created_at": c.created_at.isoformat() if c.created_at else None
                }
                for c in checkins
            ],
            "sessions_count": len(session_data),
            "sessions": session_data,
            "observations_count": obs_count
        }

    async def purge_user_data(self, user_id: str) -> Dict[str, int]:
        """Permanently delete all personal user records under Right to be Forgotten."""
        # 1. User sessions & turns
        s_stmt = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        s_res = await self.session.execute(s_stmt)
        sessions = s_res.scalars().all()
        session_ids = [s.session_id for s in sessions]

        del_turns = 0
        if session_ids:
            t_stmt = delete(ConversationTurnModel).where(ConversationTurnModel.session_id.in_(session_ids))
            t_res = await self.session.execute(t_stmt)
            del_turns = t_res.rowcount if hasattr(t_res, "rowcount") else 0

        del_sessions = 0
        if session_ids:
            del_s_stmt = delete(UserSessionModel).where(UserSessionModel.user_id == user_id)
            del_s_res = await self.session.execute(del_s_stmt)
            del_sessions = del_s_res.rowcount if hasattr(del_s_res, "rowcount") else 0

        # 2. Check-ins
        c_stmt = delete(UserCheckinModel).where(UserCheckinModel.user_id == user_id)
        c_res = await self.session.execute(c_stmt)
        del_checkins = c_res.rowcount if hasattr(c_res, "rowcount") else 0

        # 3. Observations and baselines
        obs_stmt = delete(UserObservationModel).where(UserObservationModel.user_id == user_id)
        obs_res = await self.session.execute(obs_stmt)
        del_obs = obs_res.rowcount if hasattr(obs_res, "rowcount") else 0

        base_stmt = delete(UserBaselineModel).where(UserBaselineModel.user_id == user_id)
        await self.session.execute(base_stmt)

        beh_stmt = delete(BehavioralObservationModel).where(BehavioralObservationModel.user_id == user_id)
        await self.session.execute(beh_stmt)

        v_stmt = delete(VoiceObservationModel).where(VoiceObservationModel.user_id == user_id)
        await self.session.execute(v_stmt)

        await self.session.flush()

        return {
            "deleted_sessions": del_sessions if del_sessions is not None else len(session_ids),
            "deleted_turns": del_turns if del_turns is not None else 0,
            "deleted_checkins": del_checkins if del_checkins is not None else 0,
            "deleted_observations": del_obs if del_obs is not None else 0,
        }








