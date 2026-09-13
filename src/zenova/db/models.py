"""SQLAlchemy database models for ZENOVA."""
import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from zenova.db.session import Base


class UserSessionModel(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    turns = relationship("ConversationTurnModel", back_populates="session", cascade="all, delete-orphan")
    escalations = relationship("EscalationEventModel", back_populates="session", cascade="all, delete-orphan")


class ConversationTurnModel(Base):
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("user_sessions.session_id"), index=True, nullable=False)
    turn_id = Column(Integer, nullable=False)
    speaker = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)

    # JSON representations of module outputs
    emotion_json = Column(Text, nullable=True)
    symptom_json = Column(Text, nullable=True)
    risk_json = Column(Text, nullable=True)
    behavior_json = Column(Text, nullable=True)
    voice_json = Column(Text, nullable=True)
    strategy_json = Column(Text, nullable=True)
    safety_json = Column(Text, nullable=True)
    context_json = Column(Text, nullable=True)
    feedback_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("UserSessionModel", back_populates="turns")


class EscalationEventModel(Base):
    __tablename__ = "escalation_events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)  # Acts as alert_id
    session_id = Column(String(64), ForeignKey("user_sessions.session_id"), index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    severity = Column(String(32), default="critical", nullable=False)
    trigger_type = Column(String(64), default="crisis_risk", nullable=False)
    risk_level = Column(String(32), nullable=False)
    crisis_category = Column(String(64), nullable=False)
    trigger_cues_json = Column(Text, default="[]")
    reason_json = Column(Text, nullable=True)
    context_summary_json = Column(Text, nullable=True)
    status = Column(String(32), default="pending", index=True)
    assigned_clinician_id = Column(String(64), nullable=True)
    acknowledged_by = Column(String(64), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    action_taken = Column(String(64), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(String(64), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    notes = Column(Text, nullable=True)

    session = relationship("UserSessionModel", back_populates="escalations")
    audits = relationship("EscalationAuditModel", back_populates="event", cascade="all, delete-orphan")


class EscalationAuditModel(Base):
    __tablename__ = "escalation_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(64), unique=True, index=True, nullable=False)
    alert_id = Column(String(64), ForeignKey("escalation_events.event_id"), index=True, nullable=False)
    action = Column(String(64), nullable=False)
    actor_id = Column(String(64), nullable=False)
    actor_role = Column(String(32), nullable=False)
    previous_status = Column(String(32), nullable=True)
    new_status = Column(String(32), nullable=False)
    details_json = Column(Text, default="{}", nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    event = relationship("EscalationEventModel", back_populates="audits")


class UserBaselineModel(Base):
    __tablename__ = "user_baselines"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(32), default="insufficient_data", nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    total_observations = Column(Integer, default=0, nullable=False)
    baseline_profile_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class UserObservationModel(Base):
    __tablename__ = "user_observations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), index=True, nullable=False)
    session_id = Column(String(64), nullable=True)
    turn_id = Column(Integer, nullable=True)
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class BehavioralObservationModel(Base):
    __tablename__ = "behavioral_observations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    source_device = Column(String(64), default="smartphone_passive", nullable=False)
    metrics_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class VoiceObservationModel(Base):
    __tablename__ = "voice_observations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    transcription = Column(Text, nullable=True)
    emotion = Column(String(32), nullable=True)
    confidence = Column(Float, default=0.0)
    features_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ContextSnapshotModel(Base):
    __tablename__ = "context_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    context_id = Column(String(64), unique=True, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    turn_id = Column(Integer, nullable=False)
    context_hash = Column(String(64), index=True, nullable=False)
    privacy_level = Column(String(32), default="STANDARD", nullable=False)
    context_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SafetyAuditModel(Base):
    __tablename__ = "safety_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(64), unique=True, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    turn_id = Column(Integer, nullable=True)
    model_version = Column(String(64), nullable=False)
    action = Column(String(32), nullable=False)
    is_safe = Column(Integer, nullable=False)
    violated_policies_json = Column(Text, default="[]", nullable=False)
    reason_codes_json = Column(Text, default="[]", nullable=False)
    risk_level = Column(String(32), nullable=True)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DashboardAccessAuditModel(Base):
    __tablename__ = "dashboard_access_logs"

    id = Column(Integer, primary_key=True, index=True)
    access_id = Column(String(64), unique=True, index=True, nullable=False)
    actor_id = Column(String(64), index=True, nullable=False)
    actor_role = Column(String(32), index=True, nullable=False)
    target_user_id = Column(String(64), index=True, nullable=True)
    endpoint = Column(String(128), nullable=False)
    action = Column(String(64), nullable=False)
    ip_address = Column(String(64), default="127.0.0.1", nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class ExecutionTraceModel(Base):
    __tablename__ = "execution_traces"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(64), unique=True, index=True, nullable=False)
    session_id = Column(String(64), index=True, nullable=False)
    turn_id = Column(Integer, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    status = Column(String(32), default="nominal", nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)
    degraded_modules_json = Column(Text, default="[]", nullable=False)
    spans_json = Column(Text, default="[]", nullable=False)
    metadata_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserPreferencesModel(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    save_history = Column(Boolean, default=True, nullable=False)
    enable_voice = Column(Boolean, default=True, nullable=False)
    enable_wearables = Column(Boolean, default=False, nullable=False)
    privacy_level = Column(String(32), default="standard", nullable=False)
    preferred_language = Column(String(16), default="en", nullable=False)
    communication_style = Column(String(32), default="warm_empathic", nullable=False)
    allow_clinician_sharing = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class UserCheckinModel(Base):
    __tablename__ = "user_checkins"

    id = Column(Integer, primary_key=True, index=True)
    checkin_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), index=True, nullable=False)
    mood_score = Column(Integer, nullable=False)
    valence = Column(Float, default=0.0, nullable=False)
    sleep_hours = Column(Float, default=7.0, nullable=False)
    stress_level = Column(Integer, default=3, nullable=False)
    energy_level = Column(Integer, default=3, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)








