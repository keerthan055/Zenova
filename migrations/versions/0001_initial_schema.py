"""Initial ZENOVA schema creation across all relational tables.

Revision ID: 0001
Revises: 
Create Date: 2026-09-14 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    # 1. user_sessions
    if 'user_sessions' not in existing_tables:
        op.create_table(
            'user_sessions',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_user_sessions_session_id'), 'user_sessions', ['session_id'], unique=True)
        op.create_index(op.f('ix_user_sessions_user_id'), 'user_sessions', ['user_id'], unique=False)

    # 2. conversation_turns
    if 'conversation_turns' not in existing_tables:
        op.create_table(
            'conversation_turns',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('turn_id', sa.Integer(), nullable=False),
            sa.Column('speaker', sa.String(length=32), nullable=False),
            sa.Column('content', sa.Text(), nullable=False),
            sa.Column('emotion_json', sa.Text(), nullable=True),
            sa.Column('symptom_json', sa.Text(), nullable=True),
            sa.Column('risk_json', sa.Text(), nullable=True),
            sa.Column('behavior_json', sa.Text(), nullable=True),
            sa.Column('voice_json', sa.Text(), nullable=True),
            sa.Column('strategy_json', sa.Text(), nullable=True),
            sa.Column('safety_json', sa.Text(), nullable=True),
            sa.Column('context_json', sa.Text(), nullable=True),
            sa.Column('feedback_json', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['session_id'], ['user_sessions.session_id'])
        )
        op.create_index(op.f('ix_conversation_turns_session_id'), 'conversation_turns', ['session_id'], unique=False)

    # 3. escalation_events
    if 'escalation_events' not in existing_tables:
        op.create_table(
            'escalation_events',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('event_id', sa.String(length=64), nullable=False),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('severity', sa.String(length=32), nullable=False),
            sa.Column('trigger_type', sa.String(length=64), nullable=False),
            sa.Column('risk_level', sa.String(length=32), nullable=False),
            sa.Column('crisis_category', sa.String(length=64), nullable=False),
            sa.Column('trigger_cues_json', sa.Text(), nullable=True),
            sa.Column('reason_json', sa.Text(), nullable=True),
            sa.Column('context_summary_json', sa.Text(), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False),
            sa.Column('assigned_clinician_id', sa.String(length=64), nullable=True),
            sa.Column('acknowledged_by', sa.String(length=64), nullable=True),
            sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
            sa.Column('action_taken', sa.String(length=64), nullable=True),
            sa.Column('resolution_notes', sa.Text(), nullable=True),
            sa.Column('resolved_by', sa.String(length=64), nullable=True),
            sa.Column('resolved_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(['session_id'], ['user_sessions.session_id'])
        )
        op.create_index(op.f('ix_escalation_events_event_id'), 'escalation_events', ['event_id'], unique=True)
        op.create_index(op.f('ix_escalation_events_session_id'), 'escalation_events', ['session_id'], unique=False)
        op.create_index(op.f('ix_escalation_events_status'), 'escalation_events', ['status'], unique=False)

    # 4. escalation_audit_logs
    if 'escalation_audit_logs' not in existing_tables:
        op.create_table(
            'escalation_audit_logs',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('audit_id', sa.String(length=64), nullable=False),
            sa.Column('alert_id', sa.String(length=64), nullable=False),
            sa.Column('action', sa.String(length=64), nullable=False),
            sa.Column('actor_id', sa.String(length=64), nullable=False),
            sa.Column('actor_role', sa.String(length=32), nullable=False),
            sa.Column('previous_status', sa.String(length=32), nullable=True),
            sa.Column('new_status', sa.String(length=32), nullable=False),
            sa.Column('details_json', sa.Text(), nullable=False),
            sa.Column('timestamp', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['alert_id'], ['escalation_events.event_id'])
        )
        op.create_index(op.f('ix_escalation_audit_logs_audit_id'), 'escalation_audit_logs', ['audit_id'], unique=True)
        op.create_index(op.f('ix_escalation_audit_logs_alert_id'), 'escalation_audit_logs', ['alert_id'], unique=False)

    # 5. user_baselines
    if 'user_baselines' not in existing_tables:
        op.create_table(
            'user_baselines',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('status', sa.String(length=32), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=False),
            sa.Column('total_observations', sa.Integer(), nullable=False),
            sa.Column('baseline_profile_json', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_user_baselines_user_id'), 'user_baselines', ['user_id'], unique=True)

    # 6. context_snapshots
    if 'context_snapshots' not in existing_tables:
        op.create_table(
            'context_snapshots',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('context_id', sa.String(length=64), nullable=False),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('turn_id', sa.Integer(), nullable=False),
            sa.Column('context_hash', sa.String(length=64), nullable=False),
            sa.Column('privacy_level', sa.String(length=32), nullable=False),
            sa.Column('context_json', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_context_snapshots_context_id'), 'context_snapshots', ['context_id'], unique=True)
        op.create_index(op.f('ix_context_snapshots_session_id'), 'context_snapshots', ['session_id'], unique=False)
        op.create_index(op.f('ix_context_snapshots_context_hash'), 'context_snapshots', ['context_hash'], unique=False)

    # 7. safety_audit_logs
    if 'safety_audit_logs' not in existing_tables:
        op.create_table(
            'safety_audit_logs',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('audit_id', sa.String(length=64), nullable=False),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('turn_id', sa.Integer(), nullable=True),
            sa.Column('model_version', sa.String(length=64), nullable=False),
            sa.Column('action', sa.String(length=32), nullable=False),
            sa.Column('is_safe', sa.Integer(), nullable=False),
            sa.Column('violated_policies_json', sa.Text(), nullable=False),
            sa.Column('reason_codes_json', sa.Text(), nullable=False),
            sa.Column('risk_level', sa.String(length=32), nullable=True),
            sa.Column('latency_ms', sa.Float(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_safety_audit_logs_audit_id'), 'safety_audit_logs', ['audit_id'], unique=True)
        op.create_index(op.f('ix_safety_audit_logs_session_id'), 'safety_audit_logs', ['session_id'], unique=False)

    # 8. execution_traces
    if 'execution_traces' not in existing_tables:
        op.create_table(
            'execution_traces',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
            sa.Column('trace_id', sa.String(length=64), nullable=False),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('turn_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('status', sa.String(length=32), nullable=False),
            sa.Column('latency_ms', sa.Float(), nullable=False),
            sa.Column('degraded_modules_json', sa.Text(), nullable=False),
            sa.Column('spans_json', sa.Text(), nullable=False),
            sa.Column('metadata_json', sa.Text(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_execution_traces_trace_id'), 'execution_traces', ['trace_id'], unique=True)
        op.create_index(op.f('ix_execution_traces_session_id'), 'execution_traces', ['session_id'], unique=False)
        op.create_index(op.f('ix_execution_traces_user_id'), 'execution_traces', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('execution_traces')
    op.drop_table('safety_audit_logs')
    op.drop_table('context_snapshots')
    op.drop_table('user_baselines')
    op.drop_table('escalation_audit_logs')
    op.drop_table('escalation_events')
    op.drop_table('conversation_turns')
    op.drop_table('user_sessions')
