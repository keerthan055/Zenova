"""Add authentication and user account management tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    # 1. users
    if 'users' not in existing_tables:
        op.create_table(
            'users',
            sa.Column('id', sa.String(length=64), primary_key=True, nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('display_name', sa.String(length=128), nullable=True),
            sa.Column('password_hash', sa.String(length=255), nullable=False),
            sa.Column('role', sa.String(length=32), nullable=False, server_default='user'),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('is_verified', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.Column('last_login_at', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=True)
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
        op.create_index(op.f('ix_users_role'), 'users', ['role'], unique=False)

    # 2. auth_sessions
    if 'auth_sessions' not in existing_tables:
        op.create_table(
            'auth_sessions',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('token_hash', sa.String(length=64), nullable=False),
            sa.Column('ip_address', sa.String(length=64), nullable=False, server_default='127.0.0.1'),
            sa.Column('user_agent', sa.String(length=255), nullable=True),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
        )
        op.create_index(op.f('ix_auth_sessions_session_id'), 'auth_sessions', ['session_id'], unique=True)
        op.create_index(op.f('ix_auth_sessions_user_id'), 'auth_sessions', ['user_id'], unique=False)
        op.create_index(op.f('ix_auth_sessions_token_hash'), 'auth_sessions', ['token_hash'], unique=False)

    # 3. password_reset_tokens
    if 'password_reset_tokens' not in existing_tables:
        op.create_table(
            'password_reset_tokens',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('token_hash', sa.String(length=64), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('used_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
        )
        op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
        op.create_index(op.f('ix_password_reset_tokens_token_hash'), 'password_reset_tokens', ['token_hash'], unique=False)

    # 4. email_verification_tokens
    if 'email_verification_tokens' not in existing_tables:
        op.create_table(
            'email_verification_tokens',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
            sa.Column('user_id', sa.String(length=64), nullable=False),
            sa.Column('token_hash', sa.String(length=64), nullable=False),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('used_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
        )
        op.create_index(op.f('ix_email_verification_tokens_user_id'), 'email_verification_tokens', ['user_id'], unique=False)
        op.create_index(op.f('ix_email_verification_tokens_token_hash'), 'email_verification_tokens', ['token_hash'], unique=False)

    # 5. auth_audit_logs
    if 'auth_audit_logs' not in existing_tables:
        op.create_table(
            'auth_audit_logs',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False, autoincrement=True),
            sa.Column('audit_id', sa.String(length=64), nullable=False),
            sa.Column('event_type', sa.String(length=64), nullable=False),
            sa.Column('user_id', sa.String(length=64), nullable=True),
            sa.Column('email', sa.String(length=255), nullable=True),
            sa.Column('ip_address', sa.String(length=64), nullable=False, server_default='127.0.0.1'),
            sa.Column('user_agent', sa.String(length=255), nullable=True),
            sa.Column('success', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('metadata_json', sa.Text(), nullable=False, server_default='{}'),
            sa.Column('timestamp', sa.DateTime(), nullable=True)
        )
        op.create_index(op.f('ix_auth_audit_logs_audit_id'), 'auth_audit_logs', ['audit_id'], unique=True)
        op.create_index(op.f('ix_auth_audit_logs_event_type'), 'auth_audit_logs', ['event_type'], unique=False)
        op.create_index(op.f('ix_auth_audit_logs_user_id'), 'auth_audit_logs', ['user_id'], unique=False)
        op.create_index(op.f('ix_auth_audit_logs_email'), 'auth_audit_logs', ['email'], unique=False)
        op.create_index(op.f('ix_auth_audit_logs_timestamp'), 'auth_audit_logs', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_table('auth_audit_logs')
    op.drop_table('email_verification_tokens')
    op.drop_table('password_reset_tokens')
    op.drop_table('auth_sessions')
    op.drop_table('users')
