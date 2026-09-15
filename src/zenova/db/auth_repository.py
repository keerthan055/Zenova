"""SQLAlchemy repository for user accounts, credentials, sessions, and security audits."""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from zenova.db.models import (
    UserModel,
    UserAuthSessionModel,
    PasswordResetTokenModel,
    EmailVerificationTokenModel,
    AuthAuditLogModel
)
from zenova.core.logging import get_logger

logger = get_logger("zenova.db.auth_repository")


class AuthRepository:
    """Async database repository managing authentication state and audit trails."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ==========================================================================
    # User Account Operations
    # ==========================================================================

    async def create_user(
        self,
        user_id: str,
        email: str,
        password_hash: str,
        display_name: Optional[str] = None,
        role: str = "user",
        is_verified: bool = False,
        is_active: bool = True
    ) -> UserModel:
        """Persist a new user account."""
        user = UserModel(
            id=user_id,
            email=email.strip().lower(),
            password_hash=password_hash,
            display_name=display_name,
            role=role,
            is_verified=is_verified,
            is_active=is_active,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch user by internal UUID primary key."""
        stmt = select(UserModel).where(UserModel.id == user_id)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_user_by_email(self, email: str) -> Optional[UserModel]:
        """Fetch user by normalized unique email."""
        stmt = select(UserModel).where(UserModel.email == email.strip().lower())
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserModel]:
        """Apply updates to an existing user record."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        for key, val in updates.items():
            if hasattr(user, key):
                setattr(user, key, val)
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return user

    async def update_last_login(self, user_id: str) -> None:
        """Touch last_login_at timestamp."""
        stmt = (
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(last_login_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    # ==========================================================================
    # Session & Refresh Token Operations
    # ==========================================================================

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> UserAuthSessionModel:
        """Create a new refresh session."""
        session_model = UserAuthSessionModel(
            session_id=session_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            is_revoked=False,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(session_model)
        await self.session.flush()
        return session_model

    async def get_session_by_token_hash(self, token_hash: str) -> Optional[UserAuthSessionModel]:
        """Lookup active, non-revoked session by refresh token hash."""
        now = datetime.now(timezone.utc)
        stmt = (
            select(UserAuthSessionModel)
            .where(
                UserAuthSessionModel.token_hash == token_hash,
                UserAuthSessionModel.is_revoked == False,
                UserAuthSessionModel.expires_at > now
            )
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def revoke_session_by_token_hash(self, token_hash: str) -> None:
        """Revoke a specific session."""
        stmt = (
            update(UserAuthSessionModel)
            .where(UserAuthSessionModel.token_hash == token_hash)
            .values(is_revoked=True)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all active sessions for a user (e.g. upon password reset)."""
        stmt = (
            update(UserAuthSessionModel)
            .where(UserAuthSessionModel.user_id == user_id, UserAuthSessionModel.is_revoked == False)
            .values(is_revoked=True)
        )
        res = await self.session.execute(stmt)
        await self.session.flush()
        return res.rowcount

    # ==========================================================================
    # Password Reset Tokens
    # ==========================================================================

    async def create_password_reset_token(
        self,
        user_id: str,
        token_hash: str,
        expires_at: datetime
    ) -> PasswordResetTokenModel:
        """Record a single-use password reset token hash."""
        reset_model = PasswordResetTokenModel(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(reset_model)
        await self.session.flush()
        return reset_model

    async def get_valid_password_reset_token(self, token_hash: str) -> Optional[PasswordResetTokenModel]:
        """Fetch active, unused, unexpired reset token."""
        now = datetime.now(timezone.utc)
        stmt = (
            select(PasswordResetTokenModel)
            .where(
                PasswordResetTokenModel.token_hash == token_hash,
                PasswordResetTokenModel.used_at == None,
                PasswordResetTokenModel.expires_at > now
            )
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def mark_password_reset_token_used(self, token_id: int) -> None:
        """Mark reset token as consumed."""
        stmt = (
            update(PasswordResetTokenModel)
            .where(PasswordResetTokenModel.id == token_id)
            .values(used_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    # ==========================================================================
    # Email Verification Tokens
    # ==========================================================================

    async def create_email_verification_token(
        self,
        user_id: str,
        token_hash: str,
        expires_at: datetime
    ) -> EmailVerificationTokenModel:
        """Record a single-use email verification token hash."""
        verify_model = EmailVerificationTokenModel(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            created_at=datetime.now(timezone.utc)
        )
        self.session.add(verify_model)
        await self.session.flush()
        return verify_model

    async def get_valid_email_verification_token(self, token_hash: str) -> Optional[EmailVerificationTokenModel]:
        """Fetch active, unused, unexpired verification token."""
        now = datetime.now(timezone.utc)
        stmt = (
            select(EmailVerificationTokenModel)
            .where(
                EmailVerificationTokenModel.token_hash == token_hash,
                EmailVerificationTokenModel.used_at == None,
                EmailVerificationTokenModel.expires_at > now
            )
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def mark_email_verification_token_used(self, token_id: int) -> None:
        """Mark verification token as consumed."""
        stmt = (
            update(EmailVerificationTokenModel)
            .where(EmailVerificationTokenModel.id == token_id)
            .values(used_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    # ==========================================================================
    # Security Audit Logging
    # ==========================================================================

    async def record_audit_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuthAuditLogModel:
        """Record an immutable security event."""
        # Sanitize metadata: strictly exclude password or raw token fields
        clean_meta = dict(metadata or {})
        for forbidden in ["password", "token", "password_hash", "access_token", "refresh_token"]:
            clean_meta.pop(forbidden, None)

        audit_entry = AuthAuditLogModel(
            audit_id=f"aud_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            user_id=user_id,
            email=email.strip().lower() if email else None,
            ip_address=ip_address,
            user_agent=user_agent[:250] if user_agent else None,
            success=success,
            metadata_json=json.dumps(clean_meta),
            timestamp=datetime.now(timezone.utc)
        )
        self.session.add(audit_entry)
        await self.session.flush()
        return audit_entry

    async def get_audit_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[AuthAuditLogModel]:
        """Fetch audit log records sorted newest first."""
        stmt = select(AuthAuditLogModel).order_by(AuthAuditLogModel.timestamp.desc())
        if event_type:
            stmt = stmt.where(AuthAuditLogModel.event_type == event_type)
        if user_id:
            stmt = stmt.where(AuthAuditLogModel.user_id == user_id)
        stmt = stmt.offset(offset).limit(limit)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
