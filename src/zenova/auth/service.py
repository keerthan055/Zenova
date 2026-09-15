"""High-level Authentication Service coordinating business logic, token issuance, and audits."""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession

from zenova.db.auth_repository import AuthRepository
from zenova.db.models import UserModel
from zenova.schemas.auth import (
    AuthRole,
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    SafeUserResponse,
    UserProfileUpdateRequest
)
from zenova.auth.security import (
    hash_password,
    verify_password,
    validate_password_policy,
    create_access_token,
    generate_secure_token,
    hash_token
)
from zenova.auth.email import get_email_provider
from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.auth.service")


class AuthService:
    """Orchestrates account registration, authentication, token rotation, and credential lifecycle."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = AuthRepository(session)
        self.cfg = get_system_config()

    def _to_safe_user(self, user: UserModel) -> SafeUserResponse:
        """Convert UserModel into a safe, non-sensitive Pydantic response."""
        return SafeUserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=AuthRole(user.role),
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at.isoformat() if user.created_at else datetime.now(timezone.utc).isoformat(),
            last_login_at=user.last_login_at.isoformat() if user.last_login_at else None
        )

    # ==========================================================================
    # Registration & Onboarding
    # ==========================================================================

    async def register(
        self,
        req: UserRegisterRequest,
        role: str = "user",
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> SafeUserResponse:
        """Register a new user account."""
        # 1. Password confirmation check
        if req.confirm_password is not None and req.password != req.confirm_password:
            raise ValueError("Passwords do not match.")

        # 2. Password policy validation
        valid, err = validate_password_policy(req.password)
        if not valid:
            raise ValueError(err)

        # 3. Duplicate email check
        email = req.email.strip().lower()
        existing = await self.repo.get_user_by_email(email)
        if existing:
            await self.repo.record_audit_event(
                event_type="REGISTER_DUPLICATE_ATTEMPT",
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            raise ValueError("An account with this email address already exists.")

        # 4. Hash password with Argon2id
        pwd_hash = hash_password(req.password)
        user_id = f"usr_{uuid.uuid4().hex}"

        # 5. Persist user record
        user = await self.repo.create_user(
            user_id=user_id,
            email=email,
            password_hash=pwd_hash,
            display_name=req.display_name,
            role=role,
            is_verified=False,
            is_active=True
        )

        # 6. Generate single-use email verification token
        raw_token = generate_secure_token()
        token_h = hash_token(raw_token)
        verify_expires = datetime.now(timezone.utc) + timedelta(hours=self.cfg.security.email_verification_expire_hours)
        await self.repo.create_email_verification_token(user_id, token_h, verify_expires)

        # 7. Dispatch verification email via configured provider
        email_provider = get_email_provider()
        await email_provider.send_verification_email(email, raw_token, req.display_name)

        # 8. Record audit log
        await self.repo.record_audit_event(
            event_type="REGISTER",
            user_id=user_id,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            metadata={"role": role}
        )

        logger.info(f"Registered new user account {user_id} ({email})")
        return self._to_safe_user(user)

    # ==========================================================================
    # Authentication & Login
    # ==========================================================================

    async def login(
        self,
        req: UserLoginRequest,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """Authenticate credentials and issue JWT access token + refresh token."""
        email = req.email.strip().lower()
        user = await self.repo.get_user_by_email(email)

        # Constant-time generic error defense: do not reveal user existence
        if not user or not verify_password(req.password, user.password_hash):
            await self.repo.record_audit_event(
                event_type="LOGIN_FAILURE",
                user_id=user.id if user else None,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            raise ValueError("Invalid email or password.")

        if not user.is_active:
            await self.repo.record_audit_event(
                event_type="LOGIN_DISABLED_ATTEMPT",
                user_id=user.id,
                email=email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            raise ValueError("Your account has been deactivated. Please contact support.")

        # 1. Issue short-lived JWT Access Token
        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            expires_delta=timedelta(minutes=self.cfg.security.token_expire_minutes)
        )

        # 2. Issue high-entropy Opaque Refresh Token
        raw_refresh_token = generate_secure_token()
        refresh_hash = hash_token(raw_refresh_token)

        days = self.cfg.security.refresh_token_expire_days * (2 if req.remember_me else 1)
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=days)
        session_id = f"sess_{uuid.uuid4().hex[:16]}"

        await self.repo.create_session(
            session_id=session_id,
            user_id=user.id,
            token_hash=refresh_hash,
            expires_at=refresh_expires,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # 3. Touch user last login
        await self.repo.update_last_login(user.id)

        # 4. Record audit log
        await self.repo.record_audit_event(
            event_type="LOGIN_SUCCESS",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            metadata={"session_id": session_id}
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=raw_refresh_token,
            token_type="bearer",
            expires_in=self.cfg.security.token_expire_minutes * 60,
            user=self._to_safe_user(user)
        )

    # ==========================================================================
    # Token Refresh & Rotation
    # ==========================================================================

    async def refresh_tokens(
        self,
        raw_refresh_token: str,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """Rotate refresh token and issue a fresh access token."""
        if not raw_refresh_token:
            raise ValueError("Missing refresh token.")

        token_h = hash_token(raw_refresh_token)
        session_record = await self.repo.get_session_by_token_hash(token_h)
        if not session_record:
            raise ValueError("Invalid, expired, or revoked refresh token.")

        user = await self.repo.get_user_by_id(session_record.user_id)
        if not user or not user.is_active:
            raise ValueError("Account not found or inactive.")

        # Revoke old refresh token (Strict Token Rotation)
        await self.repo.revoke_session_by_token_hash(token_h)

        # Issue new token pair
        new_access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            expires_delta=timedelta(minutes=self.cfg.security.token_expire_minutes)
        )

        new_raw_refresh = generate_secure_token()
        new_refresh_hash = hash_token(new_raw_refresh)
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=self.cfg.security.refresh_token_expire_days)
        new_session_id = f"sess_{uuid.uuid4().hex[:16]}"

        await self.repo.create_session(
            session_id=new_session_id,
            user_id=user.id,
            token_hash=new_refresh_hash,
            expires_at=refresh_expires,
            ip_address=ip_address,
            user_agent=user_agent
        )

        await self.repo.record_audit_event(
            event_type="TOKEN_REFRESH",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_raw_refresh,
            token_type="bearer",
            expires_in=self.cfg.security.token_expire_minutes * 60,
            user=self._to_safe_user(user)
        )

    # ==========================================================================
    # Logout & Revocation
    # ==========================================================================

    async def logout(
        self,
        raw_refresh_token: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: str = "127.0.0.1"
    ) -> None:
        """Revoke active session and record logout audit."""
        if raw_refresh_token:
            token_h = hash_token(raw_refresh_token)
            await self.repo.revoke_session_by_token_hash(token_h)

        await self.repo.record_audit_event(
            event_type="LOGOUT",
            user_id=user_id,
            ip_address=ip_address,
            success=True
        )

    # ==========================================================================
    # Password Reset Lifecycle
    # ==========================================================================

    async def request_password_reset(
        self,
        email: str,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> None:
        """Initiate password reset workflow without revealing email existence."""
        clean_email = email.strip().lower()
        user = await self.repo.get_user_by_email(clean_email)

        if user and user.is_active:
            raw_token = generate_secure_token()
            token_h = hash_token(raw_token)
            expires = datetime.now(timezone.utc) + timedelta(minutes=self.cfg.security.password_reset_expire_minutes)

            await self.repo.create_password_reset_token(user.id, token_h, expires)
            provider = get_email_provider()
            await provider.send_password_reset_email(clean_email, raw_token, user.display_name)

            await self.repo.record_audit_event(
                event_type="PASSWORD_RESET_REQUEST",
                user_id=user.id,
                email=clean_email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=True
            )
        else:
            await self.repo.record_audit_event(
                event_type="PASSWORD_RESET_UNKNOWN_EMAIL",
                email=clean_email,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )

    async def confirm_password_reset(
        self,
        token: str,
        new_password: str,
        confirm_password: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> None:
        """Validate single-use token and update password."""
        if confirm_password is not None and new_password != confirm_password:
            raise ValueError("Passwords do not match.")

        valid, err = validate_password_policy(new_password)
        if not valid:
            raise ValueError(err)

        token_h = hash_token(token)
        reset_record = await self.repo.get_valid_password_reset_token(token_h)
        if not reset_record:
            raise ValueError("Invalid, expired, or previously used password reset link.")

        user = await self.repo.get_user_by_id(reset_record.user_id)
        if not user or not user.is_active:
            raise ValueError("User account is inactive or not found.")

        # Hash new password
        pwd_hash = hash_password(new_password)
        await self.repo.update_user(user.id, {"password_hash": pwd_hash})

        # Mark token as consumed
        await self.repo.mark_password_reset_token_used(reset_record.id)

        # Invalidate all existing sessions (prevent session hijacking)
        revoked_count = await self.repo.revoke_all_user_sessions(user.id)

        await self.repo.record_audit_event(
            event_type="PASSWORD_RESET_SUCCESS",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            metadata={"revoked_sessions": revoked_count}
        )

    # ==========================================================================
    # Email Verification
    # ==========================================================================

    async def verify_email(
        self,
        token: str,
        ip_address: str = "127.0.0.1"
    ) -> None:
        """Validate single-use token and mark account as verified."""
        token_h = hash_token(token)
        record = await self.repo.get_valid_email_verification_token(token_h)
        if not record:
            raise ValueError("Invalid, expired, or previously used verification link.")

        user = await self.repo.get_user_by_id(record.user_id)
        if not user:
            raise ValueError("User account not found.")

        await self.repo.update_user(user.id, {"is_verified": True})
        await self.repo.mark_email_verification_token_used(record.id)

        await self.repo.record_audit_event(
            event_type="EMAIL_VERIFICATION",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            success=True
        )

    async def resend_verification(
        self,
        email: str,
        ip_address: str = "127.0.0.1"
    ) -> None:
        """Resend verification email to unverified account."""
        clean_email = email.strip().lower()
        user = await self.repo.get_user_by_email(clean_email)
        if user and not user.is_verified and user.is_active:
            raw_token = generate_secure_token()
            token_h = hash_token(raw_token)
            expires = datetime.now(timezone.utc) + timedelta(hours=self.cfg.security.email_verification_expire_hours)

            await self.repo.create_email_verification_token(user.id, token_h, expires)
            provider = get_email_provider()
            await provider.send_verification_email(clean_email, raw_token, user.display_name)

            await self.repo.record_audit_event(
                event_type="VERIFICATION_RESEND",
                user_id=user.id,
                email=clean_email,
                ip_address=ip_address,
                success=True
            )

    # ==========================================================================
    # Password Change & Profile Update
    # ==========================================================================

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        confirm_password: Optional[str] = None,
        ip_address: str = "127.0.0.1"
    ) -> None:
        """Change password for an authenticated user."""
        if confirm_password is not None and new_password != confirm_password:
            raise ValueError("New passwords do not match.")

        valid, err = validate_password_policy(new_password)
        if not valid:
            raise ValueError(err)

        user = await self.repo.get_user_by_id(user_id)
        if not user or not verify_password(current_password, user.password_hash):
            raise ValueError("Current password is incorrect.")

        pwd_hash = hash_password(new_password)
        await self.repo.update_user(user.id, {"password_hash": pwd_hash})
        await self.repo.revoke_all_user_sessions(user.id)

        await self.repo.record_audit_event(
            event_type="PASSWORD_CHANGE",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            success=True
        )

    async def update_profile(
        self,
        user_id: str,
        req: UserProfileUpdateRequest,
        ip_address: str = "127.0.0.1"
    ) -> SafeUserResponse:
        """Update profile information."""
        updates = {}
        if req.display_name is not None:
            updates["display_name"] = req.display_name.strip()

        user = await self.repo.update_user(user_id, updates)
        if not user:
            raise ValueError("User not found.")

        await self.repo.record_audit_event(
            event_type="PROFILE_UPDATE",
            user_id=user.id,
            email=user.email,
            ip_address=ip_address,
            success=True,
            metadata={"updated_fields": list(updates.keys())}
        )

        return self._to_safe_user(user)
