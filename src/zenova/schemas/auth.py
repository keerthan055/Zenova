"""ZENOVA Authentication and User Account Management Schemas."""
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, EmailStr, field_validator


class AuthRole(str, Enum):
    """Supported user and practitioner roles for RBAC."""
    USER = "user"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class SafeUserResponse(BaseModel):
    """Non-sensitive, safe user account profile exposed in API responses."""
    id: str
    email: str
    display_name: Optional[str] = None
    role: AuthRole
    is_active: bool
    is_verified: bool
    created_at: str
    last_login_at: Optional[str] = None


class UserRegisterRequest(BaseModel):
    """User registration payload."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="Account password")
    confirm_password: Optional[str] = Field(default=None, description="Password confirmation")
    display_name: Optional[str] = Field(default=None, max_length=128, description="Preferred display name")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v


class UserLoginRequest(BaseModel):
    """User login payload."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="Account password")
    remember_me: bool = Field(default=False, description="Extend refresh session expiration")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    """Structured token pair and user profile returned upon successful authentication."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: SafeUserResponse


class RefreshTokenRequest(BaseModel):
    """Optional payload for token refresh if not sent via HttpOnly cookie."""
    refresh_token: Optional[str] = None


class PasswordResetRequest(BaseModel):
    """Request for initiating password reset workflow."""
    email: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class PasswordResetConfirmRequest(BaseModel):
    """Payload for completing password reset with verified token."""
    token: str = Field(..., min_length=16, description="Single-use reset token")
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    """Payload for authenticated password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: Optional[str] = None


class EmailVerifyRequest(BaseModel):
    """Payload for verifying email address."""
    token: str = Field(..., min_length=16, description="Single-use email verification token")


class ResendVerificationRequest(BaseModel):
    """Request to resend verification email."""
    email: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserProfileUpdateRequest(BaseModel):
    """Partial profile update payload."""
    display_name: Optional[str] = Field(default=None, max_length=128)


class AuthMessageResponse(BaseModel):
    """Generic reassuring response message."""
    message: str
    success: bool = True


class AuthAuditEntry(BaseModel):
    """Audit log item representing an authentication or RBAC event."""
    audit_id: str
    event_type: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    ip_address: str
    user_agent: Optional[str] = None
    success: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class AuthAuditListResponse(BaseModel):
    """Admin response for audit event log inspection."""
    count: int
    entries: List[AuthAuditEntry]
