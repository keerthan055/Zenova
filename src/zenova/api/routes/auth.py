"""REST API endpoints for ZENOVA Authentication, Session Management, and RBAC."""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Request, Response, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from zenova.db.session import get_db
from zenova.auth.service import AuthService
from zenova.auth.dependencies import get_current_user, require_role
from zenova.db.models import UserModel
from zenova.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    SafeUserResponse,
    RefreshTokenRequest,
    PasswordResetRequest,
    PasswordResetConfirmRequest,
    PasswordChangeRequest,
    EmailVerifyRequest,
    ResendVerificationRequest,
    UserProfileUpdateRequest,
    AuthMessageResponse,
    AuthAuditListResponse,
    AuthAuditEntry
)
from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.api.routes.auth")

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def get_client_ip(request: Request) -> str:
    """Extract real client IP considering reverse proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def set_auth_cookies(response: Response, tokens: TokenResponse, remember_me: bool = False):
    """Set secure HttpOnly cookies for browser navigation."""
    cfg = get_system_config()
    secure = cfg.security.cookie_secure
    samesite = cfg.security.cookie_samesite

    # 1. Access token cookie (short-lived)
    access_max_age = cfg.security.token_expire_minutes * 60
    response.set_cookie(
        key="zenova_access_token",
        value=tokens.access_token,
        max_age=access_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        path="/"
    )

    # 2. Refresh token cookie (long-lived)
    refresh_max_age = cfg.security.refresh_token_expire_days * (2 if remember_me else 1) * 86400
    response.set_cookie(
        key="zenova_refresh_token",
        value=tokens.refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        path="/"
    )


def clear_auth_cookies(response: Response):
    """Clear session cookies upon logout."""
    cfg = get_system_config()
    secure = cfg.security.cookie_secure
    samesite = cfg.security.cookie_samesite

    response.delete_cookie(key="zenova_access_token", path="/", httponly=True, secure=secure, samesite=samesite)
    response.delete_cookie(key="zenova_refresh_token", path="/", httponly=True, secure=secure, samesite=samesite)


# ==============================================================================
# Public Authentication Endpoints
# ==============================================================================

@router.post("/register", response_model=SafeUserResponse, status_code=status.HTTP_201_CREATED)
async def register_account(
    req: UserRegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> SafeUserResponse:
    """Register a new user account with Argon2id password hashing."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    ua = request.headers.get("User-Agent")
    try:
        user = await service.register(req, role="user", ip_address=ip_addr, user_agent=ua)
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/login", response_model=TokenResponse)
async def login_account(
    req: UserLoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """Authenticate credentials and issue JWT + HttpOnly refresh cookies."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    ua = request.headers.get("User-Agent")
    try:
        tokens = await service.login(req, ip_address=ip_addr, user_agent=ua)
        set_auth_cookies(response, tokens, remember_me=req.remember_me)
        return tokens
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    request: Request,
    response: Response,
    body: Optional[RefreshTokenRequest] = None,
    db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """Rotate refresh token and issue new access token."""
    raw_token = None
    if body and body.refresh_token:
        raw_token = body.refresh_token
    else:
        raw_token = request.cookies.get("zenova_refresh_token")

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token in cookie or request body."
        )

    service = AuthService(db)
    ip_addr = get_client_ip(request)
    ua = request.headers.get("User-Agent")
    try:
        new_tokens = await service.refresh_tokens(raw_token, ip_address=ip_addr, user_agent=ua)
        set_auth_cookies(response, new_tokens)
        return new_tokens
    except ValueError as exc:
        clear_auth_cookies(response)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


@router.post("/logout", response_model=AuthMessageResponse)
async def logout_account(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Revoke refresh session and delete authentication cookies."""
    raw_token = request.cookies.get("zenova_refresh_token")
    service = AuthService(db)
    ip_addr = get_client_ip(request)

    # Attempt to resolve current user if token present (non-failing)
    user_id = None
    auth_header = request.headers.get("Authorization")
    if auth_header and "Bearer" in auth_header:
        try:
            from zenova.auth.security import decode_access_token
            payload = decode_access_token(auth_header.split()[1])
            user_id = payload.get("sub")
        except Exception:
            pass

    await service.logout(raw_refresh_token=raw_token, user_id=user_id, ip_address=ip_addr)
    clear_auth_cookies(response)
    return AuthMessageResponse(message="Successfully logged out.", success=True)


@router.post("/forgot-password", response_model=AuthMessageResponse)
async def forgot_password(
    req: PasswordResetRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Send single-use password reset link without revealing email existence."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    ua = request.headers.get("User-Agent")
    await service.request_password_reset(req.email, ip_address=ip_addr, user_agent=ua)
    return AuthMessageResponse(
        message="If an account matches this email address, password reset instructions have been sent.",
        success=True
    )


@router.post("/reset-password", response_model=AuthMessageResponse)
async def reset_password(
    req: PasswordResetConfirmRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Reset password using verified single-use token."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    ua = request.headers.get("User-Agent")
    try:
        await service.confirm_password_reset(
            token=req.token,
            new_password=req.new_password,
            confirm_password=req.confirm_new_password,
            ip_address=ip_addr,
            user_agent=ua
        )
        clear_auth_cookies(response)
        return AuthMessageResponse(message="Password has been reset successfully. Please log in.", success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/verify-email", response_model=AuthMessageResponse)
async def verify_email_address(
    req: EmailVerifyRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Verify email address using single-use verification token."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    try:
        await service.verify_email(token=req.token, ip_address=ip_addr)
        return AuthMessageResponse(message="Email verified successfully. Welcome to ZENOVA!", success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/resend-verification", response_model=AuthMessageResponse)
async def resend_email_verification(
    req: ResendVerificationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Resend email verification token."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    await service.resend_verification(email=req.email, ip_address=ip_addr)
    return AuthMessageResponse(
        message="If this account is unverified, a new verification link has been dispatched.",
        success=True
    )


# ==============================================================================
# Authenticated User Account Management
# ==============================================================================

@router.get("/me", response_model=SafeUserResponse)
async def get_current_user_profile(
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SafeUserResponse:
    """Fetch profile of currently authenticated user."""
    service = AuthService(db)
    return service._to_safe_user(current_user)


@router.post("/change-password", response_model=AuthMessageResponse)
async def change_password(
    req: PasswordChangeRequest,
    request: Request,
    response: Response,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> AuthMessageResponse:
    """Update password for authenticated user and revoke active sessions."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    try:
        await service.change_password(
            user_id=current_user.id,
            current_password=req.current_password,
            new_password=req.new_password,
            confirm_password=req.confirm_new_password,
            ip_address=ip_addr
        )
        clear_auth_cookies(response)
        return AuthMessageResponse(message="Password changed successfully. Please log in with your new password.", success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.put("/profile", response_model=SafeUserResponse)
async def update_profile(
    req: UserProfileUpdateRequest,
    request: Request,
    current_user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> SafeUserResponse:
    """Update account profile display name."""
    service = AuthService(db)
    ip_addr = get_client_ip(request)
    try:
        return await service.update_profile(user_id=current_user.id, req=req, ip_address=ip_addr)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ==============================================================================
# Admin-Only Audit Inspection
# ==============================================================================

@router.get("/audits", response_model=AuthAuditListResponse)
async def get_audit_trail(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None),
    admin_user: UserModel = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
) -> AuthAuditListResponse:
    """Inspect security and authentication audit logs (restricted to administrators)."""
    repo = AuthService(db).repo
    logs = await repo.get_audit_logs(limit=limit, offset=offset, event_type=event_type)
    import json
    entries = [
        AuthAuditEntry(
            audit_id=log.audit_id,
            event_type=log.event_type,
            user_id=log.user_id,
            email=log.email,
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            success=log.success,
            metadata=json.loads(log.metadata_json) if log.metadata_json else {},
            timestamp=log.timestamp.isoformat() if log.timestamp else ""
        )
        for log in logs
    ]
    return AuthAuditListResponse(count=len(entries), entries=entries)
