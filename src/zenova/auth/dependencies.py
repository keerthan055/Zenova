"""FastAPI authentication dependencies, RBAC guards, and IDOR protection."""
from typing import Optional, List, Callable
from fastapi import Request, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from zenova.db.session import get_db
from zenova.db.auth_repository import AuthRepository
from zenova.db.models import UserModel
from zenova.auth.security import decode_access_token
from zenova.core.logging import get_logger

logger = get_logger("zenova.auth.dependencies")


def extract_token_from_request(request: Request) -> Optional[str]:
    """Extract bearer token from Authorization header or fallback to HttpOnly cookie."""
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    # Fallback to secure cookie
    cookie_token = request.cookies.get("zenova_access_token")
    if cookie_token:
        return cookie_token

    return None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> UserModel:
    """Validate access token and return the authenticated user account."""
    token = extract_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing Bearer token or session cookie.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        payload = decode_access_token(token)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(err),
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    repo = AuthRepository(db)
    user = await repo.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support."
        )

    return user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[UserModel]:
    """Gracefully resolve user if credentials provided, without rejecting unauthenticated requests."""
    token = extract_token_from_request(request)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        repo = AuthRepository(db)
        user = await repo.get_user_by_id(user_id)
        if user and user.is_active:
            return user
    except Exception:
        return None
    return None


def require_role(*allowed_roles: str) -> Callable:
    """Dependency factory enforcing Role-Based Access Control (RBAC)."""
    async def role_checker(
        current_user: UserModel = Depends(get_current_user)
    ) -> UserModel:
        # Admin has superuser access
        if current_user.role == "admin":
            return current_user

        if current_user.role not in allowed_roles:
            logger.warning(
                f"Forbidden access: User {current_user.id} ({current_user.role}) "
                f"attempted to access endpoint requiring {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}"
            )
        return current_user

    return role_checker


def enforce_user_ownership(
    target_user_id: str,
    current_user: UserModel
) -> None:
    """IDOR Defense: Verify that target resource belongs to current user unless clinician/admin authorized."""
    if current_user.role in ("admin", "clinician"):
        return  # Clinicians and Admins have authorized cross-user access subject to audit logging

    if current_user.id != target_user_id:
        logger.warning(
            f"[IDOR VIOLATION DETECTED] User {current_user.id} attempted to access "
            f"or mutate resources owned by {target_user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to access or modify this account's records."
        )
