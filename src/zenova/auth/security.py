"""Cryptographic utilities, Argon2id password hashing, and JWT token management."""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import jwt
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError, InvalidHashError

from zenova.core.config import get_system_config
from zenova.core.logging import get_logger

logger = get_logger("zenova.auth.security")

# Modern, secure Argon2id password hasher configuration (RFC 9106 recommended)
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=1,
    hash_len=32,
    type=Type.ID
)

# Common compromised passwords blocklist
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "admin123", "letmein1", "welcome1", "iloveyou", "monkey123",
    "football", "dragon123", "master123", "sunshine", "princess", "zenova123",
    "testing123", "trustno1", "changeme", "pass1234", "superman", "starwars"
}


def hash_password(password: str) -> str:
    """Hash plaintext password using Argon2id with automatic cryptographic salt."""
    if not password or not isinstance(password, str):
        raise ValueError("Password must be a non-empty string.")
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against Argon2id hash using constant-time comparison."""
    if not password or not hashed:
        return False
    try:
        return _hasher.verify(hashed, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during password verification: {exc}")
        return False


def validate_password_policy(password: str) -> Tuple[bool, Optional[str]]:
    """Validate that password meets secure, practical baseline policy."""
    if not password or not isinstance(password, str):
        return False, "Password cannot be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password must not exceed 128 characters."
    if password.strip() != password:
        return False, "Password must not have leading or trailing whitespace."
    if password.lower() in COMMON_PASSWORDS:
        return False, "This password is too common or easily guessed. Please choose a stronger password."
    if len(set(password)) < 3:
        return False, "Password must not consist of identical repeating characters."
    return True, None


def create_access_token(
    user_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Generate a signed JWT access token for API authorization."""
    cfg = get_system_config()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=cfg.security.token_expire_minutes))
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": uuid.uuid4().hex
    }
    token = jwt.encode(payload, cfg.security.secret_key, algorithm=cfg.security.algorithm)
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    cfg = get_system_config()
    try:
        payload = jwt.decode(
            token,
            cfg.security.secret_key,
            algorithms=[cfg.security.algorithm],
            options={"require": ["sub", "exp", "iat"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError as err:
        raise ValueError(f"Invalid token: {str(err)}")


def generate_secure_token() -> str:
    """Generate a high-entropy 32-byte cryptographic random hex token."""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """Compute deterministic SHA-256 hash of an opaque token for secure DB storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
