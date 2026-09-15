"""Unit tests for cryptographic utilities, Argon2id hashing, and JWT tokens."""
import pytest
from datetime import timedelta
from zenova.auth.security import (
    hash_password,
    verify_password,
    validate_password_policy,
    create_access_token,
    decode_access_token,
    generate_secure_token,
    hash_token
)


def test_argon2id_hash_and_verify():
    pwd = "CorrectHorseBatteryStaple99!"
    hashed = hash_password(pwd)
    assert hashed.startswith("$argon2id$")
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(pwd, "") is False


def test_password_policy_validation():
    # Valid passwords
    assert validate_password_policy("StrongValidPass123!")[0] is True
    assert validate_password_policy("my-secure-long-passphrase")[0] is True

    # Too short (< 8)
    valid, err = validate_password_policy("short")
    assert valid is False
    assert "at least 8" in err

    # Empty
    valid, err = validate_password_policy("")
    assert valid is False

    # Common compromised password
    valid, err = validate_password_policy("password123")
    assert valid is False
    assert "too common" in err

    # Repeating characters
    valid, err = validate_password_policy("aaaaaaaaaa")
    assert valid is False
    assert "repeating" in err

    # Leading/trailing whitespace
    valid, err = validate_password_policy(" pass12345 ")
    assert valid is False
    assert "whitespace" in err


def test_jwt_access_token_generation_and_decoding():
    user_id = "usr_test123"
    email = "tester@zenova.ai"
    role = "user"

    token = create_access_token(user_id=user_id, email=email, role=role)
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert payload["role"] == role
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload


def test_jwt_expired_token():
    user_id = "usr_test_expired"
    email = "expired@zenova.ai"
    token = create_access_token(user_id, email, "user", expires_delta=timedelta(seconds=-10))

    with pytest.raises(ValueError, match="expired"):
        decode_access_token(token)


def test_jwt_tampered_token():
    user_id = "usr_test_tamper"
    email = "tamper@zenova.ai"
    token = create_access_token(user_id, email, "user")
    tampered = token[:-4] + "abcd"

    with pytest.raises(ValueError, match="Invalid token"):
        decode_access_token(tampered)


def test_secure_random_tokens_and_hashing():
    raw1 = generate_secure_token()
    raw2 = generate_secure_token()
    assert len(raw1) == 64  # 32 bytes hex
    assert raw1 != raw2

    h1 = hash_token(raw1)
    h2 = hash_token(raw1)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
