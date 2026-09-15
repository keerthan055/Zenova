"""Security, IDOR, RBAC, and Token Tampering verification test suite."""
import asyncio
import time
import pytest
import jwt
from fastapi.testclient import TestClient

from zenova.api.app import app
from zenova.auth.email import _dev_provider
from zenova.db.session import AsyncSessionLocal, init_db
from zenova.core.config import get_system_config


@pytest.fixture(scope="module", autouse=True)
def setup_security_suite():
    _dev_provider.clear()
    asyncio.run(init_db())
    with TestClient(app) as client:
        yield client


def create_user_with_role(email: str, password: str, role: str, display_name: str) -> dict:
    """Helper to register and elevate user role directly in test DB."""
    async def _create():
        from zenova.auth.service import AuthService
        from zenova.schemas.auth import UserRegisterRequest
        async with AsyncSessionLocal() as session:
            service = AuthService(session)
            req = UserRegisterRequest(
                email=email,
                password=password,
                confirm_password=password,
                display_name=display_name
            )
            user_resp = await service.register(req, role=role)
            await session.commit()
            return user_resp

    return asyncio.run(_create())


def test_idor_prevention_cross_user_access(setup_security_suite):
    """Verify that User A cannot read or mutate User B's resources (IDOR defense)."""
    client = setup_security_suite
    ts = int(time.time() * 1000)
    user_a_email = f"user_a_{ts}@zenova.ai"
    user_b_email = f"user_b_{ts}@zenova.ai"
    password = "SecurePassword2026!"

    # 1. Register User A and User B
    resp_a = client.post("/api/v1/auth/register", json={
        "email": user_a_email, "password": password, "confirm_password": password, "display_name": "User A"
    })
    assert resp_a.status_code == 201
    user_a_id = resp_a.json()["id"]

    resp_b = client.post("/api/v1/auth/register", json={
        "email": user_b_email, "password": password, "confirm_password": password, "display_name": "User B"
    })
    assert resp_b.status_code == 201
    user_b_id = resp_b.json()["id"]

    # 2. Login User A to obtain auth token
    login_a = client.post("/api/v1/auth/login", json={"email": user_a_email, "password": password})
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 3. User A attempts to access User B's preferences -> Expect 403 Forbidden
    resp_pref = client.get(f"/api/v1/user/preferences/{user_b_id}", headers=headers_a)
    assert resp_pref.status_code == 403
    assert "Access forbidden" in resp_pref.json()["detail"]

    # 4. User A attempts to update User B's preferences -> Expect 403 Forbidden
    resp_update = client.put(
        f"/api/v1/user/preferences/{user_b_id}",
        json={"save_history": False},
        headers=headers_a
    )
    assert resp_update.status_code == 403

    # 5. User A attempts to export User B's data -> Expect 403 Forbidden
    resp_export = client.post(f"/api/v1/user/export/{user_b_id}", headers=headers_a)
    assert resp_export.status_code == 403

    # 6. User A attempts to delete User B's account data -> Expect 403 Forbidden
    resp_delete = client.delete(f"/api/v1/user/data/{user_b_id}", headers=headers_a)
    assert resp_delete.status_code == 403

    # 7. User A CAN access their own preferences -> Expect 200 OK
    resp_own = client.get(f"/api/v1/user/preferences/{user_a_id}", headers=headers_a)
    assert resp_own.status_code == 200


def test_role_based_access_control_rbac(setup_security_suite):
    """Verify RBAC boundaries: Standard User vs Clinician vs Admin."""
    client = setup_security_suite
    ts = int(time.time() * 1000)
    password = "RbacPassword2026!"

    # Setup 3 accounts with different roles
    user_email = f"regular_{ts}@zenova.ai"
    clinician_email = f"clinician_{ts}@zenova.ai"
    admin_email = f"admin_{ts}@zenova.ai"

    create_user_with_role(user_email, password, "user", "Standard User")
    create_user_with_role(clinician_email, password, "clinician", "Clinician Dr. Smith")
    create_user_with_role(admin_email, password, "admin", "System Admin")

    # Log in each user
    login_u = client.post("/api/v1/auth/login", json={"email": user_email, "password": password})
    assert login_u.status_code == 200
    token_u = login_u.json()["access_token"]

    login_c = client.post("/api/v1/auth/login", json={"email": clinician_email, "password": password})
    assert login_c.status_code == 200
    token_c = login_c.json()["access_token"]

    login_a = client.post("/api/v1/auth/login", json={"email": admin_email, "password": password})
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]

    head_u = {"Authorization": f"Bearer {token_u}"}
    head_c = {"Authorization": f"Bearer {token_c}"}
    head_a = {"Authorization": f"Bearer {token_a}"}

    # 1. Clinician dashboard endpoint (/api/v1/dashboard/overview)
    # Standard user MUST get 403 Forbidden
    res_u_dash = client.get("/api/v1/dashboard/overview", headers=head_u)
    assert res_u_dash.status_code == 403

    # Clinician MUST get 200 OK
    res_c_dash = client.get("/api/v1/dashboard/overview", headers=head_c)
    assert res_c_dash.status_code == 200

    # Admin MUST get 200 OK (admin has superuser bypass)
    res_a_dash = client.get("/api/v1/dashboard/overview", headers=head_a)
    assert res_a_dash.status_code == 200

    # 2. Admin audit trail endpoint (/api/v1/auth/audits)
    # Standard user MUST get 403 Forbidden
    res_u_audit = client.get("/api/v1/auth/audits", headers=head_u)
    assert res_u_audit.status_code == 403

    # Clinician MUST get 403 Forbidden (audits are admin-only)
    res_c_audit = client.get("/api/v1/auth/audits", headers=head_c)
    assert res_c_audit.status_code == 403

    # Admin MUST get 200 OK
    res_a_audit = client.get("/api/v1/auth/audits", headers=head_a)
    assert res_a_audit.status_code == 200
    assert "count" in res_a_audit.json()
    assert "entries" in res_a_audit.json()


def test_token_tampering_and_expiration(setup_security_suite):
    """Verify rejection of manipulated, fake, or expired tokens."""
    client = setup_security_suite
    cfg = get_system_config()

    # 1. Totally invalid token string
    res_fake = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"})
    assert res_fake.status_code == 401

    # 2. Token signed with wrong secret key
    tampered_token = jwt.encode(
        {"sub": "fake-user-id", "email": "fake@zenova.ai", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 3600},
        "wrong-secret-key-attacker-32-chars-long",
        algorithm="HS256"
    )
    res_tampered = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered_token}"})
    assert res_tampered.status_code == 401

    # 3. Expired token (exp in the past)
    expired_token = jwt.encode(
        {"sub": "expired-user-id", "email": "exp@zenova.ai", "role": "user", "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600},
        cfg.security.secret_key,
        algorithm=cfg.security.algorithm
    )
    res_expired = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res_expired.status_code == 401
    assert "expired" in res_expired.json()["detail"].lower()


def test_single_use_token_replay_attack_prevention(setup_security_suite):
    """Verify that password reset tokens and verification tokens cannot be reused."""
    client = setup_security_suite
    ts = int(time.time() * 1000)
    email = f"replay_guard_{ts}@zenova.ai"
    password = "InitialPassword2026!"

    # Register
    client.post("/api/v1/auth/register", json={
        "email": email, "password": password, "confirm_password": password, "display_name": "Replay Tester"
    })

    # 1. Request Password Reset
    client.post("/api/v1/auth/forgot-password", json={"email": email})
    reset_token = _dev_provider.get_last_token_for(email, "password_reset")
    assert reset_token is not None

    # First reset -> 200 OK
    res_reset1 = client.post("/api/v1/auth/reset-password", json={
        "token": reset_token,
        "new_password": "NewSecretPassword2026!",
        "confirm_password": "NewSecretPassword2026!"
    })
    assert res_reset1.status_code == 200

    # Second reset with same token -> 400 Bad Request (Replay prevention)
    res_reset2 = client.post("/api/v1/auth/reset-password", json={
        "token": reset_token,
        "new_password": "AnotherPassword2026!",
        "confirm_password": "AnotherPassword2026!"
    })
    assert res_reset2.status_code == 400
    assert "invalid" in res_reset2.json()["detail"].lower() or "expired" in res_reset2.json()["detail"].lower()

    # 2. Email verification replay test
    client.post("/api/v1/auth/resend-verification", json={"email": email})
    verify_token = _dev_provider.get_last_token_for(email, "verification")
    assert verify_token is not None

    # First verification -> 200 OK
    res_v1 = client.post("/api/v1/auth/verify-email", json={"token": verify_token})
    assert res_v1.status_code == 200

    # Second verification with same token -> 400 Bad Request
    res_v2 = client.post("/api/v1/auth/verify-email", json={"token": verify_token})
    assert res_v2.status_code == 400
    assert "invalid" in res_v2.json()["detail"].lower() or "expired" in res_v2.json()["detail"].lower()


def test_credential_leakage_prevention(setup_security_suite):
    """Verify that password hashes, salts, and secret credentials are never exposed."""
    client = setup_security_suite
    ts = int(time.time() * 1000)
    email = f"leak_test_{ts}@zenova.ai"
    password = "LeakCheckPassword2026!"

    # Check register response
    reg_res = client.post("/api/v1/auth/register", json={
        "email": email, "password": password, "confirm_password": password, "display_name": "Leak Tester"
    })
    assert reg_res.status_code == 201
    reg_json = reg_res.json()
    assert "password_hash" not in reg_json
    assert "password" not in reg_json
    assert "salt" not in reg_json

    # Check login response
    log_res = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert log_res.status_code == 200
    log_json = log_res.json()
    assert "password_hash" not in log_json
    assert "password" not in log_json

    # Check /me response
    token = log_json["access_token"]
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_json = me_res.json()
    assert "password_hash" not in me_json
    assert "password" not in me_json
