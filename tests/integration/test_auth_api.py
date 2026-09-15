"""Integration tests for authentication REST endpoints and HTML web pages."""
import uuid
import pytest
from fastapi.testclient import TestClient
from zenova.api.app import app
from zenova.auth.email import _dev_provider


@pytest.fixture(scope="module", autouse=True)
def setup_client():
    _dev_provider.clear()
    with TestClient(app) as client:
        yield client


def test_auth_pages_render_html(setup_client):
    client = setup_client
    for path in ["/login", "/register", "/forgot-password", "/reset-password", "/verify-email"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers["Content-Type"]
        assert "ZENOVA" in res.text


def test_register_login_me_logout_flow(setup_client):
    client = setup_client
    uid = uuid.uuid4().hex[:6]
    email = f"api_test_flow_{uid}@zenova.ai"
    password = "FlowPassword2026!"

    # 1. Register
    reg_payload = {
        "email": email,
        "password": password,
        "confirm_password": password,
        "display_name": "API Flow User"
    }
    reg_resp = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    assert reg_data["email"] == email
    assert reg_data["role"] == "user"
    assert "password" not in reg_data

    # 2. Login
    login_payload = {"email": email, "password": password, "remember_me": True}
    login_resp = client.post("/api/v1/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    access_token = login_data["access_token"]
    refresh_token = login_data["refresh_token"]
    assert access_token is not None
    assert refresh_token is not None

    # Verify HttpOnly cookies set
    assert "zenova_access_token" in login_resp.cookies
    assert "zenova_refresh_token" in login_resp.cookies

    # 3. GET /me with Bearer token
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email
    assert me_resp.json()["display_name"] == "API Flow User"

    # 4. Refresh token
    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    assert refresh_resp.status_code == 200
    new_access = refresh_resp.json()["access_token"]
    new_refresh = refresh_resp.json()["refresh_token"]
    assert new_access is not None
    assert new_refresh != refresh_token

    # 5. Logout
    logout_resp = client.post(
        "/api/v1/auth/logout",
        cookies={"zenova_refresh_token": new_refresh}
    )
    assert logout_resp.status_code == 200
    assert logout_resp.json()["success"] is True

    # 6. Attempt refresh with revoked token -> 401
    bad_refresh = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": new_refresh}
    )
    assert bad_refresh.status_code == 401


def test_forgot_and_reset_password_api(setup_client):
    client = setup_client
    uid = uuid.uuid4().hex[:6]
    email = f"api_reset_test_{uid}@zenova.ai"
    password = "InitialPassword123!"

    # Register
    client.post("/api/v1/auth/register", json={"email": email, "password": password})

    # Forgot password
    forgot_resp = client.post("/api/v1/auth/forgot-password", json={"email": email})
    assert forgot_resp.status_code == 200
    token = _dev_provider.get_last_token_for(email, "password_reset")
    assert token is not None

    # Reset password
    new_pwd = "UpdatedSecretPassword2026!"
    reset_resp = client.post("/api/v1/auth/reset-password", json={
        "token": token,
        "new_password": new_pwd,
        "confirm_new_password": new_pwd
    })
    assert reset_resp.status_code == 200

    # Old password fails
    fail_login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert fail_login.status_code == 401

    # New password succeeds
    good_login = client.post("/api/v1/auth/login", json={"email": email, "password": new_pwd})
    assert good_login.status_code == 200


def test_email_verification_api(setup_client):
    client = setup_client
    uid = uuid.uuid4().hex[:6]
    email = f"api_verify_test_{uid}@zenova.ai"
    password = "VerifyPassword123!"

    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    token = _dev_provider.get_last_token_for(email, "verification")
    assert token is not None

    verify_resp = client.post("/api/v1/auth/verify-email", json={"token": token})
    assert verify_resp.status_code == 200
    assert verify_resp.json()["success"] is True
