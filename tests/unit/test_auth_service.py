"""Unit tests for AuthService business logic, token rotation, and credential lifecycle."""
import uuid
import pytest
from zenova.db.session import init_db, get_db_session
from zenova.auth.service import AuthService
from zenova.auth.email import _dev_provider
from zenova.schemas.auth import UserRegisterRequest, UserLoginRequest, UserProfileUpdateRequest
import asyncio


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    asyncio.run(init_db())
    _dev_provider.clear()


@pytest.mark.asyncio
async def test_register_and_duplicate_handling():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_user_{uid}@zenova.ai"
        req = UserRegisterRequest(
            email=email,
            password="SecurePassword123!",
            confirm_password="SecurePassword123!",
            display_name="Unit Tester 01"
        )
        user = await service.register(req)
        assert user.email == email
        assert user.display_name == "Unit Tester 01"
        assert user.role.value == "user"
        assert user.is_verified is False

        # Attempt duplicate registration
        with pytest.raises(ValueError, match="already exists"):
            await service.register(req)


@pytest.mark.asyncio
async def test_login_success_and_failure():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_login_{uid}@zenova.ai"
        req_reg = UserRegisterRequest(email=email, password="MySecretLoginPass123!")
        await service.register(req_reg)

        # Successful login
        req_login = UserLoginRequest(email=email, password="MySecretLoginPass123!")
        tokens = await service.login(req_login)
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.user.email == email

        # Bad password
        with pytest.raises(ValueError, match="Invalid email or password"):
            await service.login(UserLoginRequest(email=email, password="WrongPassword!"))

        # Unknown email
        with pytest.raises(ValueError, match="Invalid email or password"):
            await service.login(UserLoginRequest(email="nonexistent@zenova.ai", password="AnyPassword123!"))


@pytest.mark.asyncio
async def test_token_refresh_and_rotation():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_refresh_{uid}@zenova.ai"
        await service.register(UserRegisterRequest(email=email, password="MyRefreshPassword123!"))

        tokens = await service.login(UserLoginRequest(email=email, password="MyRefreshPassword123!"))
        orig_refresh = tokens.refresh_token

        # Refresh
        new_tokens = await service.refresh_tokens(orig_refresh)
        assert new_tokens.access_token is not None
        assert new_tokens.refresh_token != orig_refresh

        # Old refresh token should be revoked (Rotation)
        with pytest.raises(ValueError, match="Invalid, expired, or revoked"):
            await service.refresh_tokens(orig_refresh)


@pytest.mark.asyncio
async def test_logout_revocation():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_logout_{uid}@zenova.ai"
        await service.register(UserRegisterRequest(email=email, password="MyLogoutPassword123!"))

        tokens = await service.login(UserLoginRequest(email=email, password="MyLogoutPassword123!"))
        refresh = tokens.refresh_token

        # Logout
        await service.logout(raw_refresh_token=refresh, user_id=tokens.user.id)

        # Refresh attempt with logged-out token should fail
        with pytest.raises(ValueError, match="Invalid, expired, or revoked"):
            await service.refresh_tokens(refresh)


@pytest.mark.asyncio
async def test_password_reset_flow():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_reset_{uid}@zenova.ai"
        await service.register(UserRegisterRequest(email=email, password="OriginalPassword123!"))

        # Request reset
        await service.request_password_reset(email)
        token = _dev_provider.get_last_token_for(email, "password_reset")
        assert token is not None

        # Reset password
        new_pwd = "BrandNewSecurePassword123!"
        await service.confirm_password_reset(token, new_pwd, confirm_password=new_pwd)

        # Login with old password fails
        with pytest.raises(ValueError, match="Invalid email or password"):
            await service.login(UserLoginRequest(email=email, password="OriginalPassword123!"))

        # Login with new password succeeds
        tokens = await service.login(UserLoginRequest(email=email, password=new_pwd))
        assert tokens.user.email == email

        # Reuse of consumed token fails
        with pytest.raises(ValueError, match="Invalid, expired, or previously used"):
            await service.confirm_password_reset(token, "AnotherPassword123!")


@pytest.mark.asyncio
async def test_email_verification_flow():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_verify_{uid}@zenova.ai"
        await service.register(UserRegisterRequest(email=email, password="MyVerificationPassword123!"))

        token = _dev_provider.get_last_token_for(email, "verification")
        assert token is not None

        # Verify
        await service.verify_email(token)
        user = await service.repo.get_user_by_email(email)
        assert user.is_verified is True

        # Reuse of verification token fails
        with pytest.raises(ValueError, match="Invalid, expired, or previously used"):
            await service.verify_email(token)


@pytest.mark.asyncio
async def test_change_password_and_profile_update():
    uid = uuid.uuid4().hex[:6]
    async with get_db_session() as db:
        service = AuthService(db)
        email = f"unit_profile_{uid}@zenova.ai"
        user = await service.register(UserRegisterRequest(email=email, password="CurrentPassword123!"))

        # Change password
        await service.change_password(user.id, "CurrentPassword123!", "NextPassword123!", "NextPassword123!")

        # Old password fails
        with pytest.raises(ValueError):
            await service.login(UserLoginRequest(email=email, password="CurrentPassword123!"))

        # Update profile
        updated = await service.update_profile(user.id, UserProfileUpdateRequest(display_name="Updated Name"))
        assert updated.display_name == "Updated Name"
