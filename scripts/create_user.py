"""CLI utility to create development and administrative users in ZENOVA."""
import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zenova.db.session import init_db, get_db_session
from zenova.auth.service import AuthService
from zenova.schemas.auth import UserRegisterRequest


async def main():
    parser = argparse.ArgumentParser(description="Create a ZENOVA user account with specified role.")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--password", required=True, help="User password (min 8 chars)")
    parser.add_argument("--role", default="user", choices=["user", "clinician", "admin"], help="Account role")
    parser.add_argument("--name", default=None, help="Display name or alias")
    parser.add_argument("--verified", action="store_true", default=True, help="Mark email as pre-verified")

    args = parser.parse_args()

    # Ensure tables are initialized
    await init_db()

    async with get_db_session() as db:
        service = AuthService(db)
        req = UserRegisterRequest(
            email=args.email,
            password=args.password,
            display_name=args.name
        )
        try:
            user = await service.register(req, role=args.role, ip_address="127.0.0.1", user_agent="zenova-cli")
            if args.verified:
                await service.repo.update_user(user.id, {"is_verified": True})
            print(f"[SUCCESS] Created {args.role.upper()} user successfully:")
            print(f"  ID:           {user.id}")
            print(f"  Email:        {user.email}")
            print(f"  Role:         {args.role}")
            print(f"  Display Name: {args.name or '(None)'}")
            print(f"  Verified:     {args.verified}")
        except ValueError as exc:
            print(f"[ERROR] Failed to create user: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
