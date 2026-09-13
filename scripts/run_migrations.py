"""Database migration runner script for CI/CD and container startups."""
import os
import sys
from pathlib import Path
from alembic.config import Config
from alembic import command

# Add src to pythonpath
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from zenova.core.logging import get_logger

logger = get_logger("zenova.scripts.migrations")


def run_migrations(alembic_cfg_path: str = "alembic.ini", target_revision: str = "head") -> bool:
    """Execute Alembic migrations to the specified revision."""
    cfg_file = root_dir / alembic_cfg_path
    if not cfg_file.exists():
        logger.error(f"Alembic config file not found at {cfg_file}")
        return False

    alembic_cfg = Config(str(cfg_file))
    db_url = os.getenv("DATABASE_URL_SYNC", "sqlite:///data/zenova.db")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        logger.info(f"Applying database migrations to '{target_revision}' using {db_url}...")
        command.upgrade(alembic_cfg, target_revision)
        logger.info("Database migrations applied successfully.")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    rev = sys.argv[1] if len(sys.argv) > 1 else "head"
    success = run_migrations(target_revision=rev)
    sys.exit(0 if success else 1)
