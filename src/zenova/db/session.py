"""SQLAlchemy database connection and session management."""
import os
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from zenova.core.logging import get_logger

logger = get_logger("zenova.db.session")

Base = declarative_base()

# Default SQLite path
DEFAULT_DB_PATH = "data/zenova.db"
os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)

# Database URLs
SYNC_DATABASE_URL = os.getenv("DATABASE_URL_SYNC", f"sqlite:///{DEFAULT_DB_PATH}")
ASYNC_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}")

# Async engine and sessionmaker
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in ASYNC_DATABASE_URL else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Sync engine for synchronous migrations or CLI scripts
sync_engine = create_engine(
    SYNC_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in SYNC_DATABASE_URL else {}
)
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


async def init_db() -> None:
    """Initialize database tables asynchronously."""
    # Ensure all models are loaded into Base.metadata
    import zenova.db.models  # noqa: F401
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        def _migrate(connection):
            try:
                res = connection.exec_driver_sql("PRAGMA table_info(conversation_turns)").fetchall()
                col_names = [r[1] for r in res]
                if col_names and "behavior_json" not in col_names:
                    connection.exec_driver_sql("ALTER TABLE conversation_turns ADD COLUMN behavior_json TEXT")
                if col_names and "voice_json" not in col_names:
                    connection.exec_driver_sql("ALTER TABLE conversation_turns ADD COLUMN voice_json TEXT")
                if col_names and "context_json" not in col_names:
                    connection.exec_driver_sql("ALTER TABLE conversation_turns ADD COLUMN context_json TEXT")
                if col_names and "feedback_json" not in col_names:
                    connection.exec_driver_sql("ALTER TABLE conversation_turns ADD COLUMN feedback_json TEXT")

                # Migrate escalation_events columns
                esc_res = connection.exec_driver_sql("PRAGMA table_info(escalation_events)").fetchall()
                esc_cols = [r[1] for r in esc_res]
                if esc_cols:
                    new_cols = [
                        ("severity", "TEXT DEFAULT 'critical'"),
                        ("trigger_type", "TEXT DEFAULT 'crisis_risk'"),
                        ("reason_json", "TEXT"),
                        ("context_summary_json", "TEXT"),
                        ("acknowledged_by", "TEXT"),
                        ("acknowledged_at", "DATETIME"),
                        ("action_taken", "TEXT"),
                        ("resolution_notes", "TEXT"),
                        ("resolved_by", "TEXT"),
                        ("resolved_at", "DATETIME"),
                        ("updated_at", "DATETIME"),
                    ]
                    for col, col_type in new_cols:
                        if col not in esc_cols:
                            connection.exec_driver_sql(f"ALTER TABLE escalation_events ADD COLUMN {col} {col_type}")
            except Exception as e:
                logger.warning(f"Schema migration warning: {e}")

        await conn.run_sync(_migrate)
    logger.info("Database tables initialized successfully.")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for async database transactions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
