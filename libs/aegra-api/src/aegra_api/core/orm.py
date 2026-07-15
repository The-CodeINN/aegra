"""SQLAlchemy ORM setup for persistent assistant/thread/run records.

This module creates:
• `Base` – the declarative base used by our models.
• `Assistant`, `Thread`, `Run` – ORM models mirroring the bootstrap tables
  already created in ``DatabaseManager._create_metadata_tables``.
• `async_session_maker` – a factory that hands out `AsyncSession` objects
  bound to the shared engine managed by `db_manager`.
• `get_session` – FastAPI dependency helper for routers.

Nothing is auto-imported by FastAPI yet; routers will `from ...core.db import get_session`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

_logger = structlog.getLogger(__name__)

# Safety net against pathological/adversarial nesting. Real agent JSON rarely
# exceeds a few dozen levels; Python's default frame limit is ~1000. 200 is
# well above any legitimate payload and well below the interpreter ceiling.
_MAX_STRIP_DEPTH = 200


def _strip_null_bytes(value: Any, _depth: int = 0) -> Any:
    """Recursively strip U+0000 from strings inside JSON-compatible structures.

    Postgres JSONB rejects \\u0000 with UntranslatableCharacterError; agent
    output can contain literal NULL bytes from untrusted input or model
    hallucination. Stripping at the type boundary protects every JSONB column.

    Beyond ``_MAX_STRIP_DEPTH`` the value is returned untouched — a deeper
    payload than that is almost certainly adversarial, and letting Postgres
    reject it surfaces a clearer signal than a RecursionError at bind time.
    """
    if _depth >= _MAX_STRIP_DEPTH:
        _logger.warning("jsonb_strip_depth_exceeded", depth=_depth, type=type(value).__name__)
        return value
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for k, v in value.items():
            stripped_k = _strip_null_bytes(k, _depth + 1)
            if stripped_k in result:
                # Two distinct raw keys collapsed to the same stripped key — the
                # earlier value is being dropped by last-wins. Surface so silent
                # data loss is visible in logs.
                _logger.warning("jsonb_strip_key_collision", stripped_key=stripped_k)
            result[stripped_k] = _strip_null_bytes(v, _depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_strip_null_bytes(v, _depth + 1) for v in value]
    return value


class JsonbSafe(TypeDecorator):
    """JSONB column that strips NULL bytes from string values before write.

    Drop-in replacement for ``JSONB``. Read path is untouched — only
    ``process_bind_param`` runs, so existing rows are unaffected.
    """

    impl = JSONB
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        return _strip_null_bytes(value)


class Base(DeclarativeBase):
    pass


class Assistant(Base):
    __tablename__ = "assistant"

    # gen_random_uuid() is in Postgres 13+ core; no extension needed.
    assistant_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    context: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # Indexes for performance
    __table_args__ = (
        Index("idx_assistant_user", "user_id"),
        Index("idx_assistant_user_assistant", "user_id", "assistant_id", unique=True),
        Index(
            "idx_assistant_user_graph_config",
            "user_id",
            "graph_id",
            text("md5(config::text)"),
            unique=True,
        ),
    )


class AssistantVersion(Base):
    __tablename__ = "assistant_versions"

    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict | None] = mapped_column(JsonbSafe)
    context: Mapped[dict | None] = mapped_column(JsonbSafe)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)


class Thread(Base):
    __tablename__ = "thread"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, server_default=text("'idle'"))
    # Database column is 'metadata_json' (per database.py). ORM attribute 'metadata_json' must map to that column.
    metadata_json: Mapped[dict] = mapped_column("metadata_json", JsonbSafe, server_default=text("'{}'::jsonb"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # Indexes for performance
    __table_args__ = (Index("idx_thread_user", "user_id"),)


class Run(Base):
    __tablename__ = "runs"

    # gen_random_uuid() is in Postgres 13+ core; no extension needed.
    run_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    thread_id: Mapped[str] = mapped_column(Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=False)
    assistant_id: Mapped[str | None] = mapped_column(Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    input: Mapped[dict | None] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    # Some environments may not yet have a 'config' column; make it nullable without default to match existing DB.
    # If migrations add this column later, it's already represented here.
    config: Mapped[dict | None] = mapped_column(JsonbSafe, nullable=True)
    context: Mapped[dict | None] = mapped_column(JsonbSafe, nullable=True)
    output: Mapped[dict | None] = mapped_column(JsonbSafe)
    error_message: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # Worker execution: stores RunJob params so workers can reconstruct
    # the job from the database after receiving a run_id via Redis.
    execution_params: Mapped[dict | None] = mapped_column(JsonbSafe, nullable=True)

    # Lease-based crash recovery: tracks which worker owns a run and
    # when the lease expires. A background reaper re-enqueues runs
    # whose leases have expired (worker crashed).
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_runs_thread_id", "thread_id"),
        Index("idx_runs_user", "user_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_assistant_id", "assistant_id"),
        Index("idx_runs_created_at", "created_at"),
        Index("idx_runs_lease_reaper", "status", "lease_expires_at"),
    )


class ActivityLog(Base):
    """Activity log for tracking student/user interactions with the AI system"""

    __tablename__ = "activity_log"

    activity_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_type: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # e.g., "prompt", "query", "run_started", "run_completed", "run_failed"
    action_status: Mapped[str] = mapped_column(
        Text, server_default=text("'success'")
    )  # "success", "failed", "interrupted"
    details: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    metadata_json: Mapped[dict] = mapped_column("metadata_json", JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # Indexes for performance and common queries
    __table_args__ = (
        Index("idx_activity_log_user", "user_id"),
        Index("idx_activity_log_user_created", "user_id", "created_at"),
        Index("idx_activity_log_assistant", "assistant_id"),
        Index("idx_activity_log_thread", "thread_id"),
        Index("idx_activity_log_action_type", "action_type"),
        Index("idx_activity_log_created", "created_at"),
    )


class Cron(Base):
    __tablename__ = "crons"

    # gen_random_uuid() is in Postgres 13+ core; no extension needed.
    cron_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("gen_random_uuid()::text"))
    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=True
    )
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    schedule: Mapped[str] = mapped_column(Text, nullable=False)
    # JsonbSafe strips NULL bytes from user payloads — same protection as runs.input.
    payload: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"))
    metadata_dict: Mapped[dict] = mapped_column(JsonbSafe, server_default=text("'{}'::jsonb"), name="metadata")
    on_run_completed: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, server_default=text("true"), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    next_run_date: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    claimed_until: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    __table_args__ = (
        Index("idx_cron_user", "user_id"),
        Index("idx_cron_assistant_id", "assistant_id"),
        Index("idx_cron_thread_id", "thread_id"),
        Index("idx_cron_next_run", "enabled", "next_run_date"),
    )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a cached async_sessionmaker bound to db_manager.engine."""
    global async_session_maker
    if async_session_maker is None:
        from aegra_api.core.database import db_manager

        # Ensure database is initialized before getting engine
        if not db_manager.engine:
            raise RuntimeError("Database not initialized. Call db_manager.initialize() during app startup.")
        async_session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
    return async_session_maker


# Backwards-compatible alias for callers that imported the private symbol.
_get_session_maker = get_session_maker


def initialize_session_maker() -> None:
    """Initialize the session maker during app startup."""
    get_session_maker()


def reset_session_maker() -> None:
    """Reset the session maker cache on shutdown."""
    global async_session_maker
    async_session_maker = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an AsyncSession."""
    maker = get_session_maker()
    async with maker() as session:
        yield session
