"""Alembic environment configuration for Aegra database migrations."""

import threading
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection

# Import your SQLAlchemy models here
from aegra_api.core.orm import Base  # noqa: E402
from aegra_api.settings import settings  # noqa: E402
from alembic import context

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the URL from settings — this respects DATABASE_URL, individual
# POSTGRES_* vars, and preserves query params (e.g. ?sslmode=require).
config.set_main_option("sqlalchemy.url", settings.db.database_url)

# Interpret the config file for Python logging.
# Only reconfigure logging when running from CLI (main thread).
# When invoked programmatically via asyncio.to_thread(), fileConfig()
# causes a cross-thread deadlock with the application's logging.
# See: https://github.com/sqlalchemy/alembic/discussions/1483
if config.config_file_name is not None and threading.current_thread() is threading.main_thread():
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Uses the same computed sync URL as the application runtime.
    """
    connectable = create_engine(settings.db.database_url_sqlalchemy_sync, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
