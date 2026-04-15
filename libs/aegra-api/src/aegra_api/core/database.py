"""Database manager with LangGraph integration"""

import os

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import load_store_config
from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None

        # Shared pool for LangGraph components (Checkpointer + Store)
        self.lg_pool: AsyncConnectionPool | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None
        self._database_url = settings.db.database_url

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # Idempotency check: if already initialized, do nothing
        if self.engine:
            return

        # 1. SQLAlchemy Engine (app metadata, uses asyncpg)
        # We strictly limit this pool because the main load
        # is handled by LangGraph components.
        self.engine = create_async_engine(
            self._database_url,
            pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
            connect_args={"prepared_statement_cache_size": 0},  # PgBouncer compatibility
        )

        lg_max = settings.pool.LANGGRAPH_MAX_POOL_SIZE
        lg_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,  # Disable prepared statements for PgBouncer compatibility
            "row_factory": dict_row,  # LangGraph requires dictionary rows, not tuples
        }

        # Create a single shared pool.
        # 'open=False' is important to avoid RuntimeWarning; we open it explicitly below.
        self.lg_pool = AsyncConnectionPool(
            conninfo=settings.db.database_url_sync,
            min_size=settings.pool.LANGGRAPH_MIN_POOL_SIZE,
            max_size=lg_max,
            open=False,
            kwargs=lg_kwargs,
            check=AsyncConnectionPool.check_connection,
        )

        # Explicitly open the pool
        await self.lg_pool.open()

        # 2. Initialize LangGraph components using the shared pool
        # Passing 'conn=self.lg_pool' prevents components from creating their own pools.

        logger.info(f"Initializing LangGraph components with shared pool (max {lg_max} conns)...")

        self._checkpointer = AsyncPostgresSaver(
            conn=self.lg_pool,
            serde=JsonPlusSerializer(
                allowed_json_modules=[("react_agent.cost_tracker", "SessionCost")],
            ),
        )
        await self._checkpointer.setup()  # Ensure tables exist

        # Load store configuration for semantic search (if configured)
        store_config = load_store_config()
        index_config = store_config.get("index") if store_config else None

        # When the embed string targets Bedrock, materialise the embeddings object
        # here so we can inject the AWS region.  LangChain's generic init_embeddings
        # path (called inside AsyncPostgresStore.__init__) constructs BedrockEmbeddings
        # without a region, causing a ValidationError at startup.
        if index_config and isinstance(index_config.get("embed"), str):
            embed_str: str = index_config["embed"]
            if embed_str.startswith("bedrock:"):
                from langchain_aws import BedrockEmbeddings

                model_id = embed_str.removeprefix("bedrock:")
                region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_DEFAULT_REGION") or "eu-west-2"
                # Cohere models require input_type to distinguish indexing from retrieval.
                model_kwargs = (
                    {"input_type": "search_document", "truncate": "END"} if model_id.startswith("cohere.") else {}
                )
                # BedrockEmbeddings uses invoke_model which only supports SigV4 (IAM) auth.
                # ChatBedrockConverse uses converse which supports bearer token auth and may
                # set AWS_BEARER_TOKEN_BEDROCK in os.environ at init time.  botocore's
                # _set_auth_scheme_preference_signer event handler reads that env var at
                # request time and overrides the signer to 'bearer' for ALL bedrock-runtime
                # clients — including our IAM-based embedding client — causing invoke_model
                # to fail with "Invalid API Key format" or "Unable to locate auth token".
                #
                # Fix: pass Config(signature_version='v4') when creating the boto3 client.
                # botocore wraps the value as ClientConfigString, which sets
                # has_in_code_configuration=True in _set_auth_scheme_preference_signer,
                # preventing the bearer token override entirely.
                import boto3
                from botocore.config import Config as BotocoreConfig

                bedrock_access_key = os.getenv("BEDROCK_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
                bedrock_secret_key = os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
                bedrock_session_token = os.getenv("BEDROCK_AWS_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN")

                sigv4_config = BotocoreConfig(signature_version="v4")

                if bedrock_access_key and bedrock_secret_key:
                    session = boto3.Session(
                        aws_access_key_id=bedrock_access_key,
                        aws_secret_access_key=bedrock_secret_key,
                        **({"aws_session_token": bedrock_session_token} if bedrock_session_token else {}),
                    )
                    bedrock_client = session.client("bedrock-runtime", region_name=region, config=sigv4_config)
                else:
                    # No explicit IAM creds — fall back to boto3 default credential chain.
                    bedrock_client = boto3.client("bedrock-runtime", region_name=region, config=sigv4_config)
                base_embed = BedrockEmbeddings(
                    model_id=model_id,
                    region_name=region,
                    model_kwargs=model_kwargs,
                    client=bedrock_client,
                )

                # Wrap with auto-truncation for Cohere v3.  The langchain_aws
                # library validates text length client-side (2048 char limit)
                # BEFORE sending to the API, so the truncate="END" kwarg never
                # gets a chance to help.  This wrapper truncates inputs before
                # they reach that validation.
                if model_id.startswith("cohere.") and not getattr(base_embed, "_is_cohere_v4", False):
                    from aegra_api.core._truncating_embeddings import TruncatingEmbeddings

                    embed = TruncatingEmbeddings(wrapped=base_embed, max_chars=2000)
                else:
                    embed = base_embed

                index_config = {
                    **index_config,
                    "embed": embed,
                }

        # Migrate BEFORE setup() — setup() uses ADD COLUMN IF NOT EXISTS, which is
        # a no-op if a wrong-dimension column already exists from a previous model.
        if index_config and index_config.get("dims"):
            await self._migrate_vector_column(int(index_config["dims"]))

        self._store = AsyncPostgresStore(conn=self.lg_pool, index=index_config)
        await self._store.setup()  # Ensure tables / indexes exist

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info(f"Semantic store enabled with embeddings: {embed_model}")

        # Initialize the session maker for ORM operations
        from .orm import initialize_session_maker

        initialize_session_maker()

        logger.info("✅ Database and LangGraph components initialized")

    async def _migrate_vector_column(self, target_dims: int) -> None:
        """Ensure store_vectors.embedding has the correct vector dimensions.

        LangGraph tracks vector migrations in a 'vector_migrations' table — once
        applied it never re-runs them, even if the embedding model changes.
        This method detects a dim mismatch, drops store_vectors, and resets the
        tracked migration version so setup() recreates the table correctly.

        Must be called BEFORE AsyncPostgresStore.setup().
        """
        import re

        from psycopg.rows import tuple_row

        async with self.lg_pool.connection() as conn:
            # Check whether the store_vectors table and embedding column exist
            async with conn.cursor(row_factory=tuple_row) as cur:
                await cur.execute(
                    """
                    SELECT format_type(pa.atttypid, pa.atttypmod)
                    FROM pg_attribute pa
                    JOIN pg_class pc ON pa.attrelid = pc.oid
                    JOIN pg_namespace pn ON pc.relnamespace = pn.oid
                    WHERE pn.nspname = 'public'
                      AND pc.relname = 'store_vectors'
                      AND pa.attname = 'embedding'
                      AND pa.attnum > 0
                      AND NOT pa.attisdropped
                    """
                )
                row = await cur.fetchone()

            if row is None:
                logger.info("store_vectors not found; setup() will create it with correct dims")
                return

            col_type: str = row[0]  # e.g. "vector(1536)"
            logger.info(f"store_vectors.embedding current type: {col_type}")

            match = re.search(r"vector\((\d+)\)", col_type)
            if not match:
                logger.warning(f"Cannot parse vector dims from '{col_type}'; skipping migration")
                return

            current_dims = int(match.group(1))
            if current_dims == target_dims:
                logger.info(f"store_vectors.embedding already {target_dims} dims; no migration needed")
                return

            logger.info(
                f"Migrating store_vectors.embedding: {current_dims} → {target_dims} dims. "
                "Dropping table and resetting vector_migrations so setup() recreates it."
            )
            # Drop the table (CASCADE drops the index too)
            await conn.execute("DROP TABLE IF EXISTS store_vectors CASCADE")
            # Reset vector_migrations to v0 (extension-only) so setup() re-runs
            # the CREATE TABLE and CREATE INDEX migrations (v1 and v2)
            await conn.execute("DELETE FROM vector_migrations WHERE v >= 1")
            logger.info("✅ store_vectors dropped and vector_migrations reset — setup() will recreate")

    async def close(self) -> None:
        """Close database connections"""
        # Close SQLAlchemy engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None

        # Close shared LangGraph pool
        if self.lg_pool:
            await self.lg_pool.close()
            self.lg_pool = None
            self._checkpointer = None
            self._store = None

        # Reset the session maker cache
        from .orm import reset_session_maker

        reset_session_maker()

        logger.info("✅ Database connections closed")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """Return the live AsyncPostgresSaver instance."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_store(self) -> AsyncPostgresStore:
        """Return the live AsyncPostgresStore instance."""
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()
