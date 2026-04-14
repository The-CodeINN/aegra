import os
import re
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegra_api import __version__


def parse_lower(v: str) -> str:
    """Converts to lowercase and strips whitespace."""
    return v.strip().lower() if isinstance(v, str) else v


def parse_upper(v: str) -> str:
    """Converts to uppercase and strips whitespace."""
    return v.strip().upper() if isinstance(v, str) else v


# Custom types for automatic formatting
LowerStr = Annotated[str, BeforeValidator(parse_lower)]
UpperStr = Annotated[str, BeforeValidator(parse_upper)]


def _find_env_file() -> str | None:
    """Locate the nearest project .env file for local development."""
    env_override = os.getenv("_AEGRA_ENV_FILE")
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.is_file():
            return str(candidate)

    search_roots = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()

    for root in search_roots:
        for candidate_root in [root, *root.parents]:
            if candidate_root in seen:
                continue
            seen.add(candidate_root)

            env_file = candidate_root / ".env"
            if env_file.is_file():
                return str(env_file)

    return None


_ENV_FILE = _find_env_file()


class EnvBase(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
    )


class AppSettings(EnvBase):
    """General application settings."""

    PROJECT_NAME: str = "Aegra"
    VERSION: str = __version__

    # Server config
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 2026
    SERVER_URL: str | None = None

    @model_validator(mode="after")
    def _validate_keepalive_interval(self) -> "AppSettings":
        if self.KEEPALIVE_INTERVAL_SECS <= 0:
            raise ValueError(f"KEEPALIVE_INTERVAL_SECS must be greater than 0, got {self.KEEPALIVE_INTERVAL_SECS}")
        return self

    @model_validator(mode="after")
    def _derive_server_url(self) -> "AppSettings":
        """Derive SERVER_URL from HOST/PORT when not explicitly set."""
        if self.SERVER_URL is None:
            host = "localhost" if self.HOST in ("0.0.0.0", "127.0.0.1") else self.HOST  # nosec B104
            object.__setattr__(self, "SERVER_URL", f"http://{host}:{self.PORT}")
        return self

    # App logic
    AEGRA_CONFIG: str = "aegra.json"  # Default config file path
    KEEPALIVE_INTERVAL_SECS: float = 5  # Heartbeat interval for join/wait endpoints
    AUTH_TYPE: LowerStr = "noop"
    ENV_MODE: UpperStr = "LOCAL"
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: UpperStr = "INFO"
    LOG_VERBOSITY: LowerStr = "verbose"

    # Custom LMS Integration
    LMS_JWT_SECRET: str | None = None
    LMS_URL: str = "http://localhost:3000"
    ADMIN_EMAIL_ADDRESS: str | None = None
    ADMIN_PASSWORD: str | None = None
    MONGODB_URI: str | None = None
    MONGODB_DB_NAME: str | None = None

    # Title Generator
    TITLE_GENERATOR_MODEL: str = "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0"


class DatabaseSettings(EnvBase):
    """Database connection settings.

    Supports two configuration modes:
    1. DATABASE_URL (standard for containerized deployments) — parsed into individual fields
    2. Individual POSTGRES_* vars — used when DATABASE_URL is not set
    """

    DATABASE_URL: str | None = None

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "aegra"
    DB_ECHO_LOG: bool = False

    @staticmethod
    def _normalize_scheme(url: str, target_scheme: str) -> str:
        """Replace the URL scheme/driver prefix with the target scheme."""
        return re.sub(r"^postgres(?:ql)?(\+\w+)?://", f"{target_scheme}://", url)

    @computed_field
    @property
    def database_url(self) -> str:
        """Async URL for SQLAlchemy (asyncpg)."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql+asyncpg")
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync URL for LangGraph/Psycopg (postgresql://)."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql")
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def database_url_sqlalchemy_sync(self) -> str:
        """Sync URL for SQLAlchemy migrations using the psycopg v3 driver."""
        if self.DATABASE_URL:
            return self._normalize_scheme(self.DATABASE_URL, "postgresql+psycopg")
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


class PoolSettings(EnvBase):
    """Connection pool settings for SQLAlchemy and LangGraph."""

    SQLALCHEMY_POOL_SIZE: int = 10
    SQLALCHEMY_MAX_OVERFLOW: int = 20

    LANGGRAPH_MIN_POOL_SIZE: int = 5
    LANGGRAPH_MAX_POOL_SIZE: int = 20


class ObservabilitySettings(EnvBase):
    """
    Unified settings for OpenTelemetry and Vendor targets.
    Supports Fan-out configuration via OTEL_TARGETS.
    """

    # General OTEL Config
    OTEL_SERVICE_NAME: str = "aegra-backend"
    OTEL_TARGETS: str = ""  # Comma-separated: "LANGFUSE,PHOENIX"
    OTEL_CONSOLE_EXPORT: bool = False  # For local debugging

    # --- Generic OTLP Target (Default/Custom) ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_EXPORTER_OTLP_HEADERS: str | None = None

    # --- Prometheus Metrics ---
    ENABLE_PROMETHEUS_METRICS: bool = False

    # --- Langfuse Specifics ---
    LANGFUSE_BASE_URL: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # --- Phoenix Specifics ---
    PHOENIX_COLLECTOR_ENDPOINT: str = "http://127.0.0.1:6006/v1/traces"
    PHOENIX_API_KEY: str | None = None


class PushNotificationSettings(EnvBase):
    """Web Push (VAPID) settings."""

    VAPID_PUBLIC_KEY: str | None = None
    VAPID_PRIVATE_KEY: str | None = None
    VAPID_CLAIMS_EMAIL: str = "mailto:admin@dedatahub.com"


class DiscoverySettings(EnvBase):
    """Opportunity discovery settings."""

    SERPER_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    DISCOVERY_MAX_TRACKS: int = 2
    DISCOVERY_QUERIES_PER_CATEGORY: int = 2
    DISCOVERY_MAX_MANUAL_SCANS_PER_DAY: int = 4


class EmailSettings(EnvBase):
    """Email notification settings (SMTP)."""

    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_USE_TLS: bool = True
    EMAIL_FROM_ADDRESS: str = "noreply@dedatahub.com"
    EMAIL_FROM_NAME: str = "DeDataHub AI Advisor"
    EMAIL_ENABLED: bool = False  # Must be explicitly enabled


class RedisSettings(EnvBase):
    """Redis settings for the event broker.

    When REDIS_BROKER_ENABLED is True, SSE streaming uses Redis pub/sub
    instead of in-memory queues, enabling multi-instance deployments.
    """

    REDIS_BROKER_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CHANNEL_PREFIX: str = "aegra:run:"
    REDIS_MAX_CONNECTIONS: int = 250


class WorkerSettings(EnvBase):
    """Worker configuration for background graph execution.

    When REDIS_BROKER_ENABLED is True, runs are dispatched to worker
    coroutines via a Redis List job queue instead of local asyncio tasks.
    Each worker loop dequeues run_ids from Redis and spawns up to
    N_JOBS_PER_WORKER concurrent asyncio tasks for graph execution.
    """

    WORKER_COUNT: int = 3
    N_JOBS_PER_WORKER: int = 10
    WORKER_QUEUE_KEY: str = "aegra:jobs"
    WORKER_DRAIN_TIMEOUT: float = 30.0
    BG_JOB_TIMEOUT_SECS: int = 3600
    BG_JOB_MAX_RETRIES: int = 3

    # Lease-based crash recovery.
    # The lease must be long enough that a healthy worker NEVER loses it.
    # Safety margin = LEASE / HEARTBEAT = 30/10 = 3 missed heartbeats
    # before expiry (industry standard — matches Kubernetes liveness probes).
    # Worst-case recovery: ~30s lease expiry + ~20s reaper interval = ~50s.
    LEASE_DURATION_SECONDS: int = 30
    HEARTBEAT_INTERVAL_SECONDS: int = 10
    REAPER_INTERVAL_SECONDS: int = 15
    STUCK_PENDING_THRESHOLD_SECONDS: int = 120
    POSTGRES_POLL_INTERVAL_SECONDS: int = 5

    @model_validator(mode="after")
    def _validate_lease_timing(self) -> "WorkerSettings":
        if self.LEASE_DURATION_SECONDS <= 2 * self.HEARTBEAT_INTERVAL_SECONDS:
            raise ValueError(
                f"LEASE_DURATION_SECONDS ({self.LEASE_DURATION_SECONDS}) must be "
                f"greater than 2 * HEARTBEAT_INTERVAL_SECONDS ({self.HEARTBEAT_INTERVAL_SECONDS}). "
                f"A worker must survive at least 2 missed heartbeats before its lease expires."
            )
        return self


class AWSSettings(EnvBase):
    """AWS / Bedrock settings."""

    AWS_REGION_NAME: str = "eu-west-2"
    AWS_BEARER_TOKEN_BEDROCK: str | None = None


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.pool = PoolSettings()
        self.observability = ObservabilitySettings()
        self.push = PushNotificationSettings()
        self.discovery = DiscoverySettings()
        self.email = EmailSettings()
        self.redis = RedisSettings()
        self.aws = AWSSettings()
        self.worker = WorkerSettings()


settings = Settings()
