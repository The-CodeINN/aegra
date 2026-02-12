FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder

# Retrieve the uv binary directly from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /bin/uv

# Install system build dependencies required for compiling Python extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace root files (uv.lock lives at the repo root)
COPY pyproject.toml uv.lock ./

# Copy aegra-api package metadata (preserving workspace layout)
COPY libs/aegra-api/pyproject.toml libs/aegra-api/README.md libs/aegra-api/

# Install dependencies from the workspace lock file.
RUN uv export --frozen --no-dev --no-emit-project \
    --package aegra-api \
    --format=requirements-txt > requirements.txt && \
    uv pip install --system --compile-bytecode -r requirements.txt && \
    rm requirements.txt

# Copy the actual project source code.
COPY libs/aegra-api/src/ libs/aegra-api/src/

# Copy alembic files required by the build (forced includes in pyproject.toml)
COPY libs/aegra-api/alembic.ini libs/aegra-api/alembic.ini
COPY libs/aegra-api/alembic/ libs/aegra-api/alembic/

# Install the project package itself (from the aegra-api subdirectory).
RUN uv pip install --system --compile-bytecode --no-deps libs/aegra-api/

# -----------------------------
# Final, minimal runtime image
# -----------------------------
FROM base AS final

# Install only minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy runtime assets required by the app and alembic
COPY libs/aegra-api/alembic.ini ./alembic.ini
COPY libs/aegra-api/alembic/ ./alembic/
COPY aegra.json ./aegra.json
COPY auth.py ./auth.py
COPY examples/ ./examples/

EXPOSE 8000

# Add entrypoint that attempts migrations then starts the app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Run as non-root
USER app

# Entrypoint will attempt to run alembic (best-effort) then exec the CMD below
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "aegra_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
