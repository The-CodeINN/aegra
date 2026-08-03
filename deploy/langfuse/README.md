# Self-hosted Langfuse v3 (with RustFS) for ai-service tracing

Stands up the observability backend for the OTel tracing wired in spec Item 0.
The ai-service exports spans to `POST /api/public/otel/v1/traces` with header
`x-langfuse-ingestion-version: 4` (see
`libs/aegra-api/src/aegra_api/observability/targets/langfuse.py`) — that
endpoint is **Langfuse v3 only**, which is why this stack pins `langfuse:3`.
Blob storage is **RustFS** (S3-compatible, Apache 2.0) rather than MinIO.

## Stack

| Service | Purpose | Host port |
|---|---|---|
| `langfuse-web` | UI + API + **OTel ingestion endpoint** | 3000 |
| `langfuse-worker` | async trace/event processor | 3030 (loopback) |
| `postgres` | transactional store | 5432 (loopback) |
| `clickhouse` | trace/observation OLAP store | 8123 / 9004 (loopback) |
| `redis` | queue + cache | 6379 (loopback) |
| `rustfs` | S3 blob storage (events, media) | 9090 (S3) / 9091 (console) |
| `rustfs-init` | one-shot: creates the `langfuse` bucket | — |

## Run

```bash
cd deploy/langfuse
cp .env.example .env
# fill in .env — generate secrets:
#   openssl rand -base64 32   # NEXTAUTH_SECRET, SALT
#   openssl rand -hex 32      # ENCRYPTION_KEY (64 hex chars)
# and pick your own POSTGRES_PASSWORD / CLICKHOUSE_PASSWORD / REDIS_AUTH /
# RUSTFS_ACCESS_KEY / RUSTFS_SECRET_KEY / LANGFUSE_INIT_* keys.

docker compose up -d
# first boot runs DB + ClickHouse migrations; give it ~1–2 min
curl -f http://localhost:3000/api/public/ready   # 200 when ready
```

- **UI:** http://localhost:3000 — log in with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`.
- **RustFS console:** http://localhost:9091 — `RUSTFS_ACCESS_KEY` / `RUSTFS_SECRET_KEY`.

## Point ai-service at it

The ai-service is already wired — only env vars change. Add these to the
ai-service `.env` (and `.env.production.template`), using the **same** public/
secret keys you set in `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` /
`LANGFUSE_INIT_PROJECT_SECRET_KEY`:

```
OTEL_TARGETS=LANGFUSE
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...      # = LANGFUSE_INIT_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=sk-lf-...      # = LANGFUSE_INIT_PROJECT_SECRET_KEY
```

**Networking:** `LANGFUSE_BASE_URL` must be reachable *from the ai-service
container*. Options:
- ai-service runs on the **same host**, on the host network or via the
  published port → `http://host.docker.internal:3000` (Docker Desktop) or the
  host IP.
- Prod / same Docker host → put both stacks on a shared external network and
  use `http://langfuse-web:3000`:
  ```bash
  docker network create shared-observability
  # add to BOTH compose files:  networks: [shared-observability]  (external: true)
  ```

Restart ai-service, run one agent turn, and the trace appears in the Langfuse
UI. The span-enrichment from Item 0 (`langfuse.user.id`,
`langfuse.session.id` = thread id, `langfuse.trace.name` = graph id) shows up
as filterable trace attributes.

## Notes / caveats

- **RAM:** ClickHouse + the two Langfuse containers want ~4 GB. A `t3.medium`
  (per Langfuse's own guidance) is the floor for a real deployment.
- **Data residency:** everything here stays in your infra (RustFS is
  no-telemetry, `TELEMETRY_ENABLED=false`) — good for EU compliance.
- **Media/batch-export external endpoint** (`S3_EXTERNAL_ENDPOINT`) is only
  used by the browser for presigned media/download URLs; trace ingestion from
  the ai-service does **not** need it. Batch export is disabled by default here.
- **TLS/prod:** front `langfuse-web` with a TLS-terminating reverse proxy and
  set `LANGFUSE_NEXTAUTH_URL` / `S3_EXTERNAL_ENDPOINT` to the public HTTPS URLs.
