# EC2 Production Deployment

This guide covers the production deployment of **ai-service** on an AWS EC2 instance with Docker, automated CI/CD via GitHub Actions, and SSL via Nginx.

## Architecture

```
Internet
   │
   ▼
┌──────────────────────────────────────────────┐
│  EC2 (t3.large · Ubuntu 24.04 · eu-west-2)  │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  System Nginx (ports 80/443)         │    │
│  │  SSL termination · reverse proxy     │    │
│  └──────────┬───────────────────────────┘    │
│             │ proxy_pass :8000               │
│  ┌──────────▼───────────────────────────┐    │
│  │  Docker Network (aegra-net)          │    │
│  │                                      │    │
│  │  ┌────────────┐  ┌───────────────┐   │    │
│  │  │ aegra-app  │  │ aegra-postgres│   │    │
│  │  │ :8000      │──│ :5432         │   │    │
│  │  │ (FastAPI)  │  │ (pgvector/18) │   │    │
│  │  └─────┬──────┘  └───────────────┘   │    │
│  │        │          ┌───────────────┐   │    │
│  │        └──────────│ aegra-redis   │   │    │
│  │                   │ :6379         │   │    │
│  │                   │ (redis:7)     │   │    │
│  │                   └───────────────┘   │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Node.js API (:5000) · Next.js (:3000)       │
│  PM2 · CrowdSec · Certbot                    │
└──────────────────────────────────────────────┘
```

### Key decisions

| Decision   | Choice                | Reason                                                        |
| ---------- | --------------------- | ------------------------------------------------------------- |
| Nginx      | System (not Docker)   | SSL certs already exist; shared with Node.js and Next.js apps |
| PostgreSQL | Docker pgvector/pg18  | Needs pgvector for semantic search; isolated from host        |
| Redis      | Docker redis:7-alpine | Streaming broker for LangGraph; isolated from host            |
| CI/CD      | GitHub Actions → SSH  | Simple; no extra infrastructure needed                        |

## File layout

```
ai-service/
├── .env.production              # Secrets (never committed)
├── .env.production.template     # Template for secrets
├── docker-compose.prod.yml      # Production Docker stack
├── deployments/docker/Dockerfile # Multi-stage build
├── deploy/
│   ├── nginx/
│   │   └── aegra-system.conf   # System Nginx site config
│   └── scripts/
│       ├── deploy.sh           # Per-deployment script (called by CI)
│       ├── ec2-bootstrap.sh    # First-time EC2 setup
│       ├── ec2-cleanup.sh      # Remove old services
│       └── setup-ssl.sh        # SSL certificate management
└── .github/workflows/
    └── deploy.yml              # CI/CD pipeline
```

## Infrastructure details

| Component | Spec                                    |
| --------- | --------------------------------------- |
| Instance  | t3.large (2 vCPU, 8 GB RAM, 96 GB disk) |
| OS        | Ubuntu 24.04 LTS                        |
| Region    | eu-west-2 (London)                      |
| Domain    | agent.dedatahub.io                      |
| SSL       | Let's Encrypt (auto-renewed by Certbot) |

### Resource usage (post-deployment)

| Service        | CPU       | Memory                  |
| -------------- | --------- | ----------------------- |
| aegra-app      | ~0.2%     | ~126 MB (limit: 2 GB)   |
| aegra-postgres | ~0.01%    | ~43 MB (limit: 1 GB)    |
| aegra-redis    | ~0.6%     | ~3.4 MB (limit: 512 MB) |
| Node.js API    | ~0.4%     | ~239 MB                 |
| Next.js        | ~0.0%     | ~158 MB                 |
| **Total**      | **~1.2%** | **~1.5 GB / 7.6 GB**    |

## CI/CD pipeline

### Trigger

Every push to `main` (or manual `workflow_dispatch`) triggers the deploy workflow.

### Flow

```
push to main
     │
     ▼
┌─────────┐     ┌─────────┐     ┌──────────┐
│  Test   │────▶│ Deploy  │────▶│  Verify  │
│ (lint,  │     │ (SSH →  │     │ (curl    │
│  pytest)│     │ deploy  │     │  /health)│
└─────────┘     │  .sh)   │     └──────────┘
                └─────────┘
```

- **Test**: Lints with ruff, runs unit/integration tests.
- **Deploy**: SSHs into EC2, runs `deploy/scripts/deploy.sh` (git pull → docker build → restart aegra → health check → prune old images).
- **Verify**: Curls `http://localhost:8000/health` on EC2.
- Deploy proceeds **even if tests fail** (`if: always()`).
- Concurrent deploys are blocked (`concurrency: deploy-production`).

### GitHub Actions secrets required

Add these in **Settings → Secrets and variables → Actions**:

| Secret        | Value                                        |
| ------------- | -------------------------------------------- |
| `EC2_HOST`    | EC2 public IP address                        |
| `EC2_SSH_KEY` | Contents of the SSH private key (PEM format) |

Also create a **production** environment in **Settings → Environments**.

## Deployment script (`deploy.sh`)

What happens on each deploy:

1. `git fetch origin main && git reset --hard origin/main` — pull latest code
2. `docker compose build --no-cache aegra` — rebuild the app image
3. `docker compose up -d --no-deps --force-recreate aegra` — restart only the app (zero-downtime for postgres/redis)
4. Health check loop (up to 30 retries × 5s = 2.5 min)
5. `docker image prune -f` — clean up dangling images

PostgreSQL and Redis are **not restarted** during deploys — only the app container is rebuilt and restarted.

## Docker services

### PostgreSQL (aegra-postgres)

- Image: `pgvector/pgvector:pg18`
- Port: `127.0.0.1:5432` (localhost only)
- Volume: `postgres_data:/var/lib/postgresql`
- Memory limit: 1 GB
- Health check: `pg_isready`

### Redis (aegra-redis)

- Image: `redis:7-alpine`
- Port: `127.0.0.1:6379` (localhost only)
- Volume: `redis_data:/data`
- Memory limit: 512 MB, maxmemory 256 MB (LRU eviction)
- AOF persistence enabled

### Aegra App (aegra-app)

- Built from `deployments/docker/Dockerfile` (multi-stage, Python 3.11)
- Port: `127.0.0.1:8000` (accessed via system Nginx)
- Memory limit: 2 GB
- Runs as non-root user (`app`)
- Entrypoint: runs Alembic migrations, then starts uvicorn
- Health check: `curl http://localhost:8000/health`

## Nginx configuration

System Nginx at `/etc/nginx/sites-available/aegra.conf`:

- HTTP → HTTPS redirect
- SSL termination with Let's Encrypt certs
- Reverse proxy to `127.0.0.1:8000`
- WebSocket support for streaming
- Proxy timeouts: connect 60s, send 120s, read 300s (LLM calls)
- Security headers: HSTS, X-Frame-Options, X-Content-Type-Options

## First-time setup

### 1. Prepare the EC2 instance

```bash
# SSH into the instance
ssh -i key.pem ubuntu@<EC2_IP>

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
# Log out and back in for group change to take effect

# Create deploy log
sudo touch /var/log/aegra-deploy.log
sudo chown ubuntu:ubuntu /var/log/aegra-deploy.log
```

### 2. Clone the repo

```bash
sudo mkdir -p /opt/ai-service
sudo chown ubuntu:ubuntu /opt/ai-service
git clone git@github.com:dedatahub/ai-service.git /opt/ai-service
cd /opt/ai-service
```

### 3. Configure environment

```bash
# Copy template and fill in real values
cp .env.production.template .env.production

# Edit with your secrets (API keys, passwords, etc.)
nano .env.production

# Create symlink so Docker Compose reads it for variable interpolation
ln -sf .env.production .env
```

Key variables to set:

- `POSTGRES_PASSWORD` — strong random password
- `REDIS_PASSWORD` — strong random password
- `OPENAI_API_KEY` — your OpenAI key
- `ANTHROPIC_API_KEY` — your Anthropic key
- `AUTH_TYPE` — `custom` for production
- `LMS_JWT_SECRET` — JWT signing secret
- `SERVER_URL` — `https://agent.dedatahub.io`
- `POSTGRES_SSLMODE` — `disable` (Docker PostgreSQL has no SSL)

### 4. Start services

```bash
cd /opt/ai-service
docker compose -f docker-compose.prod.yml up -d
```

### 5. Configure Nginx

```bash
sudo cp deploy/nginx/aegra-system.conf /etc/nginx/sites-available/aegra.conf
sudo ln -sf /etc/nginx/sites-available/aegra.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Verify

```bash
curl https://agent.dedatahub.io/health
# Expected: {"status":"healthy","database":"connected",...}
```

## Operations

### View logs

```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# App only
docker logs -f aegra-app

# Deploy log
tail -f /var/log/aegra-deploy.log
```

### Restart services

```bash
cd /opt/ai-service

# Restart app only
docker compose -f docker-compose.prod.yml restart aegra

# Restart everything
docker compose -f docker-compose.prod.yml restart

# Full rebuild
docker compose -f docker-compose.prod.yml up -d --build aegra
```

### Check status

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
docker stats --no-stream
```

### Reclaim disk space

```bash
# Remove build cache (~6.8 GB)
docker builder prune -f

# Remove unused images
docker image prune -a -f
```

### Database access

```bash
docker exec -it aegra-postgres psql -U aegra -d aegra
```

### Manual deploy (without CI)

```bash
cd /opt/ai-service
bash deploy/scripts/deploy.sh
```

## Troubleshooting

### App fails with "SSL was required"

PostgreSQL in Docker doesn't have SSL. Set `POSTGRES_SSLMODE=disable` in `.env.production`.

### Containers restarting

```bash
docker logs <container-name>
```

Common causes:

- **PostgreSQL**: Volume mount path wrong (PG18 requires `/var/lib/postgresql`, not `/var/lib/postgresql/data`)
- **Redis**: Missing `REDIS_PASSWORD` in env (ensure `.env` symlink exists)
- **App**: Database connection error, missing env vars

### Deploy fails with "Permission denied"

```bash
sudo chown ubuntu:ubuntu /var/log/aegra-deploy.log
```

### Health check fails after deploy

The app takes ~10-15 seconds to start (migrations + uvicorn). The deploy script waits up to 2.5 minutes. If it still fails:

```bash
docker logs aegra-app --tail 50
```

### Nginx returns 502 Bad Gateway

The app container isn't running or not listening on port 8000:

```bash
docker ps | grep aegra-app
curl http://localhost:8000/health
```
