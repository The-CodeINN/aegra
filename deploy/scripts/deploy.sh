#!/usr/bin/env bash
# ============================================================
# Deploy Script - Runs on EC2 during each deployment
# Called by GitHub Actions via SSH
# Usage: bash /opt/ai-service/deploy/scripts/deploy.sh
# ============================================================
set -euo pipefail

APP_DIR="/opt/ai-service"
COMPOSE_FILE="docker-compose.prod.yml"
LOG_FILE="/var/log/aegra-deploy.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "============================================"
log "  Deploying ai-service"
log "============================================"

cd "$APP_DIR"

# ─── Pull latest code ───────────────────────────────────────
log "[1/5] Pulling latest code..."
git fetch origin main
git reset --hard origin/main

# ─── Rebuild the app container only ──────────────────────────
log "[2/5] Building new image..."
docker compose -f "$COMPOSE_FILE" build --no-cache aegra

# ─── Rolling restart (minimal downtime) ─────────────────────
log "[3/5] Restarting aegra service..."
docker compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate aegra

# ─── Wait for health check ──────────────────────────────────
log "[4/5] Waiting for service to be healthy..."
MAX_RETRIES=30
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if docker compose -f "$COMPOSE_FILE" exec -T aegra curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log "  -> Service is healthy!"
        break
    fi
    RETRY=$((RETRY + 1))
    log "  -> Waiting... ($RETRY/$MAX_RETRIES)"
    sleep 5
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    log "  !! Health check failed after ${MAX_RETRIES} retries"
    log "  !! Check logs: docker compose -f $COMPOSE_FILE logs aegra"
    exit 1
fi

# ─── Cleanup old images ─────────────────────────────────────
log "[5/5] Cleaning up old images..."
docker image prune -f

log ""
log "============================================"
log "  Deploy complete! $(date)"
log "============================================"
