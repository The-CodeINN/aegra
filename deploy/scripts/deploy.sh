#!/usr/bin/env bash
# ============================================================
# Safe Deploy Script - Automatic rollback on failure
# Called by GitHub Actions via SSH
# Usage: bash /opt/ai-service/deploy/scripts/deploy.sh
#
# Strategy:
#   1. Tag the running container image as a rollback candidate
#   2. Pull latest code
#   3. Build new image  ← production still runs the OLD image here
#   4. Swap to new container
#   5. Wait for health check
#   6. PASS → clean up rollback image
#      FAIL → restore previous image automatically
# ============================================================
set -euo pipefail

APP_DIR="/opt/ai-service"
COMPOSE_FILE="docker-compose.prod.yml"
SERVICE="aegra"
CONTAINER_NAME="aegra-app"
ROLLBACK_TAG="aegra-rollback:previous"
LOG_FILE="/var/log/aegra-deploy.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# Derive the image name docker compose assigns to this service
# (compose project name comes from the directory: /opt/ai-service → ai-service)
get_compose_image() {
    docker compose -f "$COMPOSE_FILE" config --format json 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('name','ai-service'))" 2>/dev/null \
        || basename "$APP_DIR" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]-'
}

restore_previous() {
    local reason="${1:-unknown}"
    log "!!"
    log "!! DEPLOY FAILED: $reason"
    log "!! Initiating automatic rollback to previous image..."

    if ! docker image inspect "$ROLLBACK_TAG" > /dev/null 2>&1; then
        log "!! No rollback image found — production may be down. Manual intervention required!"
        return 1
    fi

    local compose_image
    compose_image="$(get_compose_image)-${SERVICE}"
    docker tag "$ROLLBACK_TAG" "$compose_image"
    docker compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate "$SERVICE" 2>&1 | tee -a "$LOG_FILE"

    local retry=0
    while [ $retry -lt 12 ]; do
        if docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" \
               curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log "!! Rollback successful — previous version is live"
            return 0
        fi
        retry=$((retry + 1))
        sleep 5
    done
    log "!! WARNING: Rollback health check also failed — manual intervention required!"
    return 1
}

log "============================================"
log "  Safe Deploy: ai-service"
log "============================================"

cd "$APP_DIR"

# ─── [1/6] Save current image for rollback ────────────────
log "[1/6] Saving current image for rollback..."
CURRENT_IMAGE=$(docker inspect "$CONTAINER_NAME" --format '{{.Image}}' 2>/dev/null || echo "")
ROLLBACK_SAVED=false
if [ -n "$CURRENT_IMAGE" ]; then
    docker tag "$CURRENT_IMAGE" "$ROLLBACK_TAG"
    ROLLBACK_SAVED=true
    log "  Rollback image saved: $ROLLBACK_TAG"
else
    log "  No running container found — first deploy, rollback unavailable"
fi

# ─── [2/6] Pull latest code ───────────────────────────────
log "[2/6] Pulling latest code..."
git fetch origin main
git reset --hard origin/main

# ─── [3/6] Build new image ────────────────────────────────
# NOTE: production still serves traffic from the OLD image during this step.
log "[3/6] Building new image (production is running previous version)..."
if ! docker compose -f "$COMPOSE_FILE" build --no-cache "$SERVICE" 2>&1 | tee -a "$LOG_FILE"; then
    log "!! Build failed — production is untouched, no swap was attempted"
    exit 1
fi

# ─── [4/6] Swap to new container ──────────────────────────
log "[4/6] Swapping to new container..."
docker compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate "$SERVICE"

# ─── [5/6] Health check ───────────────────────────────────
log "[5/6] Waiting for health check..."
MAX_RETRIES=60   # 5 minutes total
RETRY=0
HEALTHY=false
while [ $RETRY -lt $MAX_RETRIES ]; do
    if docker compose -f "$COMPOSE_FILE" exec -T "$SERVICE" \
           curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        HEALTHY=true
        log "  Service healthy after $((RETRY * 5))s"
        break
    fi
    RETRY=$((RETRY + 1))
    log "  Waiting... ($RETRY/$MAX_RETRIES)"
    sleep 5
done

if [ "$HEALTHY" != "true" ]; then
    if [ "$ROLLBACK_SAVED" = "true" ]; then
        restore_previous "Health check timed out after ${MAX_RETRIES} retries ($((MAX_RETRIES * 5))s)"
    else
        log "!! Health check failed and no rollback image available!"
    fi
    # Show last 50 lines of container logs for diagnostics
    docker compose -f "$COMPOSE_FILE" logs --tail=50 "$SERVICE" 2>&1 | tee -a "$LOG_FILE"
    exit 1
fi

# ─── [6/6] Cleanup ────────────────────────────────────────
log "[6/6] Cleaning up old images..."
docker rmi "$ROLLBACK_TAG" 2>/dev/null || true
docker image prune -f

log ""
log "============================================"
log "  Deploy complete! $(date)"
log "============================================"
