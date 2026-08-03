#!/usr/bin/env bash
# ============================================================
# EC2 Cleanup Script - Remove ALL old ai-service components
# Run as: sudo bash deploy/scripts/ec2-cleanup.sh
# ============================================================
set -euo pipefail

echo "============================================"
echo "  EC2 Cleanup - Removing Old Services"
echo "============================================"
echo ""

# ─── Stop and remove Docker containers ───────────────────────
echo "[1/6] Stopping and removing Docker containers..."
if command -v docker &>/dev/null; then
    # Stop all containers related to our project
    docker ps -a --filter "name=aegra" --filter "name=postgres" --filter "name=redis" -q | \
        xargs -r docker stop 2>/dev/null || true
    docker ps -a --filter "name=aegra" --filter "name=postgres" --filter "name=redis" -q | \
        xargs -r docker rm -f 2>/dev/null || true

    # Also check for any compose projects
    if [ -f /opt/ai-service/docker-compose.yml ]; then
        cd /opt/ai-service && docker compose down -v --remove-orphans 2>/dev/null || true
    fi
    if [ -f /opt/ai-service/docker-compose.prod.yml ]; then
        cd /opt/ai-service && docker compose -f docker-compose.prod.yml down -v --remove-orphans 2>/dev/null || true
    fi

    echo "  -> Docker containers cleaned"
else
    echo "  -> Docker not installed, skipping"
fi

# ─── Remove Docker volumes ───────────────────────────────────
echo "[2/6] Removing Docker volumes..."
if command -v docker &>/dev/null; then
    docker volume ls -q --filter "name=postgres" | xargs -r docker volume rm 2>/dev/null || true
    docker volume ls -q --filter "name=redis" | xargs -r docker volume rm 2>/dev/null || true
    docker volume ls -q --filter "name=aegra" | xargs -r docker volume rm 2>/dev/null || true
    docker volume ls -q --filter "name=certbot" | xargs -r docker volume rm 2>/dev/null || true
    # Prune dangling volumes
    docker volume prune -f 2>/dev/null || true
    echo "  -> Docker volumes cleaned"
fi

# ─── Remove Docker images ────────────────────────────────────
echo "[3/6] Removing Docker images..."
if command -v docker &>/dev/null; then
    docker image prune -af 2>/dev/null || true
    echo "  -> Docker images cleaned"
fi

# ─── Stop systemd services (if installed natively) ───────────
echo "[4/6] Stopping systemd services..."
for svc in postgresql redis redis-server aegra ai-service; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        sudo systemctl stop "$svc" 2>/dev/null || true
        sudo systemctl disable "$svc" 2>/dev/null || true
        echo "  -> Stopped and disabled $svc"
    fi
done

# ─── Remove native packages (if installed) ───────────────────
echo "[5/6] Removing native packages..."
if dpkg -l | grep -q postgresql 2>/dev/null; then
    sudo apt-get purge -y 'postgresql*' 2>/dev/null || true
    echo "  -> PostgreSQL packages removed"
fi
if dpkg -l | grep -q redis 2>/dev/null; then
    sudo apt-get purge -y 'redis*' 2>/dev/null || true
    echo "  -> Redis packages removed"
fi
sudo apt-get autoremove -y 2>/dev/null || true

# ─── Clean old data directories ─────────────────────────────
echo "[6/6] Cleaning old data directories..."
sudo rm -rf /var/lib/postgresql 2>/dev/null || true
sudo rm -rf /var/lib/redis 2>/dev/null || true
sudo rm -rf /opt/ai-service 2>/dev/null || true
echo "  -> Old data directories removed"

# ─── Remove Docker networks ─────────────────────────────────
if command -v docker &>/dev/null; then
    docker network prune -f 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  Cleanup Complete! EC2 is now clean."
echo "============================================"
