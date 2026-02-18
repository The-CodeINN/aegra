#!/usr/bin/env bash
# ============================================================
# EC2 Bootstrap Script - First-time setup for a clean EC2
# Run as: sudo bash deploy/scripts/ec2-bootstrap.sh
#
# This script:
#  1. Installs Docker & Docker Compose
#  2. Configures firewall (UFW)
#  3. Sets up the app directory
#  4. Clones the repo
#  5. Obtains SSL certificate
#  6. Starts all services
# ============================================================
set -euo pipefail

APP_DIR="/opt/ai-service"
DOMAIN="agent.dedatahub.io"
REPO="git@github.com:dedatahub/ai-service.git"
BRANCH="main"
ADMIN_EMAIL="admin@dedatahub.com"

echo "============================================"
echo "  EC2 Bootstrap - ai-service"
echo "============================================"
echo ""

# ─── 1. System updates ──────────────────────────────────────
echo "[1/8] Updating system packages..."
apt-get update -y
apt-get upgrade -y
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw \
    htop \
    unzip \
    jq

# ─── 2. Install Docker ──────────────────────────────────────
echo "[2/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    # Add ubuntu user to docker group
    usermod -aG docker ubuntu
    echo "  -> Docker installed"
else
    echo "  -> Docker already installed"
fi

# Verify Docker Compose (comes with Docker now)
docker compose version

# ─── 3. Configure firewall ──────────────────────────────────
echo "[3/8] Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable
echo "  -> Firewall configured (SSH, HTTP, HTTPS only)"

# ─── 4. Setup SSH deploy key ────────────────────────────────
echo "[4/8] Setting up SSH for GitHub..."
# The deploy key should be added to the repo's Settings > Deploy keys
if [ ! -f /home/ubuntu/.ssh/github_deploy_key ]; then
    echo "  !! No deploy key found at /home/ubuntu/.ssh/github_deploy_key"
    echo "  !! You need to either:"
    echo "     a) Copy a deploy key:  scp -i key.pem deploy_key ubuntu@host:~/.ssh/github_deploy_key"
    echo "     b) Or generate one:    ssh-keygen -t ed25519 -f /home/ubuntu/.ssh/github_deploy_key -N ''"
    echo "     Then add the .pub key to GitHub repo Settings > Deploy Keys"

    # Generate one if it doesn't exist
    sudo -u ubuntu ssh-keygen -t ed25519 -f /home/ubuntu/.ssh/github_deploy_key -N "" -C "deploy@ec2"
    echo ""
    echo "  >> Add this PUBLIC key to https://github.com/dedatahub/ai-service/settings/keys :"
    echo ""
    cat /home/ubuntu/.ssh/github_deploy_key.pub
    echo ""
fi

# Configure SSH to use deploy key for github
cat > /home/ubuntu/.ssh/config << 'SSHEOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_deploy_key
    StrictHostKeyChecking accept-new
SSHEOF
chown ubuntu:ubuntu /home/ubuntu/.ssh/config
chmod 600 /home/ubuntu/.ssh/config

# ─── 5. Clone the repository ────────────────────────────────
echo "[5/8] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "  -> $APP_DIR already exists, pulling latest..."
    cd "$APP_DIR"
    sudo -u ubuntu git pull origin "$BRANCH"
else
    sudo -u ubuntu git clone -b "$BRANCH" "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

# ─── 6. Setup environment file ──────────────────────────────
echo "[6/8] Setting up environment..."
if [ ! -f "$APP_DIR/.env.production" ]; then
    echo "  !! No .env.production found!"
    echo "  !! Copy your production env file to $APP_DIR/.env.production"
    echo "  !! You can use .env.production.template as a starting point"
    echo ""
    echo "  Run: scp -i key.pem .env.production ubuntu@$DOMAIN:/opt/ai-service/.env.production"
else
    echo "  -> .env.production found"
fi

# ─── 7. SSL Certificate ─────────────────────────────────────
echo "[7/8] Setting up SSL certificate..."

# First start with HTTP-only config to get cert
cp "$APP_DIR/deploy/nginx/conf.d/aegra-initial.conf" "$APP_DIR/deploy/nginx/conf.d/default.conf"

# Start just nginx + app (without SSL) for ACME challenge
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml up -d postgres redis aegra nginx

echo "  -> Waiting for services to start..."
sleep 15

# Request certificate
docker compose -f docker-compose.prod.yml run --rm certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$ADMIN_EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" || {
        echo "  !! SSL certificate request failed."
        echo "  !! Make sure DNS for $DOMAIN points to this server's IP."
        echo "  !! You can retry later with: deploy/scripts/setup-ssl.sh"
    }

# Switch to full SSL config
if [ -f "/var/lib/docker/volumes/$(docker volume ls -q --filter name=certbot_certs)/letsencrypt/live/$DOMAIN/fullchain.pem" ] 2>/dev/null || \
   docker compose -f docker-compose.prod.yml exec certbot test -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null; then
    echo "  -> SSL certificate obtained! Switching to HTTPS config..."
    cp "$APP_DIR/deploy/nginx/conf.d/aegra.conf" "$APP_DIR/deploy/nginx/conf.d/default.conf"
    docker compose -f docker-compose.prod.yml restart nginx
else
    echo "  -> Running in HTTP-only mode. Run setup-ssl.sh when DNS is ready."
fi

# ─── 8. Setup cron for cert renewal ─────────────────────────
echo "[8/8] Setting up certbot auto-renewal cron..."
(crontab -l 2>/dev/null; echo "0 3 * * * cd $APP_DIR && docker compose -f docker-compose.prod.yml run --rm certbot renew && docker compose -f docker-compose.prod.yml exec nginx nginx -s reload") | sort -u | crontab -

echo ""
echo "============================================"
echo "  Bootstrap Complete!"
echo "============================================"
echo ""
echo "  Service:  https://$DOMAIN"
echo "  App Dir:  $APP_DIR"
echo ""
echo "  Next steps:"
echo "  1. Ensure .env.production is configured"
echo "  2. Add the deploy key to GitHub"
echo "  3. Add GitHub Actions secrets (see README)"
echo "  4. Restart: cd $APP_DIR && docker compose -f docker-compose.prod.yml up -d"
echo ""
