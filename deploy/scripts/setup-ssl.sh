#!/usr/bin/env bash
# ============================================================
# Setup SSL - Obtain Let's Encrypt certificate
# Run after DNS is pointed to this server
# Usage: sudo bash deploy/scripts/setup-ssl.sh
# ============================================================
set -euo pipefail

APP_DIR="/opt/ai-service"
DOMAIN="agent.dedatahub.io"
ADMIN_EMAIL="admin@dedatahub.com"

cd "$APP_DIR"

echo "Setting up SSL for $DOMAIN..."

# Ensure HTTP config is active for ACME challenge
cp deploy/nginx/conf.d/aegra-initial.conf deploy/nginx/conf.d/default.conf
docker compose -f docker-compose.prod.yml restart nginx
sleep 5

# Request certificate
docker compose -f docker-compose.prod.yml run --rm certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$ADMIN_EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN"

# Switch to HTTPS config
cp deploy/nginx/conf.d/aegra.conf deploy/nginx/conf.d/default.conf
docker compose -f docker-compose.prod.yml restart nginx

echo "SSL setup complete! https://$DOMAIN"
