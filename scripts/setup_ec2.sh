#!/bin/bash
# EC2 Initial Setup Script
# Run this once on your EC2 instance after connecting

set -e

echo "=========================================="
echo "Aegra EC2 Setup Script"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
rm get-docker.sh

# Install Docker Compose
echo "Installing Docker Compose..."
sudo apt-get install -y docker-compose-plugin

# Install Git
echo "Installing Git..."
sudo apt-get install -y git

# Create application directory
echo "Setting up application directory..."
cd /home/ubuntu
if [ ! -d "aegra" ]; then
    echo "Enter your GitHub repository URL:"
    read REPO_URL
    git clone $REPO_URL aegra
fi

cd aegra

# Setup environment file
echo "Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.production .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your actual credentials!"
    echo "Run: nano .env"
    echo ""
fi

# Generate secure passwords
POSTGRES_PASS=$(openssl rand -base64 32)
REDIS_PASS=$(openssl rand -base64 32)

echo ""
echo "=========================================="
echo "Generated Secure Passwords:"
echo "=========================================="
echo "PostgreSQL Password: $POSTGRES_PASS"
echo "Redis Password: $REDIS_PASS"
echo ""
echo "Save these passwords! They will be used in .env"
echo ""

# Setup systemd service for auto-start
echo "Setting up systemd service..."
sudo tee /etc/systemd/system/aegra.service > /dev/null <<EOF
[Unit]
Description=Aegra Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/aegra
ExecStart=/usr/bin/docker compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.prod.yml down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aegra.service

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Add your API keys and secure passwords"
echo "3. Start the application: docker compose -f docker-compose.prod.yml up -d"
echo "4. Check logs: docker compose -f docker-compose.prod.yml logs -f"
echo "5. Access your app at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "To enable GitHub auto-deploy, add these secrets to your GitHub repo:"
echo "  EC2_SSH_KEY: (your private key content)"
echo "  EC2_HOST: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
