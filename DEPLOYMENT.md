# Aegra Deployment Guide

## Quick Setup Guide for EC2 with Auto-Deploy

### 1. Initial EC2 Setup (One-time)

```bash
# Connect to your EC2
ssh -i "YOUR_SSH_KEY.pem" ubuntu@YOUR_EC2_HOST

# Download and run setup script
curl -o setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/scripts/setup_ec2.sh
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 3. Install Docker Compose
sudo apt-get install -y docker-compose-plugin

# 4. Clone repository
cd /home/ubuntu
git clone YOUR_GITHUB_REPO_URL aegra
cd aegra

# 5. Setup environment
cp .env.production .env
nano .env  # Edit with your credentials
```

### 2. Configure Environment Variables

Edit `.env` on EC2:

```bash
nano /home/ubuntu/aegra/.env
```

**Required changes:**

- Set secure `POSTGRES_PASSWORD`
- Set secure `REDIS_PASSWORD`
- Add your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Update `DATABASE_URL` with your postgres password
- Update `REDIS_URL` with your redis password

### 3. Start Application

```bash
cd /home/ubuntu/aegra
docker compose -f docker-compose.prod.yml up -d
```

Check status:

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

### 4. Setup GitHub Auto-Deploy

#### A. Add GitHub Secrets

Go to your GitHub repo → Settings → Secrets and variables → Actions

Add these secrets:

1. **EC2_SSH_KEY**: Content of your SSH private key file (keep this secret!)
2. **EC2_HOST**: Your EC2 public IP or domain name

#### B. Test Auto-Deploy

```bash
# Make any change to your code
git add .
git commit -m "test: trigger auto-deploy"
git push origin main
```

GitHub Actions will:

1. Detect the push
2. SSH into your EC2
3. Pull latest code
4. Rebuild containers
5. Restart services
6. Verify deployment

Watch progress: GitHub repo → Actions tab

### 5. Useful Commands

```bash
# View logs
docker compose -f docker-compose.prod.yml logs -f aegra

# Restart specific service
docker compose -f docker-compose.prod.yml restart aegra

# Stop all services
docker compose -f docker-compose.prod.yml down

# Start all services
docker compose -f docker-compose.prod.yml up -d

# Database backup
docker exec aegra_postgres pg_dump -U aegra_user aegra > backup_$(date +%Y%m%d).sql

# Database restore
docker exec -i aegra_postgres psql -U aegra_user aegra < backup.sql

# View container stats
docker stats

# Clean up
docker system prune -a
```

### 6. Security Checklist

- [ ] Changed default PostgreSQL password
- [ ] Changed default Redis password
- [ ] Set `DEBUG=false` in production
- [ ] Restricted EC2 security group (only ports 22, 80, 443, 8000)
- [ ] Setup SSL certificate (recommended)
- [ ] Regular backups configured
- [ ] GitHub secrets properly set

### 7. Troubleshooting

**Service won't start:**

```bash
docker compose -f docker-compose.prod.yml logs aegra
```

**Database connection issues:**

```bash
docker compose -f docker-compose.prod.yml exec postgres psql -U aegra_user -d aegra
```

**Redis connection issues:**

```bash
docker compose -f docker-compose.prod.yml exec redis redis-cli -a YOUR_REDIS_PASSWORD
```

**Auto-deploy not working:**

- Check GitHub Actions logs
- Verify EC2 security group allows inbound SSH from GitHub IPs
- Ensure SSH key is correct in GitHub secrets

### 8. Monitoring

Access your application:

- HTTP: `http://YOUR_EC2_HOST:8000`
- Health check: `http://YOUR_EC2_HOST:8000/health`

### 9. Optional: Setup Domain & SSL

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal is configured automatically
```

### 10. Cost Optimization

- Use EC2 instance with minimum 2GB RAM (t3.small or t3.medium)
- Setup CloudWatch alarms for monitoring
- Regular cleanup: `docker system prune -a --volumes`
- Consider Reserved Instances for 1-3 year commitment savings

---

## Architecture

```
┌─────────────────────────────────────────┐
│           EC2 Instance                   │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Nginx (Port 80/443)               │ │
│  └──────────┬─────────────────────────┘ │
│             │                            │
│  ┌──────────▼─────────────────────────┐ │
│  │  Aegra App (Port 8000)             │ │
│  └──────────┬─────────────────────────┘ │
│             │                            │
│  ┌──────────▼──────┐  ┌───────────────┐ │
│  │  PostgreSQL     │  │  Redis        │ │
│  │  (Port 5432)    │  │  (Port 6379)  │ │
│  └─────────────────┘  └───────────────┘ │
│                                          │
└─────────────────────────────────────────┘
            ▲
            │
     GitHub Actions
     (Auto Deploy)
```
