---
title: Production Deployment Guide
description: Complete guide to deploying Automatos AI to production with Docker, security hardening, monitoring, and scaling
---

# 🚀 Production Deployment Guide

*Deploy Automatos AI to production with confidence*

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Production Architecture](#production-architecture)
4. [Server Setup](#server-setup)
5. [Docker Deployment](#docker-deployment)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Deployment Options

| Option | Best For | Complexity | Cost |
|--------|----------|------------|------|
| **Single Server** | Small teams, MVP | Low | $50-100/mo |
| **Multi-Server** | Production, scaling | Medium | $200-500/mo |
| **Kubernetes** | Enterprise, high scale | High | $500+/mo |
| **Cloud-Managed** | Minimal ops overhead | Low | $300-800/mo |

This guide focuses on **single server Docker deployment** - the most common production setup.

---

## Prerequisites

### Required Services

- [x] **Server**: Ubuntu 22.04 LTS (minimum 4 CPU, 8GB RAM)
- [x] **Domain**: SSL certificate (Let's Encrypt)
- [x] **DNS**: A records pointing to server
- [x] **SSH Access**: Key-based authentication
- [x] **Docker**: Version 24.0+
- [x] **Docker Compose**: Version 2.20+

### API Keys Required

- [x] OpenAI API key
- [x] Anthropic API key (optional)
- [x] HuggingFace token (optional)

---

## Production Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXTERNAL                                                        │
│  ┌────────────────┐                                              │
│  │   Internet     │                                              │
│  └────────┬───────┘                                              │
│           │                                                       │
│           ▼                                                       │
│  ┌────────────────┐         443/HTTPS                            │
│  │  Nginx Proxy   │◄────────────────────────                    │
│  │  (SSL/TLS)     │                                              │
│  └────────┬───────┘                                              │
│           │                                                       │
│     ┌─────┴─────┬────────────┬──────────┐                       │
│     │           │            │          │                        │
│     ▼           ▼            ▼          ▼                        │
│  ┌──────┐  ┌────────┐  ┌────────┐  ┌────────┐                  │
│  │ API  │  │Frontend│  │Grafana │  │ Adminer│                  │
│  │ :8000│  │ :3000  │  │ :3001  │  │ :8080  │                  │
│  └──┬───┘  └────────┘  └────────┘  └────────┘                  │
│     │                                                             │
│     │  Docker Network (172.20.0.0/16)                           │
│     │                                                             │
│     ▼                                                             │
│  ┌──────────────────────────────────────────┐                   │
│  │           Data Layer                     │                   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │                   │
│  │  │PostgreSQL│  │  Redis   │  │ Logs   │ │                   │
│  │  │  :5432   │  │  :6379   │  │ Volume │ │                   │
│  │  └──────────┘  └──────────┘  └────────┘ │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  VOLUMES (Persistent Storage)                                    │
│  /var/lib/docker/volumes/automatos_*                            │
│  - postgres_data                                                 │
│  - redis_data                                                    │
│  - backend_logs                                                  │
│  - nginx_ssl                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Server Setup

### Step 1: Provision Server

**Recommended Providers**:
- DigitalOcean ($40/mo for 4 CPU, 8GB RAM)
- Linode ($36/mo for 4 CPU, 8GB RAM)
- AWS EC2 (t3.large ~$60/mo)
- Hetzner ($30/mo - best value)

**Initial Setup**:
```bash
# SSH into server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install essentials
apt install -y curl git vim htop ufw fail2ban

# Create automatos user
adduser automatos
usermod -aG sudo automatos
```

### Step 2: Configure Firewall

```bash
# Enable UFW firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw enable

# Verify
ufw status
```

### Step 3: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Add user to docker group
usermod -aG docker automatos

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Verify
docker --version
docker-compose --version
```

### Step 4: Setup SSL (Let's Encrypt)

```bash
# Install Certbot
apt install -y certbot python3-certbot-nginx

# Generate certificate
certbot certonly --standalone -d api.automatos.app -d app.automatos.app

# Verify
ls -la /etc/letsencrypt/live/api.automatos.app/
```

---

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.9'

services:
  # PostgreSQL Database
  postgres:
    image: pgvector/pgvector:pg16
    container_name: Automatos_postgres
    restart: always
    environment:
      POSTGRES_DB: orchestrator_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: '-c max_connections=200'
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - automatos_network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: Automatos_redis
    restart: always
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    networks:
      - automatos_network

  # Backend API
  backend_api:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile.prod
    container_name: Automatos_backend_api
    restart: always
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      # Database
      DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/orchestrator_db
      
      # Redis
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      
      # API Keys
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      
      # Security
      SECRET_KEY: ${SECRET_KEY}
      API_KEY: ${API_KEY}
      
      # Environment
      ENVIRONMENT: production
      LOG_LEVEL: INFO
      
      # Workers
      WORKERS: 4
    volumes:
      - backend_logs:/app/logs
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - automatos_network

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
      args:
        NEXT_PUBLIC_API_URL: https://api.automatos.app
    container_name: Automatos_frontend
    restart: always
    depends_on:
      - backend_api
    environment:
      NODE_ENV: production
      NEXT_PUBLIC_API_URL: https://api.automatos.app
    ports:
      - "3000:3000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - automatos_network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: Automatos_nginx
    restart: always
    depends_on:
      - backend_api
      - frontend
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    networks:
      - automatos_network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  backend_logs:
    driver: local

networks:
  automatos_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=frontend_limit:10m rate=200r/s;
    
    # Upstream servers
    upstream backend_api {
        server backend_api:8000;
    }
    
    upstream frontend {
        server frontend:3000;
    }
    
    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name api.automatos.app app.automatos.app;
        return 301 https://$server_name$request_uri;
    }
    
    # API Server
    server {
        listen 443 ssl http2;
        server_name api.automatos.app;
        
        # SSL Configuration
        ssl_certificate /etc/letsencrypt/live/api.automatos.app/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/api.automatos.app/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        
        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Logging
        access_log /var/log/nginx/api_access.log;
        error_log /var/log/nginx/api_error.log;
        
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
        
        # Proxy settings
        location / {
            proxy_pass http://backend_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        # WebSocket support
        location /ws {
            proxy_pass http://backend_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 86400;
        }
    }
    
    # Frontend Server
    server {
        listen 443 ssl http2;
        server_name app.automatos.app;
        
        # SSL Configuration
        ssl_certificate /etc/letsencrypt/live/app.automatos.app/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/app.automatos.app/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        
        # Security Headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        
        # Logging
        access_log /var/log/nginx/frontend_access.log;
        error_log /var/log/nginx/frontend_error.log;
        
        # Rate limiting
        limit_req zone=frontend_limit burst=50 nodelay;
        
        # Proxy settings
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }
        
        # Next.js specific
        location /_next/static {
            proxy_pass http://frontend;
            proxy_cache_valid 60m;
            add_header Cache-Control "public, max-age=3600, immutable";
        }
    }
}
```

### Environment Variables

Create `.env.prod`:

```bash
# Database
POSTGRES_PASSWORD=your_super_secure_postgres_password_here
DATABASE_URL=postgresql://postgres:your_super_secure_postgres_password_here@postgres:5432/orchestrator_db

# Redis
REDIS_PASSWORD=your_super_secure_redis_password_here

# API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Security
SECRET_KEY=your-256-bit-secret-key-here
API_KEY=your-internal-api-key-here

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Deployment Commands

```bash
# Clone repository
cd /home/automatos
git clone https://github.com/your-org/automatos-ai.git
cd automatos-ai

# Configure environment
cp .env.prod .env
nano .env  # Edit with your values

# Build and start services
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend_api
```

---

## Security Hardening

### 1. SSH Hardening

```bash
# Disable password authentication
nano /etc/ssh/sshd_config

# Set these values:
PasswordAuthentication no
PermitRootLogin no
PubkeyAuthentication yes

# Restart SSH
systemctl restart sshd
```

### 2. Fail2Ban Configuration

```bash
# Configure fail2ban for Nginx
cat > /etc/fail2ban/jail.local <<EOF
[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/*error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/*error.log
maxretry = 10
bantime = 3600
EOF

# Restart fail2ban
systemctl restart fail2ban
```

### 3. Database Security

```sql
-- Create read-only user for monitoring
CREATE USER monitoring WITH PASSWORD 'monitoring_password';
GRANT CONNECT ON DATABASE orchestrator_db TO monitoring;
GRANT USAGE ON SCHEMA public TO monitoring;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring;

-- Revoke unnecessary privileges
REVOKE ALL ON SCHEMA public FROM PUBLIC;
```

### 4. API Key Rotation

```bash
# Generate new secure keys
openssl rand -hex 32  # SECRET_KEY
openssl rand -hex 24  # API_KEY

# Update .env
nano .env

# Restart services
docker-compose -f docker-compose.prod.yml restart
```

---

## Monitoring & Logging

### Setup Grafana & Prometheus

Add to `docker-compose.prod.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: Automatos_prometheus
    restart: always
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    networks:
      - automatos_network

  grafana:
    image: grafana/grafana:latest
    container_name: Automatos_grafana
    restart: always
    depends_on:
      - prometheus
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3001:3000"
    networks:
      - automatos_network
```

### Log Aggregation

```bash
# View all container logs
docker-compose -f docker-compose.prod.yml logs

# Follow specific service
docker-compose -f docker-compose.prod.yml logs -f backend_api

# Export logs
docker logs Automatos_backend_api > backend_$(date +%Y%m%d).log
```

---

## Backup & Recovery

### Automated Database Backups

Create `/home/automatos/backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/home/automatos/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="orchestrator_db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec Automatos_postgres pg_dump -U postgres $DB_NAME | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Backup Redis
docker exec Automatos_redis redis-cli -a $REDIS_PASSWORD --rdb /data/dump.rdb
docker cp Automatos_redis:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete

# Upload to S3 (optional)
# aws s3 cp $BACKUP_DIR s3://your-bucket/automatos-backups/ --recursive

echo "Backup completed: $DATE"
```

### Schedule Backups

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /home/automatos/backup.sh >> /home/automatos/backup.log 2>&1
```

### Restore from Backup

```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore PostgreSQL
gunzip < /home/automatos/backups/postgres_20250115_020000.sql.gz | \
  docker exec -i Automatos_postgres psql -U postgres orchestrator_db

# Restore Redis
docker cp /home/automatos/backups/redis_20250115_020000.rdb Automatos_redis:/data/dump.rdb
docker-compose -f docker-compose.prod.yml restart redis

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

---

## Scaling

### Vertical Scaling (Single Server)

**Upgrade server resources**:
```bash
# Increase CPU/RAM on cloud provider
# Restart services
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

### Horizontal Scaling (Multiple Servers)

**Load Balancer Setup**:
```
            ┌─────────────┐
            │Load Balancer│
            └──────┬──────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    ┌──────┐  ┌──────┐  ┌──────┐
    │API #1│  │API #2│  │API #3│
    └───┬──┘  └───┬──┘  └───┬──┘
        └─────────┼──────────┘
                  │
          ┌───────┴───────┐
          │               │
          ▼               ▼
      ┌────────┐      ┌───────┐
      │Database│      │ Redis │
      │(Primary)│      │Cluster│
      └────────┘      └───────┘
```

### Database Replication

```yaml
  postgres_primary:
    image: pgvector/pgvector:pg16
    # ... primary config ...

  postgres_replica:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PRIMARY_HOST: postgres_primary
    # ... replica config ...
```

---

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check container status
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs backend_api

# Check health
docker inspect Automatos_postgres | grep Health -A 10
```

#### Database Connection Errors

```bash
# Verify PostgreSQL is running
docker exec Automatos_postgres psql -U postgres -c "SELECT 1"

# Check connection from backend
docker exec Automatos_backend_api curl postgres:5432

# Verify environment variables
docker exec Automatos_backend_api env | grep DATABASE_URL
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Limit container memory
docker-compose -f docker-compose.prod.yml config

# Add to service:
deploy:
  resources:
    limits:
      memory: 2G
```

### Health Checks

```bash
# API health
curl https://api.automatos.app/health

# Database health
docker exec Automatos_postgres pg_isready -U postgres

# Redis health
docker exec Automatos_redis redis-cli -a $REDIS_PASSWORD ping
```

---

## Performance Optimization

### Database Tuning

```sql
-- PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET max_connections = '200';

-- Restart PostgreSQL
SELECT pg_reload_conf();
```

### Redis Optimization

```bash
# Increase memory limit
docker-compose -f docker-compose.prod.yml up -d redis --memory=4g

# Enable persistence
redis-cli -a $REDIS_PASSWORD CONFIG SET save "900 1 300 10 60 10000"
```

---

## Maintenance

### Update Automatos

```bash
# Pull latest code
cd /home/automatos/automatos-ai
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker exec Automatos_backend_api alembic upgrade head
```

### SSL Certificate Renewal

```bash
# Renew certificates (automatic with certbot)
certbot renew

# Or manually
certbot renew --force-renewal

# Reload Nginx
docker-compose -f docker-compose.prod.yml restart nginx
```

---

## Next Steps

1. **📊 [Monitoring Guide](MONITORING_GUIDE.md)** - Advanced monitoring
2. **🔐 [Security Guide](security.md)** - Security best practices
3. **📈 [Scaling Guide](SCALING_GUIDE.md)** - Advanced scaling
4. **🛠️ [Developer Guide](DEVELOPER_GUIDE.md)** - Local development

---

**Built with ❤️ for production-grade deployments**

*Last updated: January 2025*

