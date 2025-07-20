
# Deployment Guide for Automatos AI

This guide provides step-by-step instructions for deploying the Enhanced Two-Tiered Multi-Agent Orchestration System, including both backend services and the Next.js frontend.

## Deployment Options

### Option 1: Single Server Deployment
Deploy both backend and frontend on the same server (recommended for development/testing).

### Option 2: Multi-Server Deployment  
Deploy backend and frontend on separate servers (recommended for production).

## Backend Deployment

This section covers deploying the core backend services (API, Database, Redis, Monitoring).

## Prerequisites

### Server Requirements

- **Operating System**: Ubuntu 20.04 LTS or later
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 20GB, Recommended 50GB+
- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Network**: Public IP with SSH access

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- SSH access to target servers

### Access Requirements

- SSH key access to your target server(s)
- OpenAI API key
- Domain DNS configuration (if using custom domain)

## Pre-Deployment Setup

### 1. Server Preparation

Connect to your target server:

```bash
ssh root@your-server-ip
```

Update the system:

```bash
apt update && apt upgrade -y
```

Install required packages:

```bash
apt install -y git curl wget unzip
```

### 2. Docker Installation

Install Docker Engine:

```bash
# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
apt update
apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add user to docker group (optional)
usermod -aG docker $USER
```

Verify installation:

```bash
docker --version
docker-compose --version
```

### 3. SSH Key Setup

Generate SSH keys for deployment (if not already available):

```bash
ssh-keygen -t ed25519 -C "orchestrator@your-server" -f /root/.ssh/orchestrator_key
```

Add the public key to authorized_keys:

```bash
cat /root/.ssh/orchestrator_key.pub >> /root/.ssh/authorized_keys
```

## Deployment Steps

### 1. Clone Repository

```bash
cd /opt
git clone https://github.com/your-org/Automatos_v2.git
cd Automatos_v2
```

### 2. Environment Configuration

Copy and configure environment variables:

```bash
cp .env.example .env
```

Edit the `.env` file with your specific configuration:

```bash
nano .env
```

**Required Configuration:**

```env
# Database Configuration
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_db_password_here

# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password_here

# API Configuration
API_KEY=your_secure_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# SSH Deployment Configuration
DEPLOY_HOST=your-server-ip
DEPLOY_PORT=22
DEPLOY_USER=root
DEPLOY_KEY_PATH=/app/keys/orchestrator_key

# Service Ports
MCP_PORT=8001
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password_here
```

### 3. SSH Key Setup for Container

Create keys directory and copy SSH keys:

```bash
mkdir -p keys
cp /root/.ssh/orchestrator_key keys/deploy_key
cp /root/.ssh/orchestrator_key.pub keys/deploy_key.pub
chmod 600 keys/deploy_key
chmod 644 keys/deploy_key.pub
```

### 4. Create Required Directories

```bash
mkdir -p logs vector_stores projects monitoring/grafana monitoring/prometheus
```

### 5. Configure Monitoring (Optional)

Create Prometheus configuration:

```bash
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'orchestrator'
    static_configs:
      - targets: ['mcp_bridge:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF
```

### 6. Deploy Services

Start the core services:

```bash
docker-compose up -d postgres redis mcp_bridge
```

Wait for services to be healthy:

```bash
docker-compose ps
```

Check logs:

```bash
docker-compose logs -f mcp_bridge
```

### 7. Verify Deployment

Test the API:

```bash
curl -H "X-API-Key: your_secure_api_key_here" http://localhost:8001/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "2.0.0",
  "uptime_seconds": 123.45
}
```

### 8. Start Monitoring (Optional)

Start monitoring services:

```bash
docker-compose --profile monitoring up -d
```

Access Grafana at `http://your-server-ip:3000` (admin/your_grafana_password)

## Frontend Deployment

This section covers deploying the Next.js frontend application.

### Prerequisites

- Docker and Docker Compose installed (same as backend)
- Backend API accessible (default: http://localhost:8001)
- Node.js 18+ (for local development)

### Frontend Deployment Steps

#### 1. Clone Repository (if not already done)

```bash
cd /opt
git clone https://github.com/your-org/Automatos_v2.git
cd Automatos_v2/app
```

#### 2. Environment Configuration

Copy and configure environment variables:

```bash
cp env.example .env
```

Edit the `.env` file with your specific configuration:

```bash
nano .env
```

**Required Configuration:**

```env
# Frontend Configuration
FRONTEND_PORT=3000

# API Configuration
API_BASE_URL=http://localhost:8001  # Point to your backend API

# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-here

# Database (if needed for NextAuth)
DATABASE_URL=postgresql://postgres:password@localhost:5432/orchestrator_db

# Optional: Redis Cache
REDIS_CACHE_PORT=6380
REDIS_CACHE_PASSWORD=your_redis_password
```

**For Multi-Server Deployment:**
- Set `API_BASE_URL` to your backend server's public IP/domain
- Set `NEXTAUTH_URL` to your frontend server's public IP/domain

#### 3. Deploy Frontend

Run the deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

#### 4. Verify Frontend Deployment

Test the frontend:

```bash
curl http://localhost:3000
```

Expected response: HTML content from Next.js application.

#### 5. Optional: Enable Redis Cache

To enable Redis caching for the frontend:

```bash
docker-compose --profile cache up -d
```

### Frontend Firewall Configuration

Configure UFW firewall for frontend:

```bash
# Enable UFW
ufw enable

# Allow SSH
ufw allow 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow frontend service
ufw allow 3000/tcp  # Frontend
ufw allow 6380/tcp  # Redis cache (optional)

# Check status
ufw status
```

### Frontend SSL/TLS Configuration (Recommended)

#### Using Let's Encrypt with Nginx

Install Nginx:

```bash
apt install -y nginx certbot python3-certbot-nginx
```

Create Nginx configuration for frontend:

```bash
cat > /etc/nginx/sites-available/frontend << 'EOF'
server {
    listen 80;
    server_name your-frontend-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass $http_upgrade;
    }
}
EOF
```

Enable the site:

```bash
ln -s /etc/nginx/sites-available/frontend /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

Obtain SSL certificate:

```bash
certbot --nginx -d your-frontend-domain.com
```

## Backend Firewall Configuration

Configure UFW firewall:

```bash
# Enable UFW
ufw enable

# Allow SSH
ufw allow 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow orchestrator services
ufw allow 8001/tcp  # MCP Bridge
ufw allow 3000/tcp  # Grafana (optional)
ufw allow 9090/tcp  # Prometheus (optional)

# Check status
ufw status
```

## SSL/TLS Configuration (Recommended)

### Using Let's Encrypt with Nginx

Install Nginx:

```bash
apt install -y nginx certbot python3-certbot-nginx
```

Create Nginx configuration:

```bash
cat > /etc/nginx/sites-available/orchestrator << 'EOF'
server {
    listen 80;
    server_name mcp.xplaincrypto.ai;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
```

Enable the site:

```bash
ln -s /etc/nginx/sites-available/orchestrator /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

Obtain SSL certificate:

```bash
certbot --nginx -d mcp.xplaincrypto.ai
```

## Testing the Deployment

### 1. Test Frontend (if deployed)

Access the frontend in your browser:
- **Single Server**: http://localhost:3000
- **Multi Server**: http://your-frontend-server-ip:3000

Or test via curl:

```bash
curl http://localhost:3000
```

### 2. Test Backend API

Test the API health endpoint:

```bash
curl -H "X-API-Key: your_secure_api_key_here" http://localhost:8001/health
```

### 3. Test AI Module Workflow

Create a test repository with `ai-module.yaml`:

```bash
curl -X POST http://localhost:8001/workflow \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secure_api_key_here" \
  -d '{
    "repository_url": "https://github.com/your-org/test-web-app.git"
  }'
```

### 2. Test Task Prompt Workflow

```bash
curl -X POST http://localhost:8001/workflow \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secure_api_key_here" \
  -d '{
    "repository_url": "https://github.com/your-org/simple-flask-app.git",
    "task_prompt": "Deploy a simple Flask web server with basic authentication"
  }'
```

### 3. Test SSH Command Execution

```bash
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secure_api_key_here" \
  -d '{
    "command": "docker ps",
    "security_level": "medium"
  }'
```

## Monitoring and Maintenance

### Log Management

View logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mcp_bridge

# System logs
tail -f /var/log/syslog
```

Rotate logs:

```bash
# Configure logrotate
cat > /etc/logrotate.d/orchestrator << 'EOF'
/opt/Automatos_v2/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
```

### Database Maintenance

Backup database:

```bash
docker-compose exec postgres pg_dump -U postgres orchestrator_db > backup_$(date +%Y%m%d).sql
```

Restore database:

```bash
docker-compose exec -T postgres psql -U postgres orchestrator_db < backup_20240115.sql
```

### Updates and Upgrades

Update the system:

```bash
cd /opt/Automatos_v2
git pull origin main
docker-compose build --no-cache
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker-compose logs container_name
   docker-compose ps
   ```

2. **SSH connection fails**
   ```bash
   # Check SSH key permissions
   ls -la keys/
   # Test SSH connection manually
   ssh -i keys/deploy_key root@mcp.xplaincrypto.ai
   ```

3. **Database connection issues**
   ```bash
   # Check database status
   docker-compose exec postgres pg_isready -U postgres
   # Check connection from app
   docker-compose exec mcp_bridge python -c "import psycopg2; print('DB OK')"
   ```

4. **API not responding**
   ```bash
   # Check if service is running
   curl http://localhost:8001/health
   # Check firewall
   ufw status
   # Check nginx (if using)
   nginx -t
   systemctl status nginx
   ```

### Performance Optimization

1. **Increase Docker resources**
   ```bash
   # Edit Docker daemon configuration
   nano /etc/docker/daemon.json
   ```

2. **Optimize PostgreSQL**
   ```bash
   # Tune PostgreSQL settings in docker-compose.yml
   # Add performance-related environment variables
   ```

3. **Monitor resource usage**
   ```bash
   docker stats
   htop
   df -h
   ```

## Security Hardening

### Additional Security Measures

1. **Change default ports**
2. **Implement fail2ban**
3. **Regular security updates**
4. **Monitor access logs**
5. **Use strong passwords and keys**

### Backup Strategy

1. **Database backups**: Daily automated backups
2. **Configuration backups**: Version control
3. **Log archival**: Long-term storage
4. **Disaster recovery**: Documented procedures

## Support and Maintenance

### Regular Maintenance Tasks

- **Weekly**: Check logs and system health
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Review and rotate credentials
- **Annually**: Security audit and penetration testing

## Deployment Scenarios

### Scenario 1: Single Server (Development/Testing)

Deploy everything on one server:

```bash
# 1. Deploy backend
cd /opt/Automatos_v2
docker-compose up -d

# 2. Deploy frontend
cd /opt/Automatos_v2/app
cp env.example .env
# Edit .env: API_BASE_URL=http://localhost:8001
./deploy.sh
```

**Access URLs:**
- Frontend: http://localhost:3000
- API: http://localhost:8001
- Grafana: http://localhost:3000 (if monitoring enabled)

### Scenario 2: Multi-Server (Production)

**Backend Server:**
```bash
# Deploy backend only
cd /opt/Automatos_v2
docker-compose up -d
```

**Frontend Server:**
```bash
# Deploy frontend only
cd /opt/Automatos_v2/app
cp env.example .env
# Edit .env: API_BASE_URL=http://backend-server-ip:8001
./deploy.sh
```

**Access URLs:**
- Frontend: http://frontend-server-ip:3000
- API: http://backend-server-ip:8001

### Scenario 3: Docker Swarm (Advanced)

For production deployments with high availability:

```bash
# Initialize swarm
docker swarm init

# Deploy backend stack
docker stack deploy -c docker-compose.yml backend

# Deploy frontend stack
cd app
docker stack deploy -c docker-compose.yml frontend
```

## Troubleshooting Frontend Issues

### Common Frontend Issues

1. **Frontend won't start**
   ```bash
   cd /opt/Automatos_v2/app
   docker-compose logs frontend
   docker-compose ps
   ```

2. **API connection fails**
   ```bash
   # Check API_BASE_URL in .env
   cat .env | grep API_BASE_URL
   # Test API connectivity
   curl $API_BASE_URL/health
   ```

3. **NextAuth issues**
   ```bash
   # Check NEXTAUTH_SECRET is set
   # Verify NEXTAUTH_URL matches your domain
   # Check database connection if using database adapter
   ```

4. **Build failures**
   ```bash
   # Clean and rebuild
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Contact Information

For support and issues:
- **Documentation**: Check this guide and architecture docs
- **Logs**: Always include relevant log files
- **Environment**: Provide system and configuration details
