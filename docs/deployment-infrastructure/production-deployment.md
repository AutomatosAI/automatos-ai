# Production Deployment

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)

</details>



This page covers production deployment strategies for Automatos AI, with specific guidance for Railway deployment and general best practices applicable to any cloud platform. For local development setup, see [Getting Started](#2). For Docker containerization details, see [Docker Containerization](#12.1). For environment variable reference, see [Environment Variables](#12.3).

---

## Deployment Overview

Automatos AI supports production deployment on cloud platforms using Docker containers. The system uses multi-stage Docker builds with optimized production images, centralized configuration management, and support for Railway's environment variable conventions.

**Key Production Components:**
- **Backend**: FastAPI application on port 8000 (configurable via `PORT` env var)
- **Frontend**: Next.js application on port 3000
- **PostgreSQL**: Database with pgvector extension
- **Redis**: Optional caching and pub/sub service
- **S3**: Plugin marketplace storage

Sources: [orchestrator/config.py:1-285](), [README.md:1-115]()

---

## Railway Deployment

Railway is the primary supported platform, with automatic detection of Railway-specific environment variables and deployment patterns.

### Railway Project Structure

```mermaid
graph TB
    subgraph "Railway Project"
        FE["Frontend Service<br/>(Next.js)"]
        BE["Backend Service<br/>(FastAPI)"]
        PG["PostgreSQL Plugin<br/>(Railway Managed)"]
        RD["Redis Plugin<br/>(Railway Managed)"]
    end
    
    subgraph "External Services"
        S3["AWS S3<br/>(Plugin Storage)"]
        Clerk["Clerk Auth<br/>(User Management)"]
        Composio["Composio API<br/>(Tool Integrations)"]
    end
    
    FE -->|"NEXT_PUBLIC_API_URL"| BE
    BE -->|"DATABASE_URL"| PG
    BE -->|"REDIS_URL"| RD
    BE -->|"AWS_ACCESS_KEY_ID"| S3
    FE -->|"NEXT_PUBLIC_CLERK_*"| Clerk
    BE -->|"CLERK_SECRET_KEY"| Clerk
    BE -->|"COMPOSIO_API_KEY"| Composio
```

**Railway-Specific Configuration:**

| Service | Build Command | Start Command | PORT Detection |
|---------|--------------|---------------|----------------|
| Frontend | `npm install --legacy-peer-deps && npm run build` | `npm start` | Fixed: 3000 |
| Backend | Auto (Dockerfile) | `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4` | Railway sets `$PORT` |
| PostgreSQL | Railway Plugin | - | Railway provides `DATABASE_URL` |
| Redis | Railway Plugin | - | Railway provides `REDIS_URL` |

Sources: [orchestrator/Dockerfile:73-116](), [frontend/Dockerfile:83-119](), [orchestrator/config.py:73-79]()

### Deployment Steps

#### 1. Create Railway Project

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Create new project
railway init
```

#### 2. Add Database Plugins

Add PostgreSQL and Redis plugins in Railway dashboard:
- **PostgreSQL**: Version 16 with pgvector support
- **Redis**: Version 7 with persistence

Railway automatically injects `DATABASE_URL` and `REDIS_URL` environment variables.

#### 3. Configure Environment Variables

Set these variables in Railway dashboard for backend service:

**Required:**
```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Clerk Authentication
CLERK_SECRET_KEY=sk_live_...
CLERK_JWKS_URL=https://your-clerk-domain.clerk.accounts.dev/.well-known/jwks.json

# AWS S3 (Plugin Marketplace)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
MARKETPLACE_S3_BUCKET=automatos-marketplace

# Security
API_KEY=your_secure_api_key_here

# Environment
ENVIRONMENT=production
```

**Optional:**
```bash
# CORS (Frontend Domain)
CORS_ALLOW_ORIGINS=https://your-frontend.up.railway.app,https://yourdomain.com

# Redis Configuration (if using external Redis)
REDIS_URL=redis://:password@host:6379/0

# Composio Integration
COMPOSIO_API_KEY=...
COMPOSIO_WEBHOOK_SECRET=...

# Feature Flags
ENABLE_BATCH_API=false
S3_VECTORS_ENABLED=false
```

For frontend service:

```bash
# Backend URL (Railway internal or public)
NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app

# Clerk Public Keys
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
```

Sources: [orchestrator/.env.example:1-64](), [orchestrator/config.py:28-185]()

#### 4. Deploy Services

```bash
# Deploy backend
railway up

# Deploy frontend (separate service)
railway up
```

Railway automatically builds Docker images using multi-stage Dockerfiles and deploys to production infrastructure.

---

## Production Docker Configuration

### Multi-Stage Build Architecture

Both frontend and backend use multi-stage Docker builds with separate development and production targets.

#### Backend Production Stage

```mermaid
graph LR
    Base["base Stage<br/>Python 3.11<br/>System Dependencies<br/>pip install"]
    Dev["development Stage<br/>Hot Reload<br/>Debug Tools"]
    Prod["production Stage<br/>Optimized<br/>Non-root User<br/>Minimal Image"]
    
    Base --> Dev
    Base --> Prod
```

**Key Production Optimizations:**

| Optimization | Implementation | File Reference |
|--------------|----------------|----------------|
| Minimal base image | `python:3.11-slim` | [orchestrator/Dockerfile:13]() |
| Dependency caching | `pip install --no-cache-dir` | [orchestrator/Dockerfile:33-35]() |
| Non-root user | `useradd automatos` | [orchestrator/Dockerfile:98-99]() |
| Clean temp files | Remove `__pycache__`, `.pyc` | [orchestrator/Dockerfile:94-95]() |
| Multi-worker | `--workers 4` | [orchestrator/Dockerfile:115]() |
| Health check | `curl -f http://localhost:$PORT/health` | [orchestrator/Dockerfile:107-108]() |

**Production Command:**
```dockerfile
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
```

This command:
- Uses `$PORT` environment variable (Railway requirement)
- Falls back to port 8000 if `PORT` not set
- Runs 4 worker processes for concurrency
- Binds to all interfaces (`0.0.0.0`)

Sources: [orchestrator/Dockerfile:73-116]()

#### Frontend Production Stage

```mermaid
graph LR
    Base["base Stage<br/>Node 20 Alpine<br/>npm install"]
    Builder["builder Stage<br/>npm run build<br/>NEXT_PUBLIC_* baked"]
    Prod["production Stage<br/>npm start<br/>Non-root User"]
    
    Base --> Builder
    Builder --> Prod
```

**Key Frontend Optimizations:**

| Optimization | Implementation | File Reference |
|--------------|----------------|----------------|
| Build-time env vars | `ARG NEXT_PUBLIC_*` | [frontend/Dockerfile:55-63]() |
| Production dependencies only | `npm ci --only=production` | [frontend/Dockerfile:94]() |
| Copy built artifacts | `.next`, `public`, `next.config.js` | [frontend/Dockerfile:96-100]() |
| Non-root user | `adduser nextjs` | [frontend/Dockerfile:103-105]() |
| Static optimization | Next.js SSG/ISR | Built into Next.js |

**Important:** `NEXT_PUBLIC_*` environment variables are **baked into the client bundle** at build time and cannot contain secrets. Server-side API keys must be handled via backend API routes.

Sources: [frontend/Dockerfile:51-119]()

### Security Hardening

Production images implement multiple security layers:

```mermaid
graph TB
    subgraph "Security Layers"
        User["Non-root User<br/>UID 1000"]
        Files["Read-only Filesystem<br/>(where possible)"]
        Secrets["No Secrets in Image<br/>Runtime Env Vars Only"]
        Clean["Cleaned Build Artifacts<br/>No dev dependencies"]
        Health["Health Checks<br/>Auto-restart on failure"]
    end
    
    User --> Files
    Files --> Secrets
    Secrets --> Clean
    Clean --> Health
```

**Security Checklist:**

- ✅ Non-root user for both frontend and backend
- ✅ Secrets injected at runtime (never in Dockerfile or code)
- ✅ Development dependencies removed in production
- ✅ Health checks for automatic restart on failure
- ✅ Encrypted credential storage (AES-256-GCM)
- ✅ API key authentication for backend
- ✅ Clerk JWT verification for user endpoints

Sources: [orchestrator/Dockerfile:98-115](), [frontend/Dockerfile:103-118](), [orchestrator/core/credentials/service.py:1-850]()

---

## Database Configuration

### PostgreSQL Production Setup

#### Connection Pooling

The system uses SQLAlchemy with connection pooling configured for production workloads:

```python
# Database URL format (Railway provides this)
DATABASE_URL=postgresql://user:password@host:port/database

# Connection pool settings (configured in docker-compose for reference)
max_connections=200
shared_buffers=256MB
```

**Recommended Pool Configuration:**

| Parameter | Development | Production | Notes |
|-----------|-------------|------------|-------|
| `pool_size` | 5 | 20 | Base connection pool |
| `max_overflow` | 10 | 30 | Additional connections under load |
| `pool_timeout` | 30 | 30 | Seconds to wait for connection |
| `pool_recycle` | 3600 | 1800 | Recycle connections (seconds) |

Sources: [docker-compose.yml:21-42](), [orchestrator/config.py:36-42]()

#### pgvector Extension

Production PostgreSQL must have the `pgvector` extension enabled for embedding storage:

```sql
-- Verify pgvector is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Enable extension (auto-initialized in Railway via init script)
CREATE EXTENSION IF NOT EXISTS vector;
```

Railway PostgreSQL plugin includes pgvector by default when using `pgvector/pgvector:pg16` image.

#### Backup Strategy

**Automated Backups:**
- Railway provides automatic daily backups (retained 7 days)
- Configure additional backup retention via Railway dashboard

**Manual Backups:**
```bash
# Export database dump
railway run pg_dump -Fc orchestrator_db > backup.dump

# Restore from dump
railway run pg_restore -d orchestrator_db backup.dump
```

Sources: [docker-compose.yml:21-42]()

### Redis Production Setup

#### Connection Configuration

Redis is optional but recommended for production to enable:
- Plugin content caching
- Real-time workflow updates (pub/sub)
- Composio app/action caching

**Railway REDIS_URL Format:**
```bash
# Railway provides this automatically
REDIS_URL=redis://:password@host:port/0
```

The `RedisClient` class automatically parses `REDIS_URL` with fallback to individual variables:

```mermaid
graph TD
    Start["get_redis_client()"]
    CheckURL{"REDIS_URL<br/>set?"}
    ParseURL["Parse URL<br/>host, port, password, db"]
    CheckVars{"Individual vars<br/>set?"}
    ParseVars["Use REDIS_HOST<br/>REDIS_PORT<br/>REDIS_PASSWORD"]
    NoRedis["Return None<br/>(Redis disabled)"]
    Init["init_redis_client()"]
    Return["Return client"]
    
    Start --> CheckURL
    CheckURL -->|Yes| ParseURL
    CheckURL -->|No| CheckVars
    ParseURL --> Init
    CheckVars -->|Yes| ParseVars
    CheckVars -->|No| NoRedis
    ParseVars --> Init
    Init --> Return
```

Sources: [orchestrator/core/redis/client.py:149-198](), [orchestrator/config.py:46-62]()

#### Cache Policies

Production Redis is configured with LRU eviction for memory management:

```bash
# Redis configuration (set via Railway or docker-compose)
maxmemory 256mb
maxmemory-policy allkeys-lru
```

**Cache TTL Values:**

| Cache Type | TTL (seconds) | Config Variable |
|------------|---------------|-----------------|
| Plugin content | 3600 (1 hour) | `PLUGIN_CACHE_TTL_SECONDS` |
| Composio apps | 86400 (24 hours) | `ROUTING_CACHE_TTL_HOURS` |
| Session data | Varies | Configured in application |

Sources: [docker-compose.yml:48-63](), [orchestrator/core/services/plugin_cache.py:38-47]()

#### Pub/Sub Channels

Redis pub/sub is used for real-time workflow execution updates:

**Channel Naming Convention:**
```
workflow:{workflow_id}:execution:{execution_id}
```

**Message Format:**
```json
{
  "type": "execution_started | subtask_execution_update | execution_completed",
  "data": {
    "execution_id": 123,
    "workflow_id": 456,
    ...
  }
}
```

The `RedisClient.publish_workflow_event()` method handles message formatting and publishing.

Sources: [orchestrator/core/redis/client.py:91-119]()

---

## Secrets Management

### Credential Encryption

Production credentials are encrypted using AES-256-GCM with workspace-scoped encryption keys.

#### Encryption Architecture

```mermaid
graph TB
    subgraph "Credential Storage"
        UI["UI: Create Credential"]
        API["POST /api/credentials"]
        Store["CredentialStore.create_credential()"]
        Encrypt["encryption_service.encrypt_dict()"]
        DB["Database: credentials table<br/>encrypted_data column"]
    end
    
    subgraph "Credential Retrieval"
        Service["Service Request"]
        Resolve["get_decrypted_credential()"]
        Decrypt["encryption_service.decrypt_dict()"]
        Audit["Audit Log Entry"]
        Return["Return plaintext"]
    end
    
    UI --> API
    API --> Store
    Store --> Encrypt
    Encrypt --> DB
    Service --> Resolve
    Resolve --> Decrypt
    Decrypt --> DB
    Decrypt --> Audit
    Audit --> Return
```

**Encryption Key Setup:**
```bash
# Generate encryption key (run once)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set as environment variable (Railway secret)
ENCRYPTION_KEY=your_generated_key_here
```

**Security Features:**
- AES-256-GCM encryption
- Workspace-scoped credential isolation
- Audit logging for all access
- Automatic key rotation support
- Expiration date enforcement

Sources: [orchestrator/core/credentials/service.py:42-182](), [orchestrator/core/models/credentials.py:60-103]()

### API Key Rotation

Production API keys should be rotated periodically:

```bash
# Generate new API key
openssl rand -hex 32

# Update Railway environment variable
railway variables set API_KEY=new_key_here

# Restart backend service (Railway auto-restarts on env change)
```

The `get_request_context_hybrid` authentication middleware supports API key validation:

```python
# Headers for API key authentication
X-API-Key: your_api_key_here
```

Sources: [orchestrator/config.py:66-68]()

### CORS Configuration

Production CORS must explicitly allow frontend domains:

```bash
# Single domain
CORS_ALLOW_ORIGINS=https://app.yourdomain.com

# Multiple domains (comma-separated)
CORS_ALLOW_ORIGINS=https://app.yourdomain.com,https://staging.yourdomain.com

# Railway default
CORS_ALLOW_ORIGINS=https://automotas-ai-frontend-production.up.railway.app
```

The `Config` class automatically parses and validates CORS origins:

```python
# Parsed into list, stripping whitespace
_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "...")
CORS_ALLOW_ORIGINS: str = ",".join([origin.strip() for origin in _cors_origins.split(",") if origin.strip()])
```

Sources: [orchestrator/config.py:71-79]()

---

## Health Checks & Monitoring

### Health Endpoints

Both frontend and backend include health check endpoints for monitoring and auto-restart.

#### Backend Health Check

```bash
# Health endpoint
GET /health

# Response
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "database": "connected",
  "redis": "connected"
}
```

**Docker Health Check Configuration:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1
```

Sources: [orchestrator/Dockerfile:106-108]()

#### Frontend Health Check

```bash
# Root endpoint returns Next.js HTML
GET /

# Docker health check
curl -f http://localhost:3000 || exit 1
```

**Health Check Configuration:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1
```

Sources: [frontend/Dockerfile:112-114]()

### Railway Monitoring

Railway provides built-in monitoring:
- **Metrics**: CPU, memory, network usage
- **Logs**: Aggregated stdout/stderr from all services
- **Deployments**: Automatic rollback on health check failures
- **Alerts**: Configure alerts for downtime or errors

**Access Logs:**
```bash
# View live logs
railway logs

# View logs for specific service
railway logs --service backend
```

---

## Scaling Considerations

### Horizontal Scaling

#### Backend Worker Processes

Production backend runs multiple Uvicorn workers:

```dockerfile
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
```

**Worker Count Recommendations:**

| Deployment Size | Workers | Memory per Worker | Total Memory |
|----------------|---------|-------------------|--------------|
| Small (Railway Hobby) | 2 | 256 MB | 512 MB |
| Medium (Railway Pro) | 4 | 512 MB | 2 GB |
| Large (Railway Enterprise) | 8 | 1 GB | 8 GB |

Formula: `workers = (2 * num_cpu_cores) + 1`

Sources: [orchestrator/Dockerfile:115]()

#### Database Connection Pooling

Each worker maintains its own connection pool. Ensure PostgreSQL `max_connections` accommodates all workers:

```
max_connections = (num_workers * pool_size * num_backend_instances) + overhead
```

Example for 2 backend instances with 4 workers each:
```
max_connections = (4 * 20 * 2) + 20 = 180
```

Set `max_connections=200` in PostgreSQL configuration for safety margin.

Sources: [docker-compose.yml:29]()

#### Redis Connection Management

Redis uses connection pooling to handle concurrent requests:

```python
# Connection pool configuration
self.pool = redis.ConnectionPool(
    host=host,
    port=port,
    password=password,
    db=db,
    decode_responses=True,
    max_connections=50  # Adjust based on worker count
)
```

For multiple backend instances, increase `max_connections`:
```
max_connections = num_workers * num_instances * 10
```

Sources: [orchestrator/core/redis/client.py:22-29]()

### Vertical Scaling

Railway allows easy vertical scaling through the dashboard:

**Resource Limits:**

| Tier | CPU | Memory | Disk |
|------|-----|--------|------|
| Hobby | Shared | 512 MB - 8 GB | 100 GB |
| Pro | Shared | 512 MB - 32 GB | 100 GB |
| Enterprise | Dedicated | Custom | Custom |

**Scaling Guidelines:**

1. **Monitor resource usage** via Railway dashboard
2. **Scale memory first** if OOM errors occur
3. **Scale CPU** if high latency with low memory usage
4. **Increase workers** if CPU utilization < 70% but response time is slow

---

## Environment-Specific Configuration

### Production vs Development

The `Config` class provides environment detection:

```python
@property
def IS_PRODUCTION(self) -> bool:
    return self.ENVIRONMENT.lower() == "production"

@property
def IS_DEVELOPMENT(self) -> bool:
    return self.ENVIRONMENT.lower() == "development"
```

**Key Differences:**

| Setting | Development | Production |
|---------|-------------|------------|
| `ENVIRONMENT` | `development` | `production` |
| `LOG_LEVEL` | `DEBUG` | `INFO` or `WARNING` |
| `DEBUG` | `true` | `false` |
| Docker target | `development` | `production` |
| Hot reload | Enabled | Disabled |
| Source mounts | Yes | No |
| Optimization | None | Multi-stage, cleaned |

Sources: [orchestrator/config.py:114-123]()

### Feature Flags

Production deployments can enable/disable features via environment variables:

```bash
# Batch API (experimental)
ENABLE_BATCH_API=false

# S3 Vector Storage
S3_VECTORS_ENABLED=false

# Jira Bug Reports (Pilot feature)
JIRA_BUG_REPORTS_ENABLED=true
```

Feature flags are centralized in the `Config` class:

```python
ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
S3_VECTORS_ENABLED: bool = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"
```

Sources: [orchestrator/config.py:154-174]()

---

## Troubleshooting Production Issues

### Common Issues

#### 1. Backend Fails to Start

**Symptoms:**
- Health check failures
- Crash loop in Railway logs

**Diagnosis:**
```bash
# Check logs
railway logs --service backend

# Common errors:
# - "Database connection failed"
# - "Redis connection failed"
# - "Failed to decrypt credentials"
```

**Solutions:**

| Error | Solution |
|-------|----------|
| Database connection failed | Verify `DATABASE_URL` is set and PostgreSQL plugin is running |
| Redis connection failed | Redis is optional; set `REDIS_URL` or remove Redis dependencies |
| Failed to decrypt credentials | Set `ENCRYPTION_KEY` environment variable |
| Import errors | Ensure all dependencies in `requirements.txt` are installed |

#### 2. Frontend Cannot Connect to Backend

**Symptoms:**
- "Failed to fetch" errors in browser console
- CORS errors

**Solutions:**
1. Verify `NEXT_PUBLIC_API_URL` points to backend service URL
2. Check CORS configuration includes frontend domain
3. Ensure backend service is healthy (check `/health` endpoint)

#### 3. Slow Database Queries

**Diagnosis:**
```bash
# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Check connection pool exhaustion
SELECT count(*) FROM pg_stat_activity;
```

**Solutions:**
- Increase `max_connections` in PostgreSQL
- Increase connection pool size in application
- Add database indexes for slow queries
- Enable query logging to identify bottlenecks

#### 4. Redis Connection Timeouts

**Symptoms:**
- Workflow updates not streaming
- Plugin content cache misses

**Solutions:**
```bash
# Test Redis connection
redis-cli -u $REDIS_URL ping

# Check memory usage
redis-cli -u $REDIS_URL INFO memory

# Clear cache if memory full
redis-cli -u $REDIS_URL FLUSHDB
```

Sources: [orchestrator/core/redis/client.py:121-134](), [docs/LOCAL_SETUP_GUIDE.md:145-175]()

---

## Deployment Checklist

Before deploying to production:

### Pre-Deployment

- [ ] Generate and set `ENCRYPTION_KEY` for credential encryption
- [ ] Set all required environment variables (see [Environment Variables](#12.3))
- [ ] Configure CORS with production frontend domains
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `LOG_LEVEL=INFO` or `WARNING`
- [ ] Verify LLM API keys are valid and have sufficient credits
- [ ] Create S3 bucket for plugin marketplace
- [ ] Configure Clerk authentication with production domain
- [ ] Set strong `API_KEY` for backend API authentication

### Post-Deployment

- [ ] Verify health endpoints return 200 OK
- [ ] Test database connection and schema initialization
- [ ] Test Redis connection and pub/sub functionality
- [ ] Verify Clerk authentication flow works end-to-end
- [ ] Test credential encryption/decryption
- [ ] Verify S3 plugin upload/download
- [ ] Load test with expected traffic volume
- [ ] Configure monitoring alerts in Railway
- [ ] Set up database backups (Railway auto-backups enabled)
- [ ] Document deployment configuration and runbook

### Monitoring

- [ ] Monitor CPU/memory usage in Railway dashboard
- [ ] Check application logs for errors
- [ ] Monitor database connection pool utilization
- [ ] Track API response times and error rates
- [ ] Set up alerts for service downtime
- [ ] Configure log retention and aggregation

Sources: [orchestrator/config.py:225-247](), [orchestrator/.env.example:1-64]()

---