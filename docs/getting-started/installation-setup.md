# Installation & Setup

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitignore](.gitignore)
- [README.md](README.md)
- [docker-compose.yml](docker-compose.yml)
- [docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md](docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md)
- [docs/README.md](docs/README.md)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/api/cloud_documents.py](orchestrator/api/cloud_documents.py)
- [orchestrator/core/credentials/service.py](orchestrator/core/credentials/service.py)
- [orchestrator/core/models/credentials.py](orchestrator/core/models/credentials.py)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/modules/tools/services/__init__.py](orchestrator/modules/tools/services/__init__.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)

</details>



This page guides you through installing and running Automatos AI using Docker Compose. It covers cloning the repository, configuring environment variables, starting services, and verifying the installation. For detailed configuration of individual services (LLM providers, Redis, PostgreSQL, S3, Mem0), see [Configuration Guide](2.2).

---

## Prerequisites

Before installing Automatos AI, ensure your system has:

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Git** for repository cloning
- **Minimum 8GB RAM** (16GB recommended for production)
- **10GB disk space** for Docker images and volumes
- **Port availability**: 3000 (frontend), 8000 (backend), 5432 (PostgreSQL), 6379 (Redis)

**Sources:** [README.md:98-103](), [README.md:114-121]()

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
```

### 2. Configure Environment Variables

Copy the example environment file and configure required secrets:

```bash
cp .env.example .env
```

Edit `.env` and set the following **required** variables:

```bash
# Required - Database credentials
POSTGRES_PASSWORD=your_secure_postgres_password

# Required - Redis credentials  
REDIS_PASSWORD=your_secure_redis_password

# Required - API authentication
API_KEY=your_secure_api_key

# Optional - LLM provider keys (can be set via UI later)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Note:** The system will not start without `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, and `API_KEY`. LLM provider credentials can be configured later through the Settings UI at [http://localhost:3000/settings](http://localhost:3000/settings).

**Sources:** [docker-compose.yml:5-11](), [docker-compose.yml:29](), [docker-compose.yml:56](), [docker-compose.yml:116]()

### 3. Start Services

```bash
# Start core services (postgres, redis, backend, frontend)
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d

# Include admin tools (Adminer, Gotenberg)
docker-compose --profile all up --build

# Include workspace worker for sandboxed execution
docker-compose --profile workers up --build
```

The `--build` flag ensures images are rebuilt with the latest code changes. Omit it for faster startup after the initial build.

**Sources:** [docker-compose.yml:14-16](), [docker-compose.yml:184-185]()

### 4. Access the Platform

Once all services are healthy (typically 30-60 seconds):

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main user interface |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation (Swagger) |
| **Health Check** | http://localhost:8000/health | Backend health status |
| **Adminer** (if `--profile all`) | http://localhost:8080 | PostgreSQL database browser |

**Sources:** [README.md:105-106](), [docker-compose.yml:162](), [docker-compose.yml:123](), [orchestrator/Dockerfile:78-79]()

---

## Docker Compose Service Architecture

The following diagram shows the services defined in `docker-compose.yml` and their dependencies:

### System Component Map

```mermaid
graph TB
    subgraph "Core_Services_(default_profile)"
        postgres["postgres<br/>(pgvector/pgvector:pg16)"]
        redis["redis<br/>(redis:7-alpine)"]
        backend["backend<br/>(automatos_backend)"]
        frontend["frontend<br/>(automatos_frontend)"]
    end
    
    subgraph "Worker_Services_(--profile_workers)"
        workspace_worker["workspace-worker<br/>(automatos_workspace_worker)"]
    end
    
    subgraph "Admin_Tools_(--profile_all)"
        adminer["adminer<br/>(adminer:latest)"]
        gotenberg["gotenberg<br/>(gotenberg/gotenberg:8)"]
    end
    
    subgraph "Persistent_Volumes"
        postgres_data["postgres_data"]
        redis_data["redis_data"]
        backend_logs["backend_logs"]
        workspace_data["workspace_data"]
    end
    
    frontend -->|"depends_on: service_healthy"| backend
    backend -->|"depends_on: service_healthy"| postgres
    backend -->|"depends_on: service_healthy"| redis
    workspace_worker -->|"depends_on: service_healthy"| postgres
    workspace_worker -->|"depends_on: service_healthy"| redis
    adminer -->|"depends_on"| postgres
    
    postgres -.->|"mounts"| postgres_data
    redis -.->|"mounts"| redis_data
    backend -.->|"mounts"| backend_logs
    backend -.->|"mounts (ro)"| workspace_data
    workspace_worker -.->|"mounts (rw)"| workspace_data
```

**Service Details:**

| Service | Container Name | Image | Ports | Health Check |
|---------|---------------|-------|-------|--------------|
| **postgres** | `automatos_postgres` | `pgvector/pgvector:pg16` | 5432 | `pg_isready -U postgres` |
| **redis** | `automatos_redis` | `redis:7-alpine` | 6379 | `redis-cli ping` |
| **backend** | `automatos_backend` | Built from `orchestrator/Dockerfile` | 8000 | `curl -f http://localhost:8000/health` |
| **frontend** | `automatos_frontend` | Built from `frontend/Dockerfile` | 3000 | `wget http://localhost:3000` |
| **workspace-worker** | `automatos_workspace_worker` | Built from `services/workspace-worker/Dockerfile` | 8081 | `curl -f http://localhost:8081/health` |

**Sources:** [docker-compose.yml:18-170](), [docker-compose.yml:178-193](), [orchestrator/Dockerfile:88-130](), [frontend/Dockerfile:85-114]()

---

## Environment Variables Reference

The Docker Compose stack requires specific environment variables. Below are the most critical ones; for comprehensive configuration, see [Configuration Guide](2.2).

### Required Variables

| Variable | Description | Example | Used By |
|----------|-------------|---------|---------|
| `POSTGRES_PASSWORD` | PostgreSQL root password | `SecurePass123!` | postgres, backend, workspace-worker |
| `REDIS_PASSWORD` | Redis authentication password | `RedisSecure456!` | redis, backend, workspace-worker |
| `API_KEY` | Backend API authentication key | `automatos_api_key_xyz` | backend |

**Sources:** [docker-compose.yml:29](), [docker-compose.yml:56](), [docker-compose.yml:116]()

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `orchestrator_db` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `DATABASE_URL` | Auto-generated | Full connection string (postgresql://user:pass@host:port/db) |

**Sources:** [docker-compose.yml:27-28](), [docker-compose.yml:92-97]()

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis hostname (docker-compose service name) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | **Required** | Redis auth password |

Redis is configured with security hardening — dangerous commands like `FLUSHDB`, `FLUSHALL`, and `DEBUG` are disabled via `--rename-command` in [docker-compose.yml:59-61]().

**Sources:** [docker-compose.yml:54-61](), [docker-compose.yml:100-102]()

### Authentication (Clerk)

| Variable | Description | Optional |
|----------|-------------|----------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk frontend key | Yes (for local dev without auth) |
| `CLERK_SECRET_KEY` | Clerk backend secret | Yes |
| `CLERK_JWKS_URL` | Clerk JWKS endpoint | Yes |

**Sources:** [docker-compose.yml:119-121](), [docker-compose.yml:159]()

---

## Service Initialization Sequence

The following diagram shows the startup and health check flow for core services:

### Startup Flow & Code Entities

```mermaid
sequenceDiagram
    participant DC as "docker-compose up"
    participant PG as "postgres<br/>(pgvector/pgvector:pg16)"
    participant RD as "redis<br/>(redis:7-alpine)"
    participant BE as "backend<br/>(automatos_backend)"
    participant FE as "frontend<br/>(automatos_frontend)"
    
    DC->>PG: "Start container"
    activate PG
    Note over PG: "Run init SQL:<br/>01-schema.sql"
    PG->>PG: "Health check:<br/>pg_isready -U postgres"
    PG-->>DC: "healthy"
    deactivate PG
    
    DC->>RD: "Start container"
    activate RD
    Note over RD: "Apply config:<br/>--requirepass $REDIS_PASSWORD"
    RD->>RD: "Health check:<br/>redis-cli ping"
    RD-->>DC: "healthy"
    deactivate RD
    
    DC->>BE: "Start container<br/>(depends_on: postgres, redis healthy)"
    activate BE
    Note over BE: "Run entrypoint:<br/>docker-entrypoint.sh"
    BE->>PG: "Connect via DATABASE_URL"
    BE->>BE: "init_redis_client()<br/>(core/redis/client.py)"
    BE->>BE: "Health check:<br/>/health"
    BE-->>DC: "healthy"
    deactivate BE
    
    DC->>FE: "Start container<br/>(depends_on: backend healthy)"
    activate FE
    Note over FE: "Next.js dev server:<br/>npm run dev"
    FE->>BE: "Proxy API requests to<br/>http://backend:8000"
    FE->>FE: "Health check:<br/>wget http://localhost:3000"
    FE-->>DC: "healthy"
    deactivate FE
```

**Key Initialization Steps:**

1. **PostgreSQL** ([docker-compose.yml:22-43]()):
   - Runs schema initialization from `orchestrator/database/init_complete_schema.sql` via mount [docker-compose.yml:35]().
   - Health check validates via `pg_isready` [docker-compose.yml:37]().

2. **Redis** ([docker-compose.yml:48-73]()):
   - Applies security config (disabled commands, password auth) [docker-compose.yml:54-61]().
   - Health check validates via `redis-cli ping` [docker-compose.yml:67]().

3. **Backend** ([docker-compose.yml:78-138]()):
   - Executes `docker-entrypoint.sh` [orchestrator/Dockerfile:82]().
   - Initializes `RedisClient` via `init_redis_client` [orchestrator/core/redis/client.py:141-146]().
   - Exposes health endpoint at `/health` [orchestrator/Dockerfile:78-79]().

4. **Frontend** ([docker-compose.yml:146-170]()):
   - Starts Next.js dev server [frontend/Dockerfile:48]().
   - Proxies API calls to `backend` service [docker-compose.yml:157]().
   - Health check validates page load [frontend/Dockerfile:44-45]().

**Sources:** [docker-compose.yml:35-41](), [docker-compose.yml:66-71](), [orchestrator/core/redis/client.py:141-146](), [orchestrator/Dockerfile:78-82](), [frontend/Dockerfile:44-48]()

---

## Database Initialization

PostgreSQL is initialized with the complete schema on first startup. The schema includes:

### Schema Initialization Process

```mermaid
graph LR
    A["docker-compose up<br/>(postgres service)"] --> B["Mount SQL file:<br/>01-schema.sql"]
    B --> C["PostgreSQL Entrypoint"]
    C --> D{"Schema exists?"}
    D -->|"No"| E["Execute init_complete_schema.sql"]
    D -->|"Yes"| F["Skip (idempotent)"]
    E --> G["Create tables:<br/>workspaces, agents,<br/>cloud_documents, etc."]
    G --> H["Create indexes"]
    H --> I["Enable pgvector extension"]
    I --> J["Database ready"]
    F --> J
```

**Key Database Objects Created:**

| Object Type | Examples | Purpose |
|-------------|----------|---------|
| **Tables** | `workspaces`, `agents`, `cloud_documents` | Core data models [orchestrator/api/cloud_documents.py:20-21]() |
| **Extensions** | `pgvector` | Vector similarity search for embeddings [docker-compose.yml:23]() |

**Schema Migration Notes:**
- Initial schema is applied automatically on first run via [docker-compose.yml:35]().
- For schema updates, the backend uses `alembic` [orchestrator/requirements.txt:8]().

**Sources:** [docker-compose.yml:23-35](), [orchestrator/requirements.txt:8]()

---

## Volume Mounts and Data Persistence

Docker volumes ensure data persists across container restarts:

| Volume Name | Mount Point | Service(s) | Purpose |
|-------------|-------------|------------|---------|
| `postgres_data` | `/var/lib/postgresql/data` | postgres | Database files [docker-compose.yml:34]() |
| `redis_data` | `/data` | redis | Redis persistence [docker-compose.yml:65]() |
| `backend_logs` | `/app/logs` | backend | Application logs [docker-compose.yml:128]() |
| `workspace_data` | `/workspaces` | backend (ro) | Agent workspace directories [docker-compose.yml:130]() |

**Sources:** [docker-compose.yml:33-35](), [docker-compose.yml:64-65](), [docker-compose.yml:124-130]()

---

## Verification Steps

After starting services, verify the installation:

### 1. Check Service Health

```bash
docker-compose ps
```

### 2. Test Backend Health Endpoint

```bash
curl http://localhost:8000/health
```

### 3. Test Redis Connection

```bash
docker exec automatos_backend python -c "
from core.redis.client import get_redis_client
client = get_redis_client()
print('Redis connected:', client.test_connection())
"
```

**Sources:** [orchestrator/core/redis/client.py:121-134](), [orchestrator/Dockerfile:78-79]()

---

## Python Dependencies (Backend)

The backend service installs Python packages from `orchestrator/requirements.txt`. Key dependency categories:

### Core Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >=0.115.0 | Web framework [orchestrator/requirements.txt:2]() |
| `sqlalchemy` | ==2.0.23 | ORM for database access [orchestrator/requirements.txt:7]() |
| `pydantic` | >=2.7.4 | Data validation [orchestrator/requirements.txt:16]() |

### AI & LLM Providers

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >=1.10.0 | OpenAI API client [orchestrator/requirements.txt:73]() |
| `anthropic` | >=0.40.0 | Anthropic API client [orchestrator/requirements.txt:74]() |
| `tiktoken` | >=0.5.0 | Token counting [orchestrator/requirements.txt:72]() |
| `composio-openai` | ==0.11.1 | Composio tool integration [orchestrator/requirements.txt:103]() |

### Special Installation: FutureAGI

The `futureagi` package is installed with `--no-deps` to avoid version conflicts with the core stack. This is necessary because it pins exact versions of common packages like `requests` and `pandas` that conflict with the platform's newer requirements.

**Sources:** [orchestrator/requirements.txt:1-113](), [orchestrator/Dockerfile:39-43]()

---

## Credential Encryption & Store

Automatos AI uses symmetric encryption for sensitive credentials managed via the `CredentialStore`. On first run, it attempts to load a key from `CREDENTIAL_ENCRYPTION_KEY` or generates a new one in `.credential_key`.

| Feature | Implementation |
|---------|----------------|
| **Encryption Service** | `get_encryption_service()` [orchestrator/core/credentials/encryption.py:145]() |
| **Algorithm** | Fernet (AES-128-CBC + HMAC-SHA256) |
| **Store Logic** | `create_credential()` encrypts data before SQL insertion [orchestrator/core/credentials/service.py:147]() |
| **Tenant Isolation** | `workspace_id` enforced in all CRUD operations [orchestrator/core/credentials/service.py:153]() |

**Sources:** [orchestrator/core/credentials/service.py:42-185](), [orchestrator/core/models/credentials.py:60-104]()

---

## Next Steps

1. **Configure LLM Providers** - Set up API keys via the Settings UI.
2. **Create Your First Agent** - Follow [Quick Start Tutorial](2.3).
3. **Enable Workspace Execution** - Start the `workers` profile for sandboxed code execution [docker-compose.yml:184]().

**Sources:** [docker-compose.yml:184-185]()

---