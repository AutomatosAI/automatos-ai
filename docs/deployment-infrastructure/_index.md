# Deployment & Infrastructure

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



## Purpose and Scope

This document covers the containerization, orchestration, and deployment infrastructure for Automatos AI. It explains the Docker multi-stage build process, Docker Compose service orchestration, environment variable configuration, database and cache setup, and production deployment strategies.

**Related Pages:**
- For initial setup instructions, see [Installation & Setup](#2.1)
- For configuration options, see [Configuration Guide](#2.2)
- For authentication configuration, see [Authentication Flow](#9.1)
- For credentials management, see [Credentials Management](#9.4)

---

## System Overview

Automatos AI uses a containerized architecture with six services orchestrated by Docker Compose. The system supports both development (hot-reload) and production (optimized) deployment targets through multi-stage Dockerfiles and profile-based service activation.

```mermaid
graph TB
    subgraph "Core Services (Default Profile)"
        Frontend["frontend<br/>container: automatos_frontend<br/>Port 3000"]
        Backend["backend<br/>container: automatos_backend<br/>Port 8000"]
        Postgres["postgres<br/>container: automatos_postgres<br/>image: pgvector/pgvector:pg16<br/>Port 5432"]
        Redis["redis<br/>container: automatos_redis<br/>image: redis:7-alpine<br/>Port 6379"]
    end
    
    subgraph "Worker Services (--profile workers)"
        WorkspaceWorker["workspace-worker<br/>container: automatos_workspace_worker<br/>Port 8081"]
    end
    
    subgraph "Admin Tools (--profile all)"
        Adminer["adminer<br/>container: automatos_adminer<br/>Port 8080"]
        Gotenberg["gotenberg<br/>container: automatos_gotenberg<br/>Port 3001"]
    end
    
    subgraph "Persistent Storage"
        PGData["postgres_data<br/>Named Volume"]
        RedisData["redis_data<br/>Named Volume"]
        Logs["backend_logs<br/>Named Volume"]
        WorkspaceData["workspace_data<br/>Named Volume"]
    end
    
    Frontend -->|"HTTP/SSE"| Backend
    Backend -->|"SQLAlchemy"| Postgres
    Backend -->|"redis.ConnectionPool"| Redis
    WorkspaceWorker -->|"SQL"| Postgres
    WorkspaceWorker -->|"Task Queue"| Redis
    Backend -->|"HTTP (PRD-63)"| Gotenberg
    Backend -->|"Read-Only (PRD-66)"| WorkspaceData
    WorkspaceWorker -->|"Read-Write"| WorkspaceData
    Backend --> Logs
    Postgres --> PGData
    Redis --> RedisData
    Adminer -.->|"Admin UI"| Postgres
```

**Sources:** [docker-compose.yml:1-280]()

---

## Backend Containerization

### Multi-Stage Dockerfile Architecture

The backend uses a three-stage build process to optimize for different deployment scenarios while minimizing image size.

```mermaid
graph LR
    subgraph "Stage 1: base"
        BaseImg["FROM python:3.11-slim<br/>Line 13"]
        SysDeps["apt-get install:<br/>gcc, g++, curl, git<br/>postgresql-client<br/>libmagic1, tesseract-ocr<br/>ghostscript<br/>WeasyPrint deps"]
        Requirements["COPY requirements.txt<br/>Line 36"]
        PyDeps["pip install -r requirements.txt<br/>pip install --no-deps futureagi==0.6.0"]
        NLTK["python -c 'import nltk;<br/>nltk.download(punkt, stopwords)'<br/>NLTK_DATA=/usr/local/nltk_data"]
    end
    
    subgraph "Stage 2: development"
        DevBase["FROM base as development<br/>Line 57"]
        DevEntry["RUN echo docker-entrypoint.sh<br/>Line 61"]
        DevCode["COPY . .<br/>Line 65"]
        DevDirs["mkdir logs, vector_stores,<br/>projects, exports"]
        DevUser["USER automatos<br/>Line 72"]
        DevCmd["CMD uvicorn main:app --reload<br/>Line 85"]
    end
    
    subgraph "Stage 3: production"
        ProdBase["FROM base as production<br/>Line 90"]
        ProdCode["COPY . .<br/>Line 94"]
        ProdClean["pip uninstall pytest black isort<br/>rm __pycache__, *.pyc"]
        ProdUser["USER automatos<br/>Line 115"]
        ProdCmd["CMD uvicorn --port $PORT --workers 4<br/>Line 129"]
    end
    
    BaseImg --> SysDeps --> Requirements --> PyDeps --> NLTK
    NLTK --> DevBase
    NLTK --> ProdBase
    DevBase --> DevEntry --> DevCode --> DevDirs --> DevUser --> DevCmd
    ProdBase --> ProdCode --> ProdClean --> ProdUser --> ProdCmd
```

**Sources:** [orchestrator/Dockerfile:1-130](), [orchestrator/requirements.txt:1-108]()

### Base Stage Dependencies

The base stage installs system-level dependencies required for Python packages and AI operations:

| Dependency | Purpose | Configuration |
|------------|---------|---------------|
| `gcc`, `g++` | Compile native Python extensions (numpy, pandas, scikit-learn) | [orchestrator/Dockerfile:18-19]() |
| `curl`, `git` | Health checks, repository cloning (PRD-11) | [orchestrator/Dockerfile:21-22]() |
| `postgresql-client` | Database connectivity validation | [orchestrator/Dockerfile:23]() |
| `libmagic1` | File type detection (`python-magic`) | [orchestrator/Dockerfile:24]() |
| `tesseract-ocr` | OCR for document processing (`pytesseract`) | [orchestrator/Dockerfile:25]() |
| `ghostscript` | PDF rendering support | [orchestrator/Dockerfile:26]() |
| `libpango-1.0-0`, `libcairo2`, `libgdk-pixbuf-2.0-0`, `libffi-dev` | WeasyPrint HTML-to-PDF rendering (PRD-63) | [orchestrator/Dockerfile:28-31]() |

#### Python Dependencies Installation

Python packages are installed with special handling for `futureagi` ([orchestrator/Dockerfile:42-45]()):

```bash
# Install from requirements.txt (150+ packages)
pip install --no-cache-dir -r requirements.txt

# Install futureagi without dependencies to avoid version conflicts
pip install --no-cache-dir --no-deps futureagi==0.6.0
```

The `--no-deps` flag prevents `futureagi` from overwriting already-installed package versions (`requests==2.32.3`, `pandas==2.2.2`, etc.) with its pinned dependencies.

#### NLTK Data Pre-download

NLTK tokenizers and stopwords are downloaded at build time to avoid runtime downloads ([orchestrator/Dockerfile:48-52]()):

```bash
ENV NLTK_DATA=/usr/local/nltk_data
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True, download_dir='/usr/local/nltk_data'); \
    nltk.download('stopwords', quiet=True, download_dir='/usr/local/nltk_data')" && \
    chmod -R 755 /usr/local/nltk_data
```

**Sources:** [orchestrator/Dockerfile:13-53](), [orchestrator/requirements.txt:1-108]()

### Development Stage

The development stage enables hot-reload for rapid iteration:

- **Entrypoint:** Minimal script created inline that delegates to CMD ([orchestrator/Dockerfile:61-62]())
  ```bash
  RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /usr/local/bin/docker-entrypoint.sh && \
      chmod +x /usr/local/bin/docker-entrypoint.sh
  ```
- **Code Mount:** Source code mounted from host via volume for hot-reload ([docker-compose.yml:125-127]())
- **Reload Mode:** `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` ([orchestrator/Dockerfile:85]())
- **Health Check:** `curl -f http://localhost:8000/health` every 30s, 40s start period ([orchestrator/Dockerfile:78-79]())
- **User:** Runs as `automatos` user (UID 1000) ([orchestrator/Dockerfile:72]())

**Sources:** [orchestrator/Dockerfile:57-86](), [docker-compose.yml:78-138]()

### Production Stage

The production stage creates a secure, minimal image:

1. **Cleanup:** Removes dev dependencies and caches ([orchestrator/Dockerfile:105-109]()):
   ```bash
   RUN pip uninstall -y pytest pytest-asyncio black isort 2>/dev/null || true && \
       pip cache purge && \
       rm -rf /root/.cache/pip /root/.cache/pytest /tmp/* && \
       find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
       find /app -type f -name "*.pyc" -delete 2>/dev/null || true
   ```
2. **Non-root User:** Creates `automatos` user (UID 1000) ([orchestrator/Dockerfile:112-113]())
3. **Multi-worker:** Uses 4 uvicorn workers for concurrency ([orchestrator/Dockerfile:129]())
4. **Dynamic Port:** Supports Railway's `PORT` environment variable with fallback ([orchestrator/Dockerfile:129]()):
   ```bash
   CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
   ```
5. **Health Check:** Uses `${PORT:-8000}` variable in health check command ([orchestrator/Dockerfile:122]())

**Sources:** [orchestrator/Dockerfile:90-130]()

---

## Frontend Containerization

### Multi-Stage Dockerfile Architecture

The frontend uses a four-stage build to separate dependencies, development, build, and production runtime.

```mermaid
graph LR
    subgraph "Stage 1: base"
        NodeImg["FROM node:20-alpine<br/>Line 14"]
        AlpineDeps["apk add python3,<br/>make, g++, curl"]
        PkgCopy["COPY package*.json<br/>Line 26"]
    end
    
    subgraph "Stage 2: development"
        DevBase["FROM base as development<br/>Line 31"]
        DevInstall["npm install<br/>--legacy-peer-deps<br/>Line 35"]
        DevCopy["COPY . .<br/>Line 38"]
        DevCmd["CMD npm run dev<br/>Line 48"]
    end
    
    subgraph "Stage 3: builder"
        BuildBase["FROM base as builder<br/>Line 53"]
        BuildArgs["ARG NEXT_PUBLIC_API_URL<br/>ARG NEXT_PUBLIC_CLERK_*<br/>Lines 58-63"]
        BuildEnv["ENV NEXT_PUBLIC_*<br/>Lines 66-71"]
        BuildInstall["npm install<br/>--legacy-peer-deps<br/>Line 74"]
        BuildCopy["COPY . .<br/>Line 77"]
        BuildRun["npm run build<br/>(standalone output)<br/>Line 80"]
    end
    
    subgraph "Stage 4: production"
        ProdBase["FROM node:20-alpine<br/>Line 85"]
        ProdUser["adduser nextjs<br/>UID 1001<br/>Lines 93-94"]
        ProdCopy["COPY --from=builder<br/>.next/standalone<br/>.next/static<br/>public"]
        ProdCmd["CMD node server.js<br/>Line 114"]
    end
    
    NodeImg --> AlpineDeps --> PkgCopy
    PkgCopy --> DevBase
    PkgCopy --> BuildBase
    DevBase --> DevInstall --> DevCopy --> DevCmd
    BuildBase --> BuildArgs --> BuildEnv --> BuildInstall --> BuildCopy --> BuildRun
    ProdBase --> ProdUser --> ProdCopy --> ProdCmd
```

**Sources:** [frontend/Dockerfile:1-115]()

### Build Arguments vs Runtime Variables

The frontend distinguishes between build-time and runtime environment variables:

#### Build-Time Variables (NEXT_PUBLIC_*)

These are embedded into the client JavaScript bundle during `npm run build` ([frontend/Dockerfile:58-71]()):

| Variable | Purpose | Security | Default |
|----------|---------|----------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API endpoint | Public (embedded in JS) | None (required) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk authentication | Public (publishable key) | None (required) |
| `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | Sign-in route | Public | `/sign-in` |
| `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | Sign-up route | Public | `/sign-up` |
| `NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL` | Post-login redirect | Public | `/dashboard` |
| `NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL` | Post-signup redirect | Public | `/dashboard` |

**⚠️ Security Note:** `NEXT_PUBLIC_*` variables are **embedded in client-side JavaScript** and must not contain secrets. These values are baked into the bundle at build time ([frontend/Dockerfile:66-71]()) and cannot be changed without rebuilding.

#### Runtime Variables (Server-Side Only)

Secret variables are only available in server-side contexts and are never exposed to the client:

| Variable | Purpose | Access |
|----------|---------|--------|
| `CLERK_SECRET_KEY` | Server-side Clerk authentication | Next.js API routes only |
| `NODE_ENV` | Environment mode | Server runtime |
| `HOSTNAME`, `PORT` | Server binding | Production container |

**Sources:** [frontend/Dockerfile:58-71](), [frontend/Dockerfile:111-113](), [docker-compose.yml:156-160]()

### Production Stage Optimization

The production image uses Next.js standalone output mode for minimal footprint:

1. **Standalone Build:** Next.js traces dependencies and outputs only required files ([frontend/Dockerfile:80]())
2. **Selective Copy:** Copies traced dependencies and static assets from builder stage ([frontend/Dockerfile:97-99]()):
   ```dockerfile
   COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
   COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
   COPY --from=builder --chown=nextjs:nodejs /app/public ./public
   ```
3. **Non-root User:** Runs as `nextjs` user (UID 1001, group `nodejs` GID 1001) ([frontend/Dockerfile:93-94]())
4. **Direct Execution:** Uses `node server.js` instead of npm for faster startup ([frontend/Dockerfile:114]())
5. **Health Check:** `curl -f http://localhost:3000` with 60s start period ([frontend/Dockerfile:107-108]())

The standalone output eliminates the full `node_modules` directory, reducing image size by 60-70%.

**Sources:** [frontend/Dockerfile:85-114](), [frontend/.dockerignore:1-9]()

---

## Docker Compose Orchestration

### Service Configuration

The `docker-compose.yml` defines six services with health checks and dependency ordering, organized into three profiles:

```mermaid
graph TB
    subgraph "Profile: default (Core Services)"
        PG["postgres<br/>service: postgres<br/>container: automatos_postgres<br/>Port 5432"]
        RD["redis<br/>service: redis<br/>container: automatos_redis<br/>Port 6379"]
        BE["backend<br/>service: backend<br/>container: automatos_backend<br/>Port 8000<br/>target: development"]
        FE["frontend<br/>service: frontend<br/>container: automatos_frontend<br/>Port 3000<br/>target: development"]
    end
    
    subgraph "Profile: workers"
        WW["workspace-worker<br/>service: workspace-worker<br/>container: automatos_workspace_worker<br/>Port 8081"]
    end
    
    subgraph "Profile: all (Admin Tools)"
        AD["adminer<br/>service: adminer<br/>Port 8080"]
        GT["gotenberg<br/>service: gotenberg<br/>Port 3001"]
    end
    
    PG -->|"healthcheck:<br/>pg_isready -U postgres<br/>interval: 10s"| PG
    RD -->|"healthcheck:<br/>redis-cli ping<br/>interval: 10s"| RD
    
    BE -->|"depends_on:<br/>service_healthy"| PG
    BE -->|"depends_on:<br/>service_healthy"| RD
    BE -->|"healthcheck:<br/>curl /health<br/>interval: 30s"| BE
    
    FE -->|"depends_on:<br/>service_healthy"| BE
    FE -->|"healthcheck:<br/>wget /<br/>interval: 30s"| FE
    
    WW -->|"depends_on:<br/>service_healthy"| PG
    WW -->|"depends_on:<br/>service_healthy"| RD
    WW -->|"healthcheck:<br/>curl /health<br/>interval: 30s"| WW
    
    AD -.->|"optional"| PG
    BE -.->|"HTTP POST<br/>(PRD-63)"| GT
```

**Sources:** [docker-compose.yml:1-280]()

### PostgreSQL Service

The database service uses the official `pgvector/pgvector:pg16` image with optimized settings:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Image** | `pgvector/pgvector:pg16` | PostgreSQL 16 with vector extension |
| **Container Name** | `automatos_postgres` | DNS hostname within `automatos` network |
| **Environment** | `POSTGRES_DB=${POSTGRES_DB:-orchestrator_db}` | Database name with fallback |
| | `POSTGRES_USER=${POSTGRES_USER:-postgres}` | Database user with fallback |
| | `POSTGRES_PASSWORD=${POSTGRES_PASSWORD:?required}` | Required in `.env` file |
| | `POSTGRES_INITDB_ARGS` | `-c max_connections=200 -c shared_buffers=256MB` |
| **Volumes** | `postgres_data:/var/lib/postgresql/data` | Persistent data storage |
| | `init_complete_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro` | Schema initialization (first start only) |
| **Health Check** | `pg_isready -U postgres` | Runs every 10s, 5 retries, 10s start period |
| **Port** | `${POSTGRES_PORT:-5432}:5432` | Configurable host port |

#### Connection Pooling Configuration

The `max_connections=200` setting supports concurrent workflows and API requests. SQLAlchemy connection pools in the backend are sized accordingly:

```python
# Default SQLAlchemy pool: pool_size=5, max_overflow=10 per process
# With 4 uvicorn workers: 4 * (5 + 10) = 60 connections
```

This leaves 140 connections for workspace-worker and admin tools.

#### Schema Initialization

The `init_complete_schema.sql` script runs automatically on first database creation via Docker's `entrypoint-initdb.d` mechanism ([docker-compose.yml:35]()):
- Executes only when data directory is empty (first start)
- Creates all tables, indexes, and extensions
- Loads seed data (personas, categories, system settings)

**Sources:** [docker-compose.yml:22-43]()

### Redis Service

The cache service uses Redis 7 with LRU eviction policy and security hardening:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Image** | `redis:7-alpine` | Minimal Redis 7 image |
| **Container Name** | `automatos_redis` | DNS hostname within `automatos` network |
| **Command** | `redis-server` with flags | Custom configuration via CLI arguments |
| **Security (PRD-70)** | `--requirepass ${REDIS_PASSWORD:?required}` | Password authentication required |
| | `--rename-command FLUSHDB ""` | Disable FLUSHDB (prevents data wipe) |
| | `--rename-command FLUSHALL ""` | Disable FLUSHALL (prevents data wipe) |
| | `--rename-command DEBUG ""` | Disable DEBUG command |
| **Memory** | `--maxmemory 256mb` | Memory limit for cache |
| | `--maxmemory-policy allkeys-lru` | LRU eviction when full |
| **Volumes** | `redis_data:/data` | RDB snapshot persistence |
| **Health Check** | `redis-cli --no-auth-warning -a ${REDIS_PASSWORD} ping` | Runs every 10s, 5 retries |
| **Port** | `${REDIS_PORT:-6379}:6379` | Configurable host port |

#### Security Hardening (PRD-70 FIX-05)

Dangerous Redis commands are disabled by renaming them to empty strings ([docker-compose.yml:54-61]()):

```yaml
command: >
  redis-server
  --requirepass ${REDIS_PASSWORD:?REDIS_PASSWORD is required - set in .env file}
  --maxmemory 256mb
  --maxmemory-policy allkeys-lru
  --rename-command FLUSHDB ""
  --rename-command FLUSHALL ""
  --rename-command DEBUG ""
```

**Note:** Renaming `CONFIG` (commented in source) would break `redis-cli CONFIG` commands, which may be needed for debugging.

#### Redis Usage Patterns

Redis serves three purposes in the system:
1. **Caching:** Plugin content cache, Composio metadata cache ([orchestrator/core/services/plugin_cache.py]())
2. **Pub/Sub:** Real-time workflow execution events ([orchestrator/core/redis/client.py:91-119]())
3. **Task Queues:** Workspace worker task coordination ([services/workspace-worker/task_queue.py]())

**Sources:** [docker-compose.yml:48-73]()

### Backend Service

The FastAPI service builds from the `development` target for hot-reload:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Build** | `context: ./orchestrator` | Build from orchestrator directory |
| | `target: development` | Use development stage (hot-reload) |
| **Container Name** | `automatos_backend` | DNS hostname |
| **Dependencies** | `postgres: {condition: service_healthy}` | Wait for PostgreSQL health check |
| | `redis: {condition: service_healthy}` | Wait for Redis health check |
| **Volumes** | `./orchestrator:/app` | Mount source for hot-reload |
| | `./docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh:ro` | Custom entrypoint script |
| | `backend_logs:/app/logs` | Persistent application logs |
| | `workspace_data:/workspaces:ro` | Read-only workspace access (PRD-66) |
| **Environment** | `DATABASE_URL` | PostgreSQL connection string |
| | `REDIS_HOST=redis`, `REDIS_PORT=6379`, `REDIS_PASSWORD` | Redis connection |
| | `GOTENBERG_URL=http://gotenberg:3000` | Document generation service (PRD-63) |
| | `API_KEY`, `CLERK_SECRET_KEY`, `CLERK_JWKS_URL` | Authentication |
| | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | LLM providers (optional) |
| | `ENVIRONMENT=${ENVIRONMENT:-development}` | Runtime mode |
| **Health Check** | `curl -f http://localhost:8000/health` | 30s interval, 40s start period |
| **Port** | `${API_PORT:-8000}:8000` | Configurable host port |

#### Service Discovery

The backend uses Docker Compose service names as hostnames for inter-service communication:
- `postgres` resolves to PostgreSQL container IP
- `redis` resolves to Redis container IP
- `gotenberg` resolves to Gotenberg container IP (when running)

This eliminates the need for hardcoded IP addresses or environment-specific DNS configuration.

#### Volume Mounts

Three volume types are used:
1. **Bind Mount** (`./orchestrator:/app`): Source code for hot-reload in development
2. **Named Volume** (`backend_logs`): Persistent logs across container restarts
3. **Shared Named Volume** (`workspace_data:ro`): Read-only access to workspace files for code viewer widget (PRD-66)

**Sources:** [docker-compose.yml:78-138]()

### Frontend Service

The Next.js service builds from the `development` target:

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Build** | `context: ./frontend` | Build from frontend directory |
| | `target: development` | Use development stage (hot-reload) |
| **Container Name** | `automatos_frontend` | DNS hostname |
| **Dependencies** | `backend: {condition: service_healthy}` | Wait for backend health check |
| **Volumes** | `./frontend:/app` | Mount source for hot-reload |
| | `/app/node_modules` | Anonymous volume (prevent host overwrite) |
| | `/app/.next` | Anonymous volume (prevent host overwrite) |
| **Environment** | `NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-http://localhost:8000}` | Backend API endpoint |
| | `NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL:-ws://localhost:8000/ws}` | WebSocket endpoint |
| | `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk authentication |
| | `NODE_ENV=development` | Development mode |
| **Health Check** | `wget --no-verbose --tries=1 --spider http://localhost:3000` | 30s interval, 60s start period |
| **Port** | `${FRONTEND_PORT:-3000}:3000` | Configurable host port |

#### Anonymous Volume Pattern

The anonymous volumes (`/app/node_modules`, `/app/.next`) prevent the bind mount from overwriting container-generated files ([docker-compose.yml:167-168]()):

```yaml
volumes:
  - ./frontend:/app                # Bind mount (source code)
  - /app/node_modules              # Anonymous volume (higher priority)
  - /app/.next                     # Anonymous volume (higher priority)
```

This allows hot-reload while preserving dependencies installed during image build.

#### API URL Configuration

The `NEXT_PUBLIC_API_URL` uses `localhost:8000` instead of `backend:8000` because:
- The browser (client-side) resolves this URL, not the container
- In local development, both services are exposed on `localhost`
- In production (Railway), this would be set to the backend's public URL

**Sources:** [docker-compose.yml:146-170]()

### Workspace Worker Service

The workspace worker service executes agent tasks in isolated workspaces (PRD-56 Phase 2):

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Build** | `context: ./services/workspace-worker` | Separate worker service |
| **Container Name** | `automatos_workspace_worker` | DNS hostname |
| **Profile** | `workers` | Start with `--profile workers` |
| **Dependencies** | `postgres: {condition: service_healthy}` | Database access for task metadata |
| | `redis: {condition: service_healthy}` | Task queue coordination |
| **Environment** | `DATABASE_URL` | PostgreSQL connection |
| | `REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0` | Redis task queue |
| | `WORKSPACE_VOLUME_PATH=/workspaces` | Base path for workspace directories |
| | `WORKSPACE_DEFAULT_QUOTA_GB=${WORKSPACE_DEFAULT_QUOTA_GB:-5}` | Per-workspace storage limit |
| | `WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-3}` | Concurrent task limit |
| | `WORKER_HEALTH_PORT=8081` | Health check endpoint port |
| **Volumes** | `workspace_data:/workspaces` | Read-write workspace access |
| **Resource Limits** | `cpus: 2.0`, `memory: 2G` | Prevent resource exhaustion |
| **Health Check** | `curl -f http://localhost:8081/health` | 30s interval, 15s start period |

#### Task Execution Model

The workspace worker:
1. Polls Redis task queue for pending tasks ([services/workspace-worker/task_queue.py]())
2. Creates isolated workspace directories under `/workspaces/{workspace_id}/` ([services/workspace-worker/workspace_manager.py:115-145]())
3. Executes commands with sandboxing (path safety, command whitelist) ([services/workspace-worker/executor.py:36-470]())
4. Publishes progress events via Redis Pub/Sub ([services/workspace-worker/task_queue.py]())
5. Stores results in PostgreSQL and S3 ([services/workspace-worker/storage.py]())

**Sources:** [docker-compose.yml:178-217]()

### Admin Tools Profile

Optional services are available with `--profile all`:

#### Adminer (Database Admin UI)

| Configuration | Value |
|---------------|-------|
| **Image** | `adminer:latest` |
| **Container Name** | `automatos_adminer` |
| **Port** | `${ADMINER_PORT:-8080}:8080` |
| **Environment** | `ADMINER_DEFAULT_SERVER=postgres`, `ADMINER_DESIGN=nette` |
| **Usage** | Browse to `http://localhost:8080`, login with PostgreSQL credentials |

#### Gotenberg (Document Conversion)

| Configuration | Value | Purpose |
|---------------|-------|---------|
| **Image** | `gotenberg/gotenberg:8` | Chromium + LibreOffice for conversions |
| **Container Name** | `automatos_gotenberg` |
| **Port** | `${GOTENBERG_PORT:-3001}:3000` | HTTP API |
| **Environment** | `GOTENBERG_API_TIMEOUT=120s` | Timeout for large documents |
| | `GOTENBERG_LOG_LEVEL=info` | Logging verbosity |
| **Usage** | Backend sends POST requests for DOCX/XLSX → PDF conversion (PRD-63) |

#### Starting Admin Tools

```bash
# Start with all services
docker-compose --profile all up

# Or start specific profile
docker-compose --profile workers --profile all up
```

**Sources:** [docker-compose.yml:223-253]()

---

## Configuration Management

### Centralized Configuration Class

All environment variables are accessed through a single `Config` class to prevent scattered `os.getenv()` calls:

```mermaid
graph TB
    subgraph "Configuration Sources"
        EnvFile[".env File<br/>local development"]
        EnvVars["Environment Variables<br/>Railway, Docker"]
        Database["system_settings Table<br/>LLM provider config"]
        Defaults["Hardcoded Defaults<br/>non-sensitive only"]
    end
    
    subgraph "orchestrator/config.py"
        LoadDotenv["load_dotenv<br/>Line 16"]
        ConfigClass["class Config<br/>Line 18"]
        Properties["@property methods<br/>Lines 46-228"]
        Validation["validate method<br/>Lines 225-247"]
    end
    
    subgraph "Consumers"
        MainApp["main.py<br/>app.state.config"]
        DatabaseClient["database.py<br/>get_db"]
        RedisClient["core/redis/client.py<br/>get_redis_client"]
        LLMManager["core/llm/manager.py<br/>LLMManager.__init__"]
        S3Service["core/s3/client.py<br/>S3Client.__init__"]
        ToolRouter["modules/tools/tool_router.py"]
    end
    
    EnvFile --> LoadDotenv
    EnvVars --> LoadDotenv
    LoadDotenv --> ConfigClass
    ConfigClass --> Properties
    Database --> Properties
    Defaults --> Properties
    Properties --> Validation
    
    ConfigClass --> MainApp
    ConfigClass --> DatabaseClient
    ConfigClass --> RedisClient
    ConfigClass --> LLMManager
    ConfigClass --> S3Service
    ConfigClass --> ToolRouter
```

**Sources:** [orchestrator/config.py:1-285]()

### Configuration Categories

The `Config` class organizes settings into logical groups:

#### Database Configuration

```python
# PostgreSQL (required)
POSTGRES_DB: str = os.getenv("POSTGRES_DB")
POSTGRES_USER: str = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
DATABASE_URL: str = os.getenv("DATABASE_URL")  # Overrides individual params
```

**Precedence:** If `DATABASE_URL` is set (Railway format), it takes precedence over individual parameters.

**Sources:** [orchestrator/config.py:34-42]()

#### Redis Configuration

Redis configuration supports both URL format (Railway, Heroku) and component variables:

```python
# Component variables (docker-compose.yml, local dev)
REDIS_HOST: str = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT: str = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

@property
def REDIS_URL(self) -> str:
    """Get Redis URL from env or construct from parts"""
    url = os.getenv("REDIS_URL")
    if url:
        return url
    
    if self.REDIS_HOST and self.REDIS_PORT:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    return None
```

**Precedence:**
1. `REDIS_URL` environment variable (Railway format: `redis://:password@host:port/db`)
2. Component variables (`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`)
3. Default host `127.0.0.1`, port `6379`, no password

The `get_redis_client()` function uses this configuration to initialize the connection pool ([orchestrator/core/redis/client.py:149-198]()).

**Sources:** [orchestrator/config.py:46-62]()

#### LLM Configuration (Database-Backed)

LLM settings are loaded from the `system_settings` database table, with environment variable fallbacks:

```python
@property
def LLM_PROVIDER(self) -> str:
    """Get LLM provider from system settings (database) or environment"""
    try:
        from core.llm.manager import get_system_setting
        return get_system_setting("orchestrator_llm", "provider", os.getenv("LLM_PROVIDER"))
    except Exception:
        return os.getenv("LLM_PROVIDER")
```

**Precedence:** Database → Environment → None (no hardcoded defaults)

**Sources:** [orchestrator/config.py:88-106]()

#### AWS S3 Configuration

```python
# Marketplace / Plugin Storage
MARKETPLACE_S3_BUCKET: str = os.getenv("MARKETPLACE_S3_BUCKET", "automatos-marketplace")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
PLUGIN_MAX_UPLOAD_SIZE_MB: int = int(os.getenv("PLUGIN_MAX_UPLOAD_SIZE_MB", "10"))
PLUGIN_CACHE_TTL_SECONDS: int = int(os.getenv("PLUGIN_CACHE_TTL_SECONDS", "3600"))
```

**Sources:** [orchestrator/config.py:158-185]()

#### Feature Flags

```python
ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
S3_VECTORS_ENABLED: bool = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"
```

**Sources:** [orchestrator/config.py:154-174]()

### Environment Variable Precedence

The configuration resolution follows this priority:

1. **Composite URLs** (e.g., `DATABASE_URL`, `REDIS_URL`) - highest priority
   - Format: `postgresql://user:pass@host:port/db`
   - Common in Railway, Heroku deployments
2. **Component Variables** (e.g., `POSTGRES_HOST`, `REDIS_HOST`)
   - Individual connection parameters
   - Used in docker-compose.yml
3. **Database Settings** (LLM configuration only)
   - `system_settings` table via `get_system_setting()`
   - Allows runtime configuration changes
4. **Hardcoded Defaults** (only for non-sensitive values)
   - Example: `REDIS_HOST="127.0.0.1"`, `AWS_REGION="us-east-1"`
   - Secrets (passwords, API keys) have no defaults

#### Variable Categories

| Category | Requires `.env` | Supports Database | Has Defaults |
|----------|-----------------|-------------------|--------------|
| Database credentials | Yes | No | No |
| Redis credentials | Yes (if used) | No | Host/port only |
| LLM configuration | No (optional) | Yes | No |
| AWS credentials | Yes (if S3 used) | No | Region only |
| Feature flags | No | No | Yes |
| Authentication (Clerk) | Yes | No | No |

**Sources:** [orchestrator/config.py:28-285]()

### Configuration Validation

The `validate()` method checks required settings on startup:

```python
def validate(self) -> bool:
    errors = []
    
    # Check database
    if not all([self.POSTGRES_DB, self.POSTGRES_USER, self.POSTGRES_HOST, self.POSTGRES_PORT]):
        errors.append("Database not configured")
    
    # Check API key
    if self.REQUIRE_API_KEY and not self.API_KEY:
        errors.append("API_KEY required when REQUIRE_API_KEY=true")
    
    if errors:
        logger.error("❌ Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True
```

**Usage:** Can be called at application startup to fail fast on misconfiguration.

**Sources:** [orchestrator/config.py:225-247]()

---

## Database Infrastructure

### PostgreSQL with pgvector

The database service uses PostgreSQL 16 with the `pgvector` extension for embedding-based retrieval.

#### Schema Initialization

The schema is automatically loaded on first database creation via Docker's `entrypoint-initdb.d` mechanism:

```yaml
volumes:
  - ./orchestrator/database/init_complete_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
```

**Execution:** Scripts in `/docker-entrypoint-initdb.d/` run only when the data directory is empty (first start).

**Sources:** [docker-compose.yml:34]()

#### Connection Pooling

The backend uses SQLAlchemy with connection pooling to handle concurrent requests efficiently. The database is configured for 200 max connections:

```yaml
POSTGRES_INITDB_ARGS: "-c max_connections=200 -c shared_buffers=256MB"
```

**Rationale:** Supports multiple workflow executions and API requests simultaneously.

**Sources:** [docker-compose.yml:29]()

#### Health Check Strategy

The health check uses `pg_isready` to verify database availability:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U postgres"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 10s
```

**Benefits:**
- **Fast Startup:** 10-second start period prevents premature failures
- **Reliable Detection:** `pg_isready` is more reliable than TCP checks
- **Retry Logic:** 5 retries with 10s interval handles transient failures

**Sources:** [docker-compose.yml:36-40]()

---

## Redis Infrastructure

### Cache Architecture

Redis serves three purposes: caching, real-time messaging, and task queues.

```mermaid
graph TB
    subgraph "Initialization (core/redis/client.py)"
        GetClient["get_redis_client<br/>Line 149"]
        CheckURL["Check config.REDIS_URL<br/>Line 161"]
        ParseURL["urlparse<br/>Lines 164-168"]
        CheckComponents["Fallback:<br/>REDIS_HOST, REDIS_PORT<br/>Lines 184-186"]
        InitClient["init_redis_client<br/>Line 141"]
        TestConn["test_connection<br/>Line 121"]
    end
    
    subgraph "Usage: Caching"
        PluginCache["PluginContentCache<br/>core/services/plugin_cache.py<br/>Key: plugin:content:slug:version"]
        ComposioCache["ComposioAppCache<br/>modules/tools/services/metadata_sync.py<br/>Key: composio:apps:*"]
        RouterCache["RoutingDecisionCache<br/>consumers/router/universal_router.py<br/>Key: routing:cache:*"]
    end
    
    subgraph "Usage: Pub/Sub"
        WorkflowEvents["publish_workflow_event<br/>Line 91<br/>Channel: workflow:ID:execution:ID"]
        TaskEvents["WorkerEventPublisher<br/>workspace-worker/events.py<br/>Channel: workspace:task:ID:events"]
    end
    
    subgraph "Usage: Task Queue"
        TaskQueue["WorkspaceTaskQueue<br/>workspace-worker/task_queue.py<br/>Key: workspace:tasks:pending"]
    end
    
    GetClient --> CheckURL
    CheckURL -->|"REDIS_URL set"| ParseURL
    CheckURL -->|"else"| CheckComponents
    ParseURL --> InitClient
    CheckComponents --> InitClient
    InitClient --> TestConn
    
    TestConn --> PluginCache
    TestConn --> ComposioCache
    TestConn --> RouterCache
    TestConn --> WorkflowEvents
    TestConn --> TaskEvents
    TestConn --> TaskQueue
```

**Sources:** [orchestrator/core/redis/client.py:149-198](), [orchestrator/core/services/plugin_cache.py:1-200]()

### Connection Management

The `RedisClient` class uses a connection pool for efficiency:

```python
class RedisClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 6379, 
                 password: Optional[str] = None, db: int = 0):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
            max_connections=50
        )

    def get_redis(self):
        """Get a Redis connection from the pool"""
        return redis.Redis(connection_pool=self.pool)
```

#### Connection Pool Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_connections` | 50 | Supports concurrent workflows (10-15), API requests (20-30), worker tasks (5-10) |
| `decode_responses` | `True` | Automatic UTF-8 decoding (JSON strings) |
| `db` | 0 | Default database (single-tenant per Redis instance) |

With 4 uvicorn workers, each can use ~12 connections without exhausting the pool.

#### Context Manager for Pub/Sub

```python
@contextmanager
def pubsub_client(self):
    """Context manager for pub/sub client (synchronous)"""
    redis_client = self.get_redis()
    pubsub = redis_client.pubsub()
    try:
        yield pubsub
    finally:
        pubsub.close()
        redis_client.close()
```

Ensures proper cleanup of pub/sub subscriptions ([orchestrator/core/redis/client.py:38-46]()).

**Sources:** [orchestrator/core/redis/client.py:14-46]()

### Lazy Initialization Pattern

Redis is optional—if not configured, the system gracefully degrades:

```python
# Global singleton
_redis_client: Optional[RedisClient] = None

def get_redis_client() -> Optional[RedisClient]:
    """
    Lazy initialization with graceful degradation.
    Returns None if Redis is not configured (optional service).
    """
    global _redis_client
    if _redis_client is None:
        from config import config
        
        redis_url = config.REDIS_URL
        if not redis_url:
            host = config.REDIS_HOST
            port = config.REDIS_PORT
            
            if not host or not port:
                logger.warning("Redis not configured. Redis features disabled.")
                return None
            
            try:
                init_redis_client(host=host, port=int(port), password=config.REDIS_PASSWORD)
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                return None
    
    return _redis_client
```

#### Graceful Degradation Behavior

When Redis is unavailable (`get_redis_client()` returns `None`):
- **Caching:** Skipped, falls back to direct S3/database access
- **Pub/Sub:** Events not published, SSE clients poll instead
- **Task Queues:** Workspace worker tasks unavailable (HTTP API still works)
- **Core Functions:** Agent creation, chat, recipe execution all continue working

#### Callers Check for None

All Redis-dependent code checks for `None`:

```python
redis_client = get_redis_client()
if redis_client:
    redis_client.publish_workflow_event(...)
else:
    logger.debug("Redis unavailable, skipping event publish")
```

**Sources:** [orchestrator/core/redis/client.py:149-198]()

### Plugin Content Caching

The `PluginContentCache` wraps S3 access with Redis caching:

```python
class PluginContentCache:
    CONTENT_PREFIX = "plugin:content:"
    
    async def get_plugin_content(self, slug: str, version: str) -> Dict[str, str]:
        cache_key = f"{self.CONTENT_PREFIX}{slug}:{version}"
        
        # 1. Try cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Plugin content cache HIT for %s@%s", slug, version)
            return json.loads(cached)
        
        # 2. Fetch from S3
        logger.debug("Plugin content cache MISS for %s@%s — fetching from S3", slug, version)
        s3 = self._get_s3()
        file_keys = await s3.list_plugin_files(slug, version)
        
        # Build files dict by downloading each file
        files = {}
        for key in file_keys:
            content = await s3.download_file_content(key)
            filename = key.split('/')[-1]
            files[filename] = content
        
        # 3. Populate cache
        self._cache_set(cache_key, json.dumps(files), ttl=self.ttl)
        return files
```

#### Cache Configuration

| Setting | Default | Environment Variable | Purpose |
|---------|---------|---------------------|---------|
| **TTL** | 3600 seconds (1 hour) | `PLUGIN_CACHE_TTL_SECONDS` | Balance between freshness and S3 costs |
| **Key Pattern** | `plugin:content:{slug}:{version}` | N/A | Unique per plugin version |
| **Size Limit** | 10 MB per plugin | `PLUGIN_MAX_UPLOAD_SIZE_MB` | Prevents cache exhaustion |

#### Performance Impact

Without cache:
- Marketplace page load: 15-20 S3 API calls per plugin
- Cost: $0.0004 per 1000 requests (S3 GET)
- Latency: 50-100ms per plugin

With cache (1 hour TTL):
- Marketplace page load: 0 S3 calls (after first load)
- Cost: Redis memory only
- Latency: <5ms per plugin

**Sources:** [orchestrator/core/services/plugin_cache.py:119-159](), [orchestrator/config.py:180-185]()

### Pub/Sub for Real-Time Updates

Workflow execution events are published to Redis channels for SSE streaming:

```python
def publish_workflow_event(
    self,
    workflow_id: int,
    execution_id: int,
    event_type: str,
    data: Dict[str, Any]
) -> bool:
    channel = f"workflow:{workflow_id}:execution:{execution_id}"
    message = {
        "type": event_type,
        "data": {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            **data
        }
    }
    return self.publish(channel, message)
```

**Channel Naming:** `workflow:{workflow_id}:execution:{execution_id}` allows fine-grained subscriptions.

**Sources:** [orchestrator/core/redis/client.py:91-119]()

---

## Production Deployment

### Railway Deployment

Automatos AI is optimized for Railway deployment with automatic detection of Railway-specific environment variables.

#### PORT Variable Handling

Railway provides a dynamic `PORT` variable that changes on each deployment. The backend Dockerfile handles this:

```dockerfile
# Production command - use PORT env var (Railway provides this)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
```

**Fallback:** If `PORT` is not set (local deployment), defaults to 8000.

**Sources:** [orchestrator/Dockerfile:114-115]()

#### Database URL Format

Railway provides PostgreSQL as `DATABASE_URL` in the connection string format:

```
postgresql://user:password@host:port/database
```

The `Config` class prioritizes `DATABASE_URL` over component variables:

```python
DATABASE_URL: str = os.getenv("DATABASE_URL")  # If set, overrides individual params
```

**Sources:** [orchestrator/config.py:42]()

#### Redis URL Format

Similarly, Railway provides Redis as `REDIS_URL`:

```
redis://:password@host:port/0
```

The `REDIS_URL` property handles URL parsing:

```python
@property
def REDIS_URL(self) -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    
    if self.REDIS_HOST and self.REDIS_PORT:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    return None
```

**Sources:** [orchestrator/config.py:51-62]()

### Environment-Specific Configuration

The `ENVIRONMENT` variable controls behavior across dev/staging/production:

```python
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

@property
def IS_PRODUCTION(self) -> bool:
    return self.ENVIRONMENT.lower() == "production"

@property
def IS_DEVELOPMENT(self) -> bool:
    return self.ENVIRONMENT.lower() == "development"
```

**Usage Examples:**
- **Logging:** More verbose in development
- **CORS:** Stricter in production
- **Error Messages:** Detailed stack traces only in development

**Sources:** [orchestrator/config.py:114-123]()

### CORS Configuration

The backend allows multiple frontend origins via comma-separated list:

```python
_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,https://automotas-ai-frontend-production.up.railway.app")
CORS_ALLOW_ORIGINS: str = ",".join([origin.strip() for origin in _cors_origins.split(",") if origin.strip()])
```

**Railway Default:** Includes both localhost (dev) and Railway frontend domain.

**Sources:** [orchestrator/config.py:72-79]()

### Scaling Considerations

#### Stateless Architecture

Both frontend and backend are stateless, enabling horizontal scaling:

- **Frontend:** Multiple Next.js instances can run behind a load balancer
- **Backend:** Multiple uvicorn workers handle concurrent requests
- **Database:** PostgreSQL with connection pooling handles load
- **Redis:** Single instance sufficient for caching/Pub/Sub at current scale

#### Multi-Worker Backend

The production Dockerfile uses 4 uvicorn workers:

```dockerfile
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4"
```

**Worker Count:** Can be increased via environment variable override if needed.

**Sources:** [orchestrator/Dockerfile:115]()

#### Database Connection Limits

Each uvicorn worker maintains its own connection pool. The database is configured for 200 max connections:

```
max_connections = 200
```

**Calculation:** With 4 workers and ~10 connections per worker, this supports 5 backend instances.

**Sources:** [docker-compose.yml:29]()

### Security Hardening

#### Non-Root Users

Both production images run as non-root users:

**Backend:**
```dockerfile
RUN useradd -m -u 1000 automatos && \
    chown -R automatos:automatos /app
USER automatos
```

**Frontend:**
```dockerfile
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001 && \
    chown -R nextjs:nodejs /app
USER nextjs
```

**Sources:** [orchestrator/Dockerfile:98-101](), [frontend/Dockerfile:103-107]()

#### Secret Management

Secrets are never hardcoded or committed:

- **Development:** `.env` file (git-ignored via `.gitignore`)
- **Production:** Environment variables set in Railway dashboard
- **Credentials:** Stored encrypted in database via `CredentialStore`

**Sources:** [.gitignore:99-105](), [orchestrator/core/credentials/service.py:42-465]()

---

## Local Development Setup

### Quick Start Commands

```bash
# 1. Clone repository
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai

# 2. Copy environment template and configure
cp orchestrator/.env.example orchestrator/.env
# Edit .env file:
#   - Set POSTGRES_PASSWORD (required)
#   - Set REDIS_PASSWORD (required)
#   - Set API_KEY (required for API auth)
#   - Set OPENAI_API_KEY or ANTHROPIC_API_KEY (for LLM features)
#   - Set CLERK_* variables (for authentication)

# 3. Start core services (default profile)
docker-compose up --build

# 4. Or start with workspace worker
docker-compose --profile workers up --build

# 5. Or start with all services (admin tools + worker)
docker-compose --profile all --profile workers up --build
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Clerk sign-in |
| **Backend API Docs** | http://localhost:8000/docs | API key in header |
| **Adminer (DB UI)** | http://localhost:8080 | postgres / POSTGRES_PASSWORD |
| **PostgreSQL** | `localhost:5432` | Connection via client |
| **Redis** | `localhost:6379` | `redis-cli -a REDIS_PASSWORD` |

### Service Startup Order

Due to health check dependencies, services start in this order:
1. **PostgreSQL** (10s start period)
2. **Redis** (5s start period)
3. **Backend** (40s start period) - waits for DB + Redis healthy
4. **Frontend** (60s start period) - waits for backend healthy
5. **Workspace Worker** (15s start period) - waits for DB + Redis healthy (if `--profile workers`)

**Sources:** [docker-compose.yml:1-280](), [README.md:1-150]()

### Manual Service Startup (Non-Docker)

For active development without Docker:

```bash
# 1. Start PostgreSQL and Redis
sudo service postgresql start
sudo service redis-server start

# 2. Initialize database
cd orchestrator
sudo -u postgres psql -d orchestrator_db -f database/init_complete_schema.sql

# 3. Start backend
python3 main.py

# 4. Start frontend (separate terminal)
cd ../frontend
npm install --legacy-peer-deps
npm run dev
```

**Sources:** [docs/LOCAL_SETUP_GUIDE.md:177-199]()

### Port Assignments

| Service | Port | Protocol | Configurable Via |
|---------|------|----------|------------------|
| Frontend | 3000 | HTTP | `FRONTEND_PORT` env var |
| Backend | 8000 | HTTP | `API_PORT` env var |
| PostgreSQL | 5432 | PostgreSQL | `POSTGRES_PORT` env var |
| Redis | 6379 | Redis | `REDIS_PORT` env var |
| Workspace Worker | 8081 | HTTP (health) | Fixed |
| Adminer | 8080 | HTTP | `ADMINER_PORT` env var |
| Gotenberg | 3001 | HTTP | `GOTENBERG_PORT` env var |

All ports can be customized via `.env` file or environment variables.

**Sources:** [docker-compose.yml:32-251]()

---

## Troubleshooting

### Common Deployment Issues

#### Port Conflicts

**Symptom:** `Error: bind: address already in use`

**Solution:**
```bash
# Check what's using the port
netstat -tlnp | grep -E "(8000|3000|5432|6379)"

# Stop conflicting service or change ports in docker-compose.yml
```

**Sources:** [docs/LOCAL_SETUP_GUIDE.md:173-174]()

#### Database Connection Failures

**Symptom:** `sqlalchemy.exc.OperationalError: could not connect to server`

**Solution:**
1. Check PostgreSQL health: `docker-compose ps postgres`
2. Verify health check passes: `docker-compose logs postgres | grep healthy`
3. Check connection string in `.env`: `DATABASE_URL` or `POSTGRES_*` variables

**Sources:** [docker-compose.yml:76-87]()

#### Redis Connection Failures

**Symptom:** `redis.exceptions.ConnectionError: Error connecting to Redis`

**Solution:**
1. Check Redis health: `docker-compose ps redis`
2. Verify password: `docker-compose exec redis redis-cli -a automatos_redis_dev ping`
3. System gracefully degrades if Redis unavailable (caching disabled)

**Sources:** [orchestrator/core/redis/client.py:149-198]()

#### Frontend Build Failures

**Symptom:** `Error: Cannot find module ...` during build

**Solution:**
```bash
# Clear node_modules and rebuild
cd frontend
rm -rf node_modules .next
npm install --legacy-peer-deps
npm run build
```

**Sources:** [docs/LOCAL_SETUP_GUIDE.md:150-153]()

#### Missing Environment Variables

**Symptom:** `Configuration validation failed: API_KEY required`

**Solution:**
1. Copy `.env.example` to `.env`: `cp orchestrator/.env.example orchestrator/.env`
2. Fill in required values (at minimum: LLM API keys)
3. Restart services: `docker-compose restart backend`

**Sources:** [orchestrator/config.py:225-247](), [orchestrator/.env.example:1-64]()

### Health Check Debugging

To debug service health:

```bash
# Check all service health statuses
docker-compose ps

# View health check logs for specific service
docker-compose logs postgres | grep -i health
docker-compose logs backend | grep -i health

# Manually execute health check command
docker-compose exec postgres pg_isready -U postgres
docker-compose exec backend curl -f http://localhost:8000/health
```

**Sources:** [docker-compose.yml:36-121]()

---

## Summary

The Automatos AI deployment infrastructure provides:

1. **Multi-stage Docker builds** for optimized development and production images
2. **Docker Compose orchestration** with health checks and dependency management
3. **Centralized configuration** via `config.py` with database-backed settings
4. **PostgreSQL with pgvector** for relational data and embeddings
5. **Redis caching and Pub/Sub** for performance and real-time updates
6. **Railway-optimized deployment** with dynamic port and URL handling
7. **Security hardening** via non-root users and encrypted credentials

The architecture supports both local development (hot-reload, debug tools) and production deployment (multi-worker, optimized images) with minimal configuration changes.

**Sources:** [orchestrator/Dockerfile:1-116](), [frontend/Dockerfile:1-120](), [docker-compose.yml:1-197](), [orchestrator/config.py:1-285]()

---