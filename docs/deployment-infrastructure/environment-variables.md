# Environment Variables

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [frontend/app/admin/plugins/page.tsx](frontend/app/admin/plugins/page.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/api/agent_plugins.py](orchestrator/api/agent_plugins.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)

</details>



This document describes the centralized environment variable configuration system in Automatos AI. All configuration is managed through a single source of truth (`config.py`) that provides typed access to environment variables across the entire application.

**Related documentation**: For database setup and initialization, see [Database Setup](#15.4). For deployment configuration and Docker Compose usage, see [Docker Compose Setup](#15.2).

---

## Configuration Architecture

Automatos AI uses a **centralized configuration pattern** where all environment variable access is isolated to a single module: [orchestrator/config.py:1-424](). This design eliminates scattered `os.getenv()` calls throughout the codebase and provides type-safe configuration access with defaults and validation.

### Configuration Loading Flow

```mermaid
graph TD
    EnvFile[".env file<br/>(root directory)"]
    DockerCompose["docker-compose.yml<br/>environment section"]
    RailwayVars["Railway Variables<br/>(Production)"]
    
    LoadDotenv["load_dotenv()<br/>config.py:26"]
    ConfigClass["Config class<br/>config.py:28-423"]
    ConfigSingleton["config singleton<br/>config.py:413"]
    
    MainApp["main.py<br/>Import config"]
    Services["Service modules<br/>(factories, routers, etc)"]
    
    EnvFile -->|"development"| LoadDotenv
    DockerCompose -->|"docker"| ConfigClass
    RailwayVars -->|"production"| ConfigClass
    
    LoadDotenv --> ConfigClass
    ConfigClass --> ConfigSingleton
    ConfigSingleton --> MainApp
    ConfigSingleton --> Services
    
    ConfigClass -->|"validates"| Required["Required variables:<br/>POSTGRES_*, API_KEY"]
    ConfigClass -->|"provides defaults"| Optional["Optional variables:<br/>LLM_*, Feature flags"]
```

**Sources**: [orchestrator/config.py:1-424](), [orchestrator/main.py:24-29](), [docker-compose.yml:1-280]()

### Single Source of Truth Pattern

The `Config` class in [orchestrator/config.py:28-423]() is the **only** location where `os.getenv()` is called. All other modules import from `config`:

```python
# CORRECT - Import from centralized config
from config import config
database_url = config.DATABASE_URL

# INCORRECT - Never do this elsewhere
import os
database_url = os.getenv("DATABASE_URL")  # ❌ Scattered config access
```

This pattern provides:
- **Type safety**: Properties return typed values (str, int, bool)
- **Validation**: Required variables are checked on startup
- **Defaults**: Sensible defaults for optional variables
- **Documentation**: All variables documented in one place
- **Testing**: Easy to mock configuration

**Sources**: [orchestrator/config.py:1-10]()

---

## Environment Variable Categories

### Database Configuration

PostgreSQL is the primary data store. These variables configure the connection:

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `POSTGRES_DB` | string | ✅ | - | Database name |
| `POSTGRES_USER` | string | ✅ | - | Database user |
| `POSTGRES_PASSWORD` | string | ✅ | - | Database password |
| `POSTGRES_HOST` | string | ✅ | - | Database host |
| `POSTGRES_PORT` | string | ✅ | - | Database port (typically 5432) |
| `DATABASE_URL` | string | ❌ | (constructed) | Full connection URL (overrides individual params) |
| `SQL_DEBUG` | bool | ❌ | `false` | Enable SQL query logging |

**SSL Enforcement**: The `get_database_url()` method [orchestrator/config.py:48-58]() automatically appends `sslmode=require` for non-local hosts (production databases), implementing **PRD-70 FIX-05** security requirements.

**Sources**: [orchestrator/config.py:37-58](), [docker-compose.yml:91-97]()

### Redis Configuration

Redis is optional but enables caching, pub/sub, and task queues:

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `REDIS_HOST` | string | ❌ | - | Redis host |
| `REDIS_PORT` | string | ❌ | - | Redis port (typically 6379) |
| `REDIS_PASSWORD` | string | ❌ | - | Redis password (recommended) |
| `REDIS_DB` | string | ❌ | `0` | Redis database number |
| `REDIS_URL` | string | ❌ | (constructed) | Full Redis URL (Railway/Heroku format) |

The `REDIS_URL` property [orchestrator/config.py:68-79]() constructs the URL from components if not explicitly provided. Redis clients use lazy initialization [orchestrator/core/redis/client.py:149-198]() and gracefully degrade if Redis is unavailable.

**PRD-70 Security**: Docker Compose [docker-compose.yml:54-62]() disables dangerous commands (`FLUSHDB`, `FLUSHALL`, `DEBUG`) via `--rename-command` to prevent data loss.

**Sources**: [orchestrator/config.py:62-79](), [orchestrator/core/redis/client.py:149-198](), [docker-compose.yml:48-73]()

### API Security

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `API_KEY` | string | ⚠️ | - | Backend API key (required if `REQUIRE_API_KEY=true`) |
| `ORCHESTRATOR_API_KEY` | string | ❌ | (falls back to `API_KEY`) | Alias for `API_KEY` |
| `REQUIRE_API_KEY` | bool | ❌ | `true` | Enforce API key validation |
| `REQUIRE_AUTH` | bool | ❌ | `true` | Enforce Clerk JWT authentication |
| `AUTH_DEBUG` | bool | ❌ | `false` | Enable detailed auth logging |

**Authentication Architecture**: When `REQUIRE_AUTH=true`, all API endpoints validate Clerk JWT tokens via middleware [orchestrator/main.py:632-641](). The `get_request_context_hybrid()` dependency [orchestrator/core/auth/hybrid.py](referenced in routes) extracts workspace context from JWT claims.

**Sources**: [orchestrator/config.py:84-89](), [orchestrator/main.py:632-641]()

### CORS Configuration

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `CORS_ALLOW_ORIGINS` | string | ❌ | `http://localhost:3000` | Comma-separated list of allowed origins |

The value is parsed and cleaned [orchestrator/main.py:557-558]() to handle whitespace in comma-separated lists. For Railway deployment, include your frontend domain (e.g., `https://automotas-ai-frontend-production.up.railway.app`).

**Sources**: [orchestrator/config.py:98-99](), [orchestrator/main.py:556-567]()

### LLM Provider Keys

Automatos AI supports 8+ LLM providers. Keys are optional at startup but required for using each provider:

| Variable | Type | Provider | Description |
|----------|------|----------|-------------|
| `OPENAI_API_KEY` | string | OpenAI | GPT-4, GPT-3.5-turbo models |
| `ANTHROPIC_API_KEY` | string | Anthropic | Claude 3 models (Opus, Sonnet, Haiku) |
| `OPENROUTER_API_KEY` | string | OpenRouter | Access to 100+ models via unified API |
| `GOOGLE_API_KEY` | string | Google | Gemini models (aliases: `GEMINI_API_KEY`) |
| `AZURE_OPENAI_API_KEY` | string | Azure OpenAI | Azure-hosted OpenAI models |
| `AZURE_OPENAI_ENDPOINT` | string | Azure OpenAI | Azure endpoint URL |
| `AZURE_OPENAI_API_VERSION` | string | Azure OpenAI | API version (default: `2024-02-15-preview`) |
| `XAI_API_KEY` | string | xAI | Grok models |
| `COHERE_API_KEY` | string | Cohere | Rerank API for RAG |

**Dynamic LLM Settings**: `LLM_PROVIDER` and `LLM_MODEL` are loaded from the database `system_settings` table [orchestrator/config.py:116-134]() rather than hardcoded defaults, allowing runtime configuration changes via the Settings UI.

**Sources**: [orchestrator/config.py:104-136](), [orchestrator/requirements.txt:66-70]()

### Environment & Logging

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `ENVIRONMENT` | string | ❌ | `development` | Runtime environment (`development`, `production`) |
| `LOG_LEVEL` | string | ❌ | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

Properties `IS_PRODUCTION` and `IS_DEVELOPMENT` [orchestrator/config.py:145-150]() provide boolean checks for environment-specific logic.

**Sources**: [orchestrator/config.py:141-150]()

### Authentication (Clerk)

Clerk provides JWT-based authentication with workspace isolation:

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `CLERK_SECRET_KEY` | string | ✅ | Clerk backend secret key (validates JWTs) |
| `CLERK_PUBLISHABLE_KEY` | string | ✅ | Clerk frontend publishable key |
| `CLERK_JWKS_URL` | string | ❌ | JWKS endpoint (auto-configured if not provided) |
| `CLERK_AUDIENCE` | string | ❌ | Expected JWT audience |
| `DEFAULT_WORKSPACE_ID` | string | ❌ | Fallback workspace for single-tenant mode |
| `WORKSPACE_ID` | string | ❌ | Override workspace (for scripts/workers) |
| `CREDENTIAL_ENCRYPTION_KEY` | string | ✅ | Fernet key for encrypting stored credentials |

**Workspace Context Flow**:

```mermaid
graph LR
    Request["HTTP Request<br/>Authorization: Bearer {jwt}"]
    ClerkMiddleware["Clerk JWT Validation<br/>hybrid.py"]
    ExtractClaims["Extract Claims<br/>(user_id, org_id)"]
    WorkspaceHeader["X-Workspace-ID header<br/>(optional)"]
    RequestContext["RequestContext<br/>(workspace_id, user_id)"]
    
    Request --> ClerkMiddleware
    Request --> WorkspaceHeader
    ClerkMiddleware --> ExtractClaims
    ExtractClaims --> RequestContext
    WorkspaceHeader --> RequestContext
    
    RequestContext --> DatabaseQuery["Database queries filtered<br/>by workspace_id"]
```

**Sources**: [orchestrator/config.py:169-176](), [orchestrator/main.py:632-641]()

### URLs (Backend, Frontend, API)

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `BACKEND_URL` | string | ❌ | `http://localhost:8000` | Backend API base URL |
| `FRONTEND_URL` | string | ❌ | `http://localhost:3000` | Frontend application URL |
| `API_URL` | string | ❌ | `http://localhost:8000` | External API URL (for docs) |
| `NEXT_PUBLIC_API_URL` | string | ✅ | - | Frontend env: API URL (embedded in client bundle) |

**Frontend Build-Time Variables**: `NEXT_PUBLIC_*` variables [frontend/Dockerfile:58-71]() are baked into the Next.js client bundle during build. They **must not contain secrets** (see [frontend/Dockerfile:56]() warning).

**Sources**: [orchestrator/config.py:180-182](), [frontend/Dockerfile:56-71]()

### External Service URLs

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENROUTER_BASE_URL` | string | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `OPENROUTER_SITE_URL` | string | `https://automatos.app` | Site URL for OpenRouter attribution |
| `COHERE_RERANK_URL` | string | `https://api.cohere.com/v2/rerank` | Cohere rerank endpoint |
| `RAILWAY_GQL_URL` | string | `https://backboard.railway.app/graphql/v2` | Railway GraphQL API |
| `INTERNAL_API_HOSTNAME` | string | `automatos-ai.railway.internal` | Railway private network hostname |
| `SKILLS_SH_API_URL` | string | `https://api.skills.sh/v1` | Skills.sh API endpoint |
| `COMPOSIO_API_KEY` | string | - | Composio integration key |
| `COMPOSIO_API_BASE_URL` | string | `https://backend.composio.dev/api/v3` | Composio API endpoint |

**Sources**: [orchestrator/config.py:187-196]()

### Routing (Universal Orchestrator Router)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `COMPOSIO_WEBHOOK_SECRET` | string | - | Webhook signature validation |
| `ROUTING_CACHE_TTL_HOURS` | int | `24` | Routing decision cache TTL |
| `ROUTING_LLM_CONFIDENCE_THRESHOLD` | float | `0.5` | Minimum confidence for LLM routing |

**Sources**: [orchestrator/config.py:199-203]()

### GitHub Integration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GITHUB_REPO_OWNER` | string | - | Default repository owner |
| `GITHUB_REPO_NAME` | string | - | Default repository name |
| `GITHUB_DEFAULT_BRANCH` | string | `main` | Default branch |
| `GITHUB_WEBHOOK_SECRET` | string | - | GitHub webhook signature validation |
| `GITHUB_WEBHOOK_WORKSPACE_ID` | string | - | Workspace for GitHub webhooks |
| `GITHUB_PR_WORKFLOW_NAME` | string | `PR Code Review` | Workflow triggered on PR events |

**Sources**: [orchestrator/config.py:205-211]()

### Task Runner & Physical Workspaces

**PRD-56** introduces sandboxed workspace execution:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TASK_RUNNER_BACKEND` | string | `local` | Backend type (`local`, `queued`, `kubernetes`) |
| `WORKSPACE_VOLUME_PATH` | string | `/workspaces` | Filesystem path for workspace directories |
| `WORKSPACE_DEFAULT_QUOTA_GB` | int | `5` | Default storage quota per workspace |
| `WORKER_CONCURRENCY` | int | `3` | Max concurrent tasks per worker |
| `WORKER_INTERNAL_URL` | string | `http://localhost:8081` | Workspace worker HTTP endpoint |
| `WORKER_INTERNAL_TOKEN` | string | - | Worker authentication token |

**Workspace Worker Architecture**:

```mermaid
graph TB
    Backend["FastAPI Backend<br/>(orchestrator)"]
    Worker["Workspace Worker<br/>(services/workspace-worker)"]
    Volume["Docker Volume<br/>workspace_data"]
    Redis["Redis Task Queue"]
    
    Backend -->|"Submit task via HTTP"| Worker
    Backend -->|"Enqueue task"| Redis
    Worker -->|"Poll queue"| Redis
    Worker -->|"Read/write files"| Volume
    Backend -->|"Read files (view only)"| Volume
    
    Volume -->|"Isolation"| WS1["/workspaces/ws-{uuid}"]
    Volume -->|"Isolation"| WS2["/workspaces/ws-{uuid}"]
```

**Sources**: [orchestrator/config.py:220-227](), [docker-compose.yml:174-217]()

### Railway API

Access to Railway service logs for debugging:

| Variable | Type | Description |
|----------|------|-------------|
| `RAILWAY_API_TOKEN` | string | Railway API token |
| `RAILWAY_PROJECT_ID` | string | Project UUID |
| `RAILWAY_ENVIRONMENT_ID` | string | Environment UUID |

**Sources**: [orchestrator/config.py:230-233]()

### Feature Flags

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_BATCH_API` | bool | `false` | Enable batch processing API |
| `HEARTBEAT_ENABLED` | bool | `true` | Enable heartbeat service |
| `RECIPE_SCHEDULER_ENABLED` | bool | `true` | Enable cron scheduler for recipes |
| `CHANNELS_ENABLED` | bool | `true` | Enable channel manager (PRD-55) |
| `SEMANTIC_TOOL_ROUTING` | bool | `true` | Enable semantic tool discovery |

**Startup Integration**: Feature flags control service initialization in [orchestrator/main.py:338-365]() lifespan context.

**Sources**: [orchestrator/config.py:238-243](), [orchestrator/main.py:338-365]()

### AWS S3 Vectors (PRD-42)

Cloud document storage with vector search:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AWS_REGION` | string | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | string | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | string | - | AWS secret key |
| `S3_VECTORS_ENABLED` | bool | `false` | Enable S3 vector storage |
| `S3_VECTORS_BUCKET` | string | - | S3 bucket for vectors |
| `S3_VECTORS_INDEX_NAME` | string | `documents-index` | Index name |
| `S3_VECTORS_DIMENSION` | int | `2048` | Vector dimension |
| `S3_VECTORS_METRIC` | string | `cosine` | Distance metric |
| `S3_DOCUMENTS_BUCKET` | string | `automatos-ai` | General document storage bucket |

**Sources**: [orchestrator/config.py:247-260]()

### FutureAGI Integration (PRD-58)

Prompt scoring and optimization:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FUTUREAGI_API_KEY` | string | - | FutureAGI API key |
| `FUTUREAGI_SECRET_KEY` | string | - | FutureAGI secret key |
| `FUTUREAGI_ENABLED` | bool | `false` | Enable FutureAGI evaluation |
| `AGENT_OPT_WORKER_URL` | string | `http://agent-opt-worker.railway.internal:8080` | Worker service URL |

**Sources**: [orchestrator/config.py:264-268]()

### Marketplace & Plugin System

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MARKETPLACE_S3_BUCKET` | string | `automatos-marketplace` | S3 bucket for plugin files |
| `MARKETPLACE_LOCAL_DIR` | string | - | Local directory fallback (dev) |
| `PLUGIN_MAX_UPLOAD_SIZE_MB` | int | `10` | Max plugin upload size |
| `PLUGIN_LLM_SCAN_MODEL` | string | `claude-haiku-4-20250414` | Model for security scanning |
| `PLUGIN_CACHE_TTL_SECONDS` | int | `3600` | Plugin content cache TTL |

**Plugin Content Caching**: [orchestrator/core/services/plugin_cache.py:1-263]() wraps S3 access with Redis caching to reduce latency.

**Sources**: [orchestrator/config.py:278-284](), [orchestrator/core/services/plugin_cache.py:22-48]()

### Recipe Execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RECIPE_SCRATCHPAD_TTL` | int | `3600` | Scratchpad TTL (seconds) |
| `RECIPE_LOG_RETENTION_DAYS` | int | `30` | Log retention period |
| `RECIPE_LOG_S3_BUCKET` | string | `automatos-ai` | S3 bucket for execution logs |
| `MEM0_API_URL` | string | `http://automatos-mem0-server.railway.internal` | Mem0 memory service |
| `MEM0_API_KEY` | string | - | Mem0 authentication key |

**Sources**: [orchestrator/config.py:288-293]()

### Embeddings

| Variable | Type | Description |
|----------|------|-------------|
| `EMBEDDING_PROVIDER` | string | Provider (e.g., `openai`, `cohere`) |
| `EMBEDDING_MODEL` | string | Model name (e.g., `text-embedding-ada-002`) |
| `VECTOR_STORE_DIMENSIONS` | int | Vector dimension (default: `2048`) |

**Sources**: [orchestrator/config.py:297-300]()

### PandasAI

Data analysis and chart generation:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PANDASAI_API_KEY` | string | - | PandasAI API key |
| `PANDASAI_MODEL` | string | - | Model for data queries |
| `PANDASAI_OUTPUT_DIR` | string | `/tmp/pandasai_charts` | Chart output directory |

**Sources**: [orchestrator/config.py:304-307]()

### Agent Execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTOMATOS_WORKSPACE` | string | `/tmp/automatos_workspace` | Temporary workspace |
| `AUTOMATOS_RESULTS_BASE` | string | `/var/lib/automatos/results` | Results storage |
| `AUTOMATOS_DOCUMENTS_DIRS` | string | - | Document search paths |
| `IMAGE_STORE_LOCAL_DIR` | string | - | Generated image storage |
| `GOTENBERG_URL` | string | `http://gotenberg:3000` | Document conversion service |
| `DOCUMENT_STORAGE_DIR` | string | `documents` | Document storage path |

**Gotenberg Integration**: [docker-compose.yml:242-253]() runs a Gotenberg sidecar for PDF generation (PRD-63).

**Sources**: [orchestrator/config.py:312-317](), [docker-compose.yml:242-253]()

### RAG Settings (Database-Backed)

These settings are loaded from the `system_settings` table [orchestrator/config.py:334-362]() rather than environment variables, allowing runtime configuration:

| Setting | Property | Default | Description |
|---------|----------|---------|-------------|
| `rag.min_similarity` | `RAG_MIN_SIMILARITY` | `0.65` | Minimum similarity threshold for retrieval |
| `rag.top_k` | `RAG_TOP_K` | `5` | Default number of results |
| `rag.rerank_enabled` | `RAG_RERANK_ENABLED` | `false` | Enable Cohere reranking |

**Sources**: [orchestrator/config.py:334-362]()

---

## Setting Environment Variables

### Local Development

1. Copy the example file:
   ```bash
   cp orchestrator/.env.example .env
   ```

2. Edit `.env` with your values:
   ```bash
   # Required
   POSTGRES_PASSWORD=your_secure_password
   REDIS_PASSWORD=your_redis_password
   API_KEY=your_api_key
   
   # Optional - Add LLM keys as needed
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Start services:
   ```bash
   docker-compose up --build
   ```

**Sources**: [orchestrator/.env.example:1-65]()

### Docker Compose

Environment variables are defined in [docker-compose.yml:86-122]() under each service's `environment` section:

```yaml
backend:
  environment:
    POSTGRES_HOST: postgres
    POSTGRES_PORT: 5432
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    REDIS_HOST: redis
    OPENAI_API_KEY: ${OPENAI_API_KEY:-}  # Optional with fallback
```

Variables prefixed with `${VAR}` are read from the `.env` file or shell environment. The `:?` syntax requires a variable; `:-` provides a default.

**Sources**: [docker-compose.yml:78-138]()

### Railway Deployment

Set variables in the Railway dashboard under **Variables**:

1. **Service Variables** (specific to a service):
   - `backend` service: `POSTGRES_PASSWORD`, `API_KEY`, `OPENAI_API_KEY`, etc.
   - `frontend` service: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`

2. **Shared Variables** (available to all services):
   - Database connection strings
   - Redis URL
   - AWS credentials

3. **Private Networking**: Use Railway's internal hostnames:
   ```bash
   # Backend to Worker communication
   WORKER_INTERNAL_URL=http://workspace-worker.railway.internal:8081
   
   # Backend to Redis
   REDIS_URL=redis://:password@redis.railway.internal:6379/0
   ```

**Frontend Build-Time Variables**: In Railway frontend service settings, ensure `NEXT_PUBLIC_API_URL` points to the backend public URL (e.g., `https://automatos-ai-production.up.railway.app`).

**Sources**: [orchestrator/config.py:187-192]()

---

## Validation & Error Handling

### Startup Validation

The `Config.validate()` method [orchestrator/config.py:363-386]() checks required variables on application startup:

```mermaid
graph TD
    Startup["Application Startup"]
    Validate["config.validate()"]
    CheckDB["Check Database Config"]
    CheckAPIKey["Check API Key"]
    
    Startup --> Validate
    Validate --> CheckDB
    Validate --> CheckAPIKey
    
    CheckDB -->|"Missing"| LogError["Log Error:<br/>POSTGRES_* required"]
    CheckAPIKey -->|"Missing + REQUIRE_API_KEY=true"| LogError
    
    LogError --> NonFatal["Continue Startup<br/>(non-blocking)"]
    
    CheckDB -->|"Valid"| Success["Validation Success"]
    CheckAPIKey -->|"Valid"| Success
```

**Non-Blocking Validation**: Validation warnings are logged but do not halt startup [orchestrator/config.py:419-420](). This allows the application to start even if optional services (Redis, LLM keys) are not configured.

**Sources**: [orchestrator/config.py:363-420]()

### Runtime Error Handling

Services using environment variables implement graceful degradation:

1. **Redis Client** [orchestrator/core/redis/client.py:149-198]():
   - Returns `None` if Redis is not configured
   - Logs warning: `"Redis not configured... Redis features disabled"`
   - Workflow pub/sub and caching become no-ops

2. **LLM Manager** [orchestrator/core/llm/manager.py]():
   - Falls back through 6-level credential resolution
   - Tries: workspace creds → user creds → global creds → env vars → system settings → error

3. **S3 Services**:
   - Check `AWS_ACCESS_KEY_ID` before attempting S3 operations
   - Raise explicit errors if keys missing

**Sources**: [orchestrator/core/redis/client.py:149-198]()

### Debug Configuration

The `Config.print_config()` method [orchestrator/config.py:387-410]() prints a summary with optional secret masking:

```python
from config import config

# Print config (secrets masked)
config.print_config(show_secrets=False)
```

Output:
```
============================================================
AUTOMATOS AI CONFIGURATION
============================================================
Environment: development
Database: orchestrator_db@localhost:5432
Redis: localhost:6379
LLM Provider: openai (gpt-4)
OpenAI Key: ✅ Set
Anthropic Key: ❌ Not set
API Key: ✅ Set
API Key Required: True
============================================================
```

**Sources**: [orchestrator/config.py:387-410]()

---

## Environment-Specific Configurations

### Development Mode

```bash
# .env for local development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
REQUIRE_AUTH=false  # Disable Clerk for local testing
SQL_DEBUG=true      # Enable query logging

# Use local services
POSTGRES_HOST=localhost
REDIS_HOST=localhost
```

**Hot Reload**: [orchestrator/Dockerfile:85]() uses `--reload` flag in development target. Frontend hot-reload is enabled by mounting source code [docker-compose.yml:163-168]().

**Sources**: [orchestrator/Dockerfile:56-86](), [docker-compose.yml:141-170]()

### Production Mode

```bash
# Railway environment variables
ENVIRONMENT=production
LOG_LEVEL=INFO
REQUIRE_AUTH=true
SQL_DEBUG=false

# Railway-provided variables
DATABASE_URL=postgresql://user:pass@db.railway.internal/db?sslmode=require
REDIS_URL=redis://:pass@redis.railway.internal:6379/0

# Disable dev-only features
ENABLE_BATCH_API=false
```

**Production Optimizations**:
- Multi-stage Docker builds [orchestrator/Dockerfile:89-130]() remove dev dependencies
- Uvicorn runs with 4 workers [orchestrator/Dockerfile:129]()
- Next.js uses standalone output [frontend/Dockerfile:84-114]() for minimal image size
- Swagger UI disabled [orchestrator/main.py:533-535]() when `ENVIRONMENT=production`

**Sources**: [orchestrator/Dockerfile:89-130](), [frontend/Dockerfile:84-114](), [orchestrator/main.py:533-535]()

---

## Security Best Practices

### Credential Storage

1. **Never commit `.env` to git**: Listed in `.gitignore`
2. **Encrypt credentials at rest**: Use `CREDENTIAL_ENCRYPTION_KEY` for database-stored credentials
3. **Use secrets management**: Railway Variables, AWS Secrets Manager, or HashiCorp Vault for production
4. **Rotate keys regularly**: Especially `API_KEY`, `CLERK_SECRET_KEY`, and LLM keys

**Sources**: [orchestrator/config.py:176]()

### Frontend Environment Variables

⚠️ **CRITICAL**: `NEXT_PUBLIC_*` variables are **embedded in the client-side JavaScript bundle** during build [frontend/Dockerfile:58-71](). They are visible to all users and **must not contain secrets**.

**Safe for `NEXT_PUBLIC_`**:
- API URLs: `NEXT_PUBLIC_API_URL`
- Clerk publishable key: `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`

**Never use `NEXT_PUBLIC_` for**:
- API keys
- Secret keys
- Database credentials
- Internal URLs

**Sources**: [frontend/Dockerfile:56-71]()

### SSL/TLS

- **Database**: `get_database_url()` [orchestrator/config.py:48-58]() enforces `sslmode=require` for production
- **Redis**: Use `rediss://` (TLS) in production for `REDIS_URL`
- **HTTP**: Railway provides automatic HTTPS termination

**Sources**: [orchestrator/config.py:48-58]()

---

## Troubleshooting

### Common Issues

**"Configuration validation failed"**
- Check [orchestrator/main.py:419-420]() logs for missing variables
- Run `config.validate()` to see specific errors
- Verify `.env` file is in the correct location

**"Redis not configured"**
- Redis is optional; system will degrade gracefully
- Check `REDIS_HOST` and `REDIS_PORT` are set
- Verify Redis container is running: `docker ps | grep redis`

**"Database connection failed"**
- Verify `POSTGRES_*` variables are set correctly
- Check database container health: `docker-compose ps postgres`
- Test connection: `psql -h localhost -U postgres -d orchestrator_db`

**"NEXT_PUBLIC_API_URL not set"**
- Check Railway frontend service variables
- Verify URL points to backend public URL (not internal hostname)
- Rebuild frontend after changing: `docker-compose build frontend`

**Sources**: [orchestrator/config.py:363-386](), [orchestrator/main.py:419-420]()

---

## Reference: Complete Variable List

See the following files for comprehensive examples:

- **Backend variables**: [orchestrator/.env.example:1-65]()
- **Docker Compose mapping**: [docker-compose.yml:86-203]()
- **Frontend build args**: [frontend/Dockerfile:58-71]()
- **Config class properties**: [orchestrator/config.py:28-423]()

**Configuration Precedence**:
1. Environment variables (highest)
2. `.env` file
3. `config.py` defaults (lowest)

**Sources**: [orchestrator/.env.example:1-65](), [orchestrator/config.py:1-424](), [docker-compose.yml:1-280]()

---