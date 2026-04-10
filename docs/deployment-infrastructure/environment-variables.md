# Environment Variables

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitignore](.gitignore)
- [docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md](docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md)
- [infrastructure/.env.example](infrastructure/.env.example)
- [infrastructure/docker-compose.core.yml](infrastructure/docker-compose.core.yml)
- [infrastructure/docker-compose.data.yml](infrastructure/docker-compose.data.yml)
- [infrastructure/docker-compose.landing.yml](infrastructure/docker-compose.landing.yml)
- [infrastructure/docker-compose.memory.yml](infrastructure/docker-compose.memory.yml)
- [infrastructure/docker-compose.monitoring.yml](infrastructure/docker-compose.monitoring.yml)
- [infrastructure/docker-compose.voice.yml](infrastructure/docker-compose.voice.yml)
- [infrastructure/docker-compose.yml](infrastructure/docker-compose.yml)
- [infrastructure/railway-manifest.json](infrastructure/railway-manifest.json)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/core/credentials/service.py](orchestrator/core/credentials/service.py)
- [orchestrator/core/models/credentials.py](orchestrator/core/models/credentials.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)

</details>



This document describes the environment variable configuration system used across all Automatos AI services. Environment variables control database connections, external service credentials, service discovery, and deployment-specific settings.

For deployment infrastructure, see [Production Deployment](20.6). For credential management in the UI, see [Credentials Management](17.5).

---

## Overview

Automatos AI uses environment variables for all external configuration to support modular deployment via Docker Compose and Railway. Variables are loaded from `.env` files in local development and from platform-provided environments in production.

The system follows a three-tier loading strategy:

1.  **Environment variables** (highest priority) — set by hosting platform or shell.
2.  **`.env` file** — loaded via `python-dotenv` in the backend application lifecycle.
3.  **Hardcoded defaults** — fallback values in centralized config.

**Sources:** [orchestrator/.env.example:1-65](), [infrastructure/docker-compose.core.yml:28-112](), [infrastructure/railway-manifest.json:63-93]()

---

## Environment Variable Loading Flow

The following diagram illustrates how configuration flows from environment sources into the core system entities.

**Diagram: Configuration Injection Pipeline**
```mermaid
graph TB
    subgraph "Environment_Sources"
        EnvFile[".env file"]
        ComposeEnv["docker-compose.core.yml<br/>environment section"]
        PlatformEnv["Railway Manifest<br/>(env_keys)"]
    end
    
    subgraph "Code_Entity_Config_Manager"
        ConfigModule["main.py<br/>(FastAPI App)"]
    end
    
    EnvFile --> ConfigModule
    ComposeEnv --> ConfigModule
    PlatformEnv --> ConfigModule
    
    subgraph "System_Consumers"
        DB["PostgreSQL / pgvector<br/>(infrastructure/docker-compose.data.yml)"]
        RedisSvc["Redis Cache/Queue<br/>(infrastructure/docker-compose.data.yml)"]
        LLM["LLMManager<br/>(API Providers)"]
        Auth["Clerk Auth Service<br/>(JWT Verification)"]
    end
    
    ConfigModule --> DB
    ConfigModule --> RedisSvc
    ConfigModule --> LLM
    ConfigModule --> Auth
```

**Sources:** [infrastructure/docker-compose.core.yml:19-112](), [infrastructure/docker-compose.data.yml:19-98](), [infrastructure/railway-manifest.json:63-93]()

---

## Required Environment Variables

These variables **must** be set for the system to function. Missing required variables will cause startup failures.

### Core Infrastructure

| Variable | Purpose | Example | Used By |
| :--- | :--- | :--- | :--- |
| `POSTGRES_PASSWORD` | PostgreSQL admin password | `secure_db_pass_123` | `pgvector` container, `automatos-ai-api` |
| `REDIS_PASSWORD` | Redis authentication | `secure_redis_pass` | `redis` container, `automatos-ai-api` |
| `API_KEY` | Internal API authentication | `automatos_api_key_xyz` | `automatos-ai-api`, `automotas-ai-frontend` |
| `CREDENTIAL_ENCRYPTION_KEY` | Key for encrypting stored user credentials | `base64_encoded_key` | `CredentialStore` |

**Sources:** [orchestrator/.env.example:6,11,16](), [infrastructure/docker-compose.core.yml:33,39,59,61](), [orchestrator/core/credentials/service.py:56]()

### Authentication (Clerk)

| Variable | Purpose | Example |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk public key (client-side) | `pk_test_...` |
| `CLERK_SECRET_KEY` | Clerk secret key (server-side) | `sk_test_...` |
| `CLERK_JWKS_URL` | Clerk JWT verification endpoint | `https://clerk.../jwks` |

**Security Note:** `NEXT_PUBLIC_*` variables are embedded in the client bundle during the Next.js build process. Only the publishable key should have this prefix.

**Sources:** [infrastructure/docker-compose.core.yml:55-57](), [infrastructure/railway-manifest.json:117-119]()

---

## Database and Cache Configuration

### PostgreSQL with pgvector
The system uses `pgvector` for semantic search and memory storage.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `POSTGRES_HOST` | `localhost` | PostgreSQL server hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `orchestrator_db` | Database name |
| `DATABASE_URL` | (required) | Full SQLAlchemy connection string |

**Sources:** [orchestrator/.env.example:2-6](), [infrastructure/docker-compose.core.yml:30-34](), [infrastructure/docker-compose.data.yml:31-35]()

### Redis Configuration
Redis serves as the L1 memory tier, Pub/Sub broker, and task queue.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `REDIS_HOST` | `redis` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_URL` | (required) | Full connection URL (e.g., `redis://default:pass@host:6379`) |

**Sources:** [orchestrator/.env.example:9-11](), [infrastructure/docker-compose.core.yml:36-39](), [infrastructure/docker-compose.data.yml:65-66]()

---

## LLM Provider Configuration

Automatos AI supports a multi-provider strategy. While variables can be set in the environment, the system also supports a dynamic **Credential Store** for per-workspace keys.

**Diagram: LLM API Key Resolution**
```mermaid
graph TD
    subgraph "Request_Context"
        Req["Agent Execution Request"]
    end

    subgraph "Resolution_Logic_CredentialStore"
        Store["CredentialStore.create_credential()<br/>(core/credentials/service.py)"]
        Env["os.getenv('OPENAI_API_KEY', ...)"]
    end

    Req --> Store
    Store -- "Not Found" --> Env
    Env -- "Found" --> Provider["LLM Provider Client<br/>(OpenAI/Anthropic/Gemini)"]
    Store -- "Found (Encrypted)" --> Decrypt["EncryptionService.encrypt_dict()"]
    Decrypt --> Provider
```

| Variable | Purpose |
| :--- | :--- |
| `OPENAI_API_KEY` | Key for OpenAI models |
| `ANTHROPIC_API_KEY` | Key for Anthropic models |
| `GOOGLE_API_KEY` | Key for Google Gemini models |
| `OPENROUTER_API_KEY` | Key for OpenRouter (aggregator) |
| `LLM_PROVIDER` | Default provider (e.g., `anthropic`) |
| `LLM_MODEL` | Default model (e.g., `claude-sonnet-4-20250514`) |

**Encryption:** Credentials stored in the database are encrypted using `encryption_service.encrypt_dict()` before being persisted in the `Credential` model.

**Sources:** [orchestrator/.env.example:18-24](), [infrastructure/docker-compose.core.yml:41-48](), [orchestrator/core/credentials/service.py:147-153]()

---

## AWS and Storage Configuration

Used for **Cloud Document Sync** and **Marketplace** asset storage.

| Variable | Purpose |
| :--- | :--- |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 and S3 Vectors |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 and S3 Vectors |
| `AWS_REGION` | Target AWS region (e.g., `us-east-1`) |
| `S3_VECTORS_ENABLED` | Toggle for using S3 as a vector storage backend |
| `S3_VECTORS_BUCKET` | Bucket name for vector storage |

**Sources:** [orchestrator/.env.example:48-51,61-64](), [infrastructure/docker-compose.core.yml:87-91]()

---

## Service-Specific Configuration

### Universal Router and Webhooks
| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `COMPOSIO_KEY` | API key for Composio tool integration | (none) |
| `COMPOSIO_WEBHOOK_SECRET` | Secret to validate incoming tool webhooks | (none) |
| `ROUTING_CACHE_TTL_HOURS` | TTL for routing decisions in Redis | `24` |

**Sources:** [orchestrator/.env.example:38-39](), [infrastructure/docker-compose.core.yml:64-65]()

### Long-Term Memory (Mem0)
| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `MEM0_API_URL` | Endpoint for the Mem0 OpenMemory server | `http://mem0-server:8765` |
| `MEM0_PG_PASSWORD` | Password for dedicated memory database | (required) |

**Sources:** [infrastructure/docker-compose.core.yml:68](), [infrastructure/docker-compose.memory.yml:27,55]()

### Voice Services
| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `VOICE_ENABLED` | Toggle for voice interaction capabilities | `false` |
| `VOICE_SERVICE_URL` | URL for the TTS/STT voice service | `http://voice-service:8300` |
| `TTS_ENGINE` | Engine for Text-to-Speech | `chatterbox` |

**Sources:** [infrastructure/docker-compose.core.yml:69-70](), [infrastructure/docker-compose.voice.yml:32,69]()

---

## System and Logging
| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Deployment stage (`development`, `production`) | `production` |
| `LOG_LEVEL` | Verbosity of backend logs (`DEBUG`, `INFO`, `ERROR`) | `INFO` |
| `REQUIRE_AUTH` | Toggle for Clerk JWT enforcement | `false` (local) |
| `SERVICE_NAME` | Identifier for the service in logs/metrics | `automatos-ai-api` |

**Sources:** [orchestrator/.env.example:29,33,58](), [infrastructure/docker-compose.core.yml:75,109-110]()

---

## Frontend Build-Time Variables

Next.js requires certain variables at build time to bake them into the static assets.

| Variable | Purpose |
| :--- | :--- |
| `NEXT_PUBLIC_API_URL` | The public URL of the FastAPI backend |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk public key for client-side auth |
| `NEXT_PUBLIC_VOICE_PIPELINE_URL` | WebSocket URL for voice interactions |

**Implementation:** These are passed as build arguments (`args`) in the `automotas-ai-frontend` service definition and injected into the environment.

**Sources:** [infrastructure/docker-compose.core.yml:136-142,150-153](), [infrastructure/railway-manifest.json:116-119]()

---