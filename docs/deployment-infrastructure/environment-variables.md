# Environment Variables

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitignore](.gitignore)
- [.gitleaksignore](.gitleaksignore)
- [Makefile](Makefile)
- [docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md](docs/PRDS/126-BUSINESS-KNOWLEDGE-GRAPH.md)
- [docs/PRDS/PRD-234-SESSION-MODE-SUBSCRIPTION-RUNTIME.md](docs/PRDS/PRD-234-SESSION-MODE-SUBSCRIPTION-RUNTIME.md)
- [docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md)
- [frontend/components/settings/GeneralSettingsTab.tsx](frontend/components/settings/GeneralSettingsTab.tsx)
- [frontend/components/settings/SessionModeTab.tsx](frontend/components/settings/SessionModeTab.tsx)
- [graphify-out/snapshots/bucket-1-pre-drop.sql](graphify-out/snapshots/bucket-1-pre-drop.sql)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/alembic/versions/prd135_drop_bucket_1.py](orchestrator/alembic/versions/prd135_drop_bucket_1.py)
- [orchestrator/core/models/system_settings.py](orchestrator/core/models/system_settings.py)
- [orchestrator/core/seeds/seed_system_settings.py](orchestrator/core/seeds/seed_system_settings.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/modules/memory/write_contract.py](orchestrator/modules/memory/write_contract.py)
- [orchestrator/services/cli_host_service.py](orchestrator/services/cli_host_service.py)
- [orchestrator/tests/test_prd206_write_contract.py](orchestrator/tests/test_prd206_write_contract.py)
- [orchestrator/tests/test_prd234_s1a_cli_hosts_realdb.py](orchestrator/tests/test_prd234_s1a_cli_hosts_realdb.py)
- [orchestrator/tests/test_system_settings_null_flags.py](orchestrator/tests/test_system_settings_null_flags.py)
- [services/cli-host/automatos_cli_host/allowlist.py](services/cli-host/automatos_cli_host/allowlist.py)
- [services/cli-host/automatos_cli_host/hook_server.py](services/cli-host/automatos_cli_host/hook_server.py)

</details>



This document describes the environment variable configuration system used across all Automatos AI services. Environment variables control database connections, external service credentials, feature flags, and service-specific settings across the 19-service production topology.

For deployment infrastructure, see [Production Deployment](20.6). For credential management in the UI, see [Credentials Management](17.5).

---

## Overview

Automatos AI uses environment variables for all external configuration to support multiple deployment targets (Docker Compose, Railway, Kubernetes) without code changes. Variables are loaded from `.env` files in local development and from platform-provided environment in production.

The system follows a three-tier loading strategy:

1.  **Environment variables** (highest priority) — set by hosting platform or shell.
2.  **`.env` file** — loaded via `python-dotenv` in the backend application lifecycle [orchestrator/main.py:24-26]().
3.  **Hardcoded defaults** — fallback values in centralized config [orchestrator/config.py:28-132]().

The codebase includes an `.env.example` in the root for the full platform stack and a service-specific `orchestrator/.env.example` for the core API.

**Sources:** [orchestrator/config.py:24-27](), [orchestrator/main.py:24-26](), [orchestrator/.env.example:1-64](), [.gitignore:109-116]()

---

## Configuration Injection Pipeline

The following diagram illustrates how configuration flows from environment sources into the core system entities.

**Diagram: Configuration Injection Pipeline**
```mermaid
graph TB
    subgraph "Environment_Sources"
        EnvFile[".env file"]
        ComposeEnv["docker-compose.yml<br/>environment section"]
        PlatformEnv["Platform Environment<br/>(Railway/Cloud)"]
        SystemSettingsDB["SystemSetting<br/>(Database)"]
    end
    
    subgraph "Code_Entity_Config_Manager"
        ConfigModule["Config<br/>(orchestrator/config.py)"]
    end
    
    EnvFile --> ConfigModule
    ComposeEnv --> ConfigModule
    PlatformEnv --> ConfigModule
    SystemSettingsDB --> ConfigModule
    
    subgraph "System_Consumers"
        DB["init_database<br/>(core/database/database.py)"]
        RedisSvc["REDIS_URL<br/>(orchestrator/config.py)"]
        LLM["LLMManager<br/>(API Providers)"]
        Auth["get_request_context_hybrid<br/>(core/auth/hybrid.py)"]
        MemoryJobs["Memory_Jobs<br/>(orchestrator/config.py)"]
        CliHostService["CliHostService<br/>(services/cli_host_service.py)"]
    end
    
    ConfigModule --> DB
    ConfigModule --> RedisSvc
    ConfigModule --> LLM
    ConfigModule --> Auth
    ConfigModule --> MemoryJobs
    ConfigModule --> CliHostService
```

**Sources:** [orchestrator/config.py:28-132](), [orchestrator/main.py:24-29](), [orchestrator/.env.example:1-64](), [orchestrator/core/models/system_settings.py:59-73](), [orchestrator/services/cli_host_service.py:28]()

---

## Required Environment Variables

These variables **must** be set for the system to function. Missing required variables will cause startup failures in production.

### Core Infrastructure

| Variable | Purpose | Example | Used By |
| :--- | :--- | :--- | :--- |
| `POSTGRES_PASSWORD` | PostgreSQL admin password | `secure_db_pass_123` | `postgres`, `backend`, `workspace-worker` |
| `REDIS_PASSWORD` | Redis authentication | `secure_redis_pass` | `redis`, `backend`, `workspace-worker` |
| `API_KEY` | Internal API authentication | `automatos_api_key_xyz` | `backend` auth middleware |
| `DATABASE_URL` | Full connection string | `postgresql://user:pass@host:port/db` | `Config.get_database_url()` |

These variables are declared as `${VAR:?message}` in `docker-compose.yml` to ensure they are set before `docker compose up` can proceed [docs/getting-started/self-hosting.md:29-38]().

**Sources:** [orchestrator/config.py:37-42](), [orchestrator/config.py:63-65](), [orchestrator/.env.example:1-16](), [docs/getting-started/self-hosting.md:29-38]()

---

## Database and Cache Configuration

### PostgreSQL with pgvector
The system uses `pgvector` for semantic search. The `Config` class enforces SSL for production hosts [orchestrator/config.py:47-58]().

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `POSTGRES_HOST` | `localhost` | PostgreSQL server hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `orchestrator_db` | Database name |
| `SQL_DEBUG` | `false` | Toggles SQLAlchemy echo [orchestrator/config.py:43]() |

**Sources:** [orchestrator/config.py:37-58](), [orchestrator/.env.example:1-6]()

### Redis Configuration
Redis serves as the L1 memory tier, Pub/Sub broker, and task queue. The `REDIS_URL` property dynamically constructs the connection string if only individual parts are provided [orchestrator/config.py:68-79]().

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database index |

**Sources:** [orchestrator/config.py:63-79](), [orchestrator/.env.example:8-11]()

---

## LLM Provider Configuration

Automatos AI supports a multi-provider strategy. While variables can be set in the environment, the system also supports a dynamic **Credential Store** for per-workspace keys.

**Diagram: LLM Configuration Resolution**
```mermaid
graph TD
    subgraph "Request_Flow"
        Req["Agent Execution"]
    end

    subgraph "Resolution_Hierarchy"
        WS_Config["workspace.settings.byok_preferences<br/>(SystemSetting.category='orchestrator_llm')"]
        Env_Keys["os.getenv('OPENAI_API_KEY')"]
        Auto_Agent["Auto Agent model_config"]
    end

    Req --> WS_Config
    WS_Config -- "Missing or not configured" --> Env_Keys
    Env_Keys -- "Missing" --> Auto_Agent
```

The `SystemSetting` model and `seed_system_settings` function manage LLM configurations in the database, categorizing them into `orchestrator_llm` (Auto - premium, user-facing), `system_llm` (System - cheap-fast internal), and `embeddings` (vectorization) [orchestrator/core/models/system_settings.py:31-33](), [orchestrator/core/seeds/seed_system_settings.py:166-170](). These settings include `provider`, `model`, `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `timeout_seconds`, and `max_retries` [orchestrator/core/seeds/seed_system_settings.py:41-157]().

| Variable | Purpose |
| :--- | :--- |
| `OPENAI_API_KEY` | Key for OpenAI models |
| `ANTHROPIC_API_KEY` | Key for Anthropic models |
| `LLM_PROVIDER` | Default provider (e.g., `openai`) |
| `LLM_MODEL` | Default model (e.g., `gpt-4`) |

**Sources:** [orchestrator/config.py:119-125](), [orchestrator/.env.example:18-26](), [orchestrator/core/seeds/seed_auto_agent.py:68-78](), [orchestrator/core/models/system_settings.py:31-33](), [orchestrator/core/seeds/seed_system_settings.py:41-157](), [orchestrator/core/seeds/seed_system_settings.py:166-170]()

---

## Memory Tier Configuration (PRD-79)

The Unified Memory Service uses several environment variables to control retention, decay, and promotion across the 5-layer architecture.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `MEMORY_DECAY_RATE` | `0.004` | Ebbinghaus decay rate per hour [orchestrator/config.py:106]() |
| `MEMORY_PROMOTION_MIN_IMPORTANCE` | `0.7` | Threshold for L2→L3 promotion [orchestrator/config.py:117]() |
| `MEMORY_SESSION_TTL_SECONDS` | `86400` | TTL for active L1 sessions [orchestrator/config.py:85]() |
| `MEMORY_JOBS_ENABLED` | `true` | Toggles background consolidation jobs [orchestrator/config.py:131]() |

**Sources:** [orchestrator/config.py:82-132]()

---

## Universal Router and Tools

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `COMPOSIO_WEBHOOK_SECRET` | Secret to validate tool webhooks | (none) |
| `ROUTING_CACHE_TTL_HOURS` | TTL for routing decisions in Redis | `24` |
| `ROUTING_LLM_CONFIDENCE_THRESHOLD` | Threshold for Tier 3 routing | `0.5` |

**Sources:** [orchestrator/.env.example:37-41]()

---

## System and Logging

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Deployment stage (`development`, `production`) | `production` |
| `LOG_LEVEL` | Verbosity of backend logs | `INFO` |
| `DEBUG` | Toggle for FastAPI debug mode | `false` |
| `REQUIRE_AUTH` | Toggle for Clerk JWT enforcement | `false` (local) |

The `GeneralSettingsTab` in the frontend allows users to configure `environment` and `log_level` via the UI, which are then stored as `SystemSetting` entries [frontend/components/settings/GeneralSettingsTab.tsx:83-119]().

**Sources:** [orchestrator/.env.example:28-34,56-58](), [frontend/components/settings/GeneralSettingsTab.tsx:83-119]()

---

## Gitleaks Ignore

The `.gitleaksignore` file specifies patterns and specific commit hashes to ignore during Gitleaks scans. This is crucial for preventing false positives, especially for test fixtures that intentionally contain secret-like strings to validate exclusion mechanisms. For example, a test for the memory exclusion validator includes a fake "api_key = ..." string that is meant to be refused by the write path [orchestrator/tests/test_prd206_write_contract.py:7-8](). Another entry addresses a system setting name that matched a generic API key pattern due to entropy [orchestrator/tests/test_system_settings_null_flags.py:1-3]().

**Sources:** [.gitleaksignore:1-20](), [orchestrator/tests/test_prd206_write_contract.py:7-8](), [orchestrator/tests/test_system_settings_null_flags.py:1-3]()

---

## Service Configuration

### CLI Host Service (Session Mode)

The `cli_host_service` manages local CLI sessions for agents, particularly for "Session Mode" (PRD-234). This service uses environment variables for configuration, though many settings are managed through the `CliHost` database model and its associated pairing and token mechanisms [orchestrator/services/cli_host_service.py:10-15]().

**Diagram: CLI Host Service Configuration**
```mermaid
graph TD
    subgraph "CLI_Host_Configuration"
        CliHostModel["CliHost Model<br/>(core/models/cli_hosts.py)"]
        CliHostService["cli_host_service<br/>(orchestrator/services/cli_host_service.py)"]
        ConfigModule["Config<br/>(orchestrator/config.py)"]
    end

    subgraph "CLI_Host_Runtime"
        CliHostApp["automatos_cli_host<br/>(services/cli-host)"]
    end

    ConfigModule --> CliHostService
    CliHostModel --> CliHostService
    CliHostService -- "Provides runtime config" --> CliHostApp

    CliHostApp -- "Reports capabilities" --> CliHostModel
    CliHostApp -- "Claims tasks" --> CliHostService
```

The `CliHost` model stores `capabilities` (e.g., available CLIs, models) and `status` (PENDING, PAIRED, REVOKED) [orchestrator/services/cli_host_service.py:80-85](), [orchestrator/services/cli_host_service.py:126-127](). The `automatos_cli_host` application, which runs on the user's machine, uses an `allowlist.py` to define which commands are permitted [services/cli-host/automatos_cli_host/allowlist.py]().

**Sources:** [orchestrator/services/cli_host_service.py:10-15](), [orchestrator/services/cli_host_service.py:80-85](), [orchestrator/services/cli_host_service.py:126-127](), [services/cli-host/automatos_cli_host/allowlist.py]()

### Plugin Marketplace / S3

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `MARKETPLACE_S3_BUCKET` | S3 bucket for marketplace assets | `automatos-marketplace` |
| `AWS_ACCESS_KEY_ID` | AWS Access Key ID | (none) |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Access Key | (none) |
| `AWS_REGION` | AWS region for S3 | `us-east-1` |
| `PLUGIN_MAX_UPLOAD_SIZE_MB` | Max size for plugin uploads | `10` |
| `PLUGIN_LLM_SCAN_MODEL` | LLM used for scanning plugins | `claude-haiku-4-20250414` |
| `PLUGIN_CACHE_TTL_SECONDS` | TTL for plugin metadata cache | `3600` |

**Sources:** [orchestrator/.env.example:48-55]()

---

## Gitignore and Sensitive Files

The `.gitignore` file is configured to exclude various sensitive or generated files from version control. This includes:

*   `.mcp.json`: MCP config, potentially containing API keys.
*   `envs/*.local`: Personal local environment overrides that may hold secrets.
*   `.env`, `.env.local`, `.env.*.local`: Environment files containing secrets.
*   `.credential_key`: Auto-generated credential encryption keys.
*   `tests/e2e/.auth/`: E2E dev-browser auth fixtures, containing live Clerk session material.
*   `orchestrator/scripts/eval/**/live/`: Real-tenant evaluation gold sets, containing sensitive client corpus data.
*   `/workspaces/`: The workspace-worker host directory, where agent deliverables are stored locally.

**Sources:** [.gitignore:1-150]()

---