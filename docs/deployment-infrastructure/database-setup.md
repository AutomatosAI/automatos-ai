# Database Setup

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.github/workflows/test.yml](.github/workflows/test.yml)
- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [infrastructure/.env.example](infrastructure/.env.example)
- [infrastructure/railway-manifest.json](infrastructure/railway-manifest.json)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/conftest.py](orchestrator/conftest.py)
- [orchestrator/core/database/migrations/044_pinned_documents.sql](orchestrator/core/database/migrations/044_pinned_documents.sql)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/modules/codegraph/tests/conftest.py](orchestrator/modules/codegraph/tests/conftest.py)
- [orchestrator/modules/learning/tests/conftest.py](orchestrator/modules/learning/tests/conftest.py)
- [orchestrator/modules/rag/pinned_context.py](orchestrator/modules/rag/pinned_context.py)
- [orchestrator/modules/rag/retrieval_filters.py](orchestrator/modules/rag/retrieval_filters.py)
- [orchestrator/modules/search/tests/conftest.py](orchestrator/modules/search/tests/conftest.py)
- [orchestrator/modules/search/tests/test_math_foundations.py](orchestrator/modules/search/tests/test_math_foundations.py)
- [orchestrator/modules/tools/discovery/actions_documents.py](orchestrator/modules/tools/discovery/actions_documents.py)
- [orchestrator/modules/tools/discovery/handlers_documents.py](orchestrator/modules/tools/discovery/handlers_documents.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)
- [orchestrator/scripts/init_test_db.py](orchestrator/scripts/init_test_db.py)
- [orchestrator/tests/test_dockerfile_prod_parity.py](orchestrator/tests/test_dockerfile_prod_parity.py)
- [orchestrator/tests/test_document_pinning.py](orchestrator/tests/test_document_pinning.py)
- [orchestrator/tests/test_read_document_tool.py](orchestrator/tests/test_read_document_tool.py)
- [orchestrator/tests/test_retrieval_filters.py](orchestrator/tests/test_retrieval_filters.py)

</details>



## Purpose and Scope

This document covers the PostgreSQL database configuration, initialization, and management for Automatos AI. It details the connection management via SQLAlchemy, the `pgvector` extension setup, schema initialization, and the versioned system prompt registry.

For application-level data models and ORM patterns, see [Backend Architecture](18.3). For complete environment variable reference, see [Environment Variables](20.3).

---

## PostgreSQL with pgvector

Automatos AI uses **PostgreSQL** with the **pgvector extension** for vector similarity search. The system relies on native vector operations for semantic routing and memory retrieval across its 5-layer memory architecture.

The `pgvector` dependency is specified in `orchestrator/requirements.txt` [orchestrator/requirements.txt:17](). The Docker Compose setup uses the `pgvector/pgvector:pg16` image for the `postgres` service, ensuring the extension is available [docker-compose.yml:31]().

### Connection Management

The database layer uses SQLAlchemy for connection pooling and session management.

*   **Engine Configuration**: The engine is created with a `pool_size` of 10 and `max_overflow` of 20, recycling connections every hour to ensure stability [orchestrator/core/database/database.py:83-91]().
*   **SSL Enforcement**: In production environments, the system automatically appends `sslmode=require` to the database URL [orchestrator/core/database/database.py:74-80]().
*   **Credential Resolution**: The system attempts to fetch connection parameters (host, port, user, password) from a secure `credential_resolver` before falling back to standard environment variables like `DATABASE_URL` [orchestrator/core/database/database.py:23-67]().

### Session Lifecycle and Safety

To prevent "idle in transaction" states that can block DDL and hold row locks during long-lived LLM calls, the system implements strict session handling:

*   **`get_db()`**: A FastAPI dependency that ensures a `rollback()` is performed before `close()` in the `finally` block, ensuring no transaction lingers [orchestrator/core/database/database.py:105-116]().
*   **`end_open_transaction()`**: A utility to explicitly commit a transaction before an `await` block (e.g., an LLM call), sitting idle instead of idle-in-transaction [orchestrator/core/database/database.py:132-146]().

**Sources:** [orchestrator/core/database/database.py:23-146](), [orchestrator/tests/test_w1s8_get_db_lifecycle.py:1-13](), [orchestrator/requirements.txt:17](), [docker-compose.yml:31]()

---

## Schema Initialization and Migrations

### Database Bootstrapping Flow

The system initializes through a tiered process: structural creation, versioning via Alembic, and prompt seeding.

```mermaid
flowchart TD
    subgraph "Phase 1: Structure (SQLAlchemy)"
        [core.database.database] --> ["create_tables()"]
        ["create_tables()"] --> ["Base.metadata.create_all()"]
    end

    subgraph "Phase 2: Versioning (Alembic)"
        ["Base.metadata.create_all()"] --> [Migration_Scripts]
        [Migration_Scripts] --> [seed_auto_agents_existing_workspaces.py]
    end

    subgraph "Phase 3: Prompt Seeding"
        [seed_auto_agents_existing_workspaces.py] --> [seed_system_prompts.py]
        [seed_system_prompts.py] --> ["PROMPT_MANIFEST"]
        ["PROMPT_MANIFEST"] --> [Database_Ready]
    end
```
**Sources:** [orchestrator/core/database/database.py:96-104](), [orchestrator/core/seeds/seed_system_prompts.py:23-102]()

### Alembic Migrations

Alembic is used for database schema migrations. The `orchestrator` Dockerfile's `development` stage includes a `CMD` that runs `alembic upgrade heads` before starting the Uvicorn server [orchestrator/Dockerfile:124](). This ensures that the database schema is always up-to-date with the application code. The `heads` argument is used to apply all pending migrations, even if there are multiple unmerged branches in the migration history.

### `init_fresh_db` and `init_test_db.py`

For fresh installations or testing environments, the system provides mechanisms to initialize the database without relying on a full migration history.

*   **`init_test_db.py`**: This script is used in the CI environment to create all tables directly from SQLAlchemy models [orchestrator/scripts/init_test_db.py:53](). It also handles the creation of tables that do not have SQLAlchemy models, such as `document_chunks` and `codegraph_projects`, using raw SQL [orchestrator/scripts/init_test_db.py:63-151](). This script checks for `pgvector` availability and conditionally adds the `vector` type to columns [orchestrator/scripts/init_test_db.py:30-45]().
*   **`init_fresh_db`**: While not explicitly shown in the provided files, this typically refers to a similar process for local development setups, often involving `Base.metadata.create_all()` and seeding.

The `orchestrator-tests` job in `test.yml` explicitly calls `python scripts/init_test_db.py` to prepare the test database [test.yml:109]().

### SQL Migrations Directory

The `orchestrator/core/database/migrations` directory contains SQL migration scripts. For example, `044_pinned_documents.sql` is a raw SQL migration for pinned documents [orchestrator/core/database/migrations/044_pinned_documents.sql](). These are managed by Alembic.

**Sources:** [orchestrator/Dockerfile:124](), [orchestrator/scripts/init_test_db.py:53-151](), [test.yml:109](), [orchestrator/core/database/migrations/044_pinned_documents.sql]()

### System Prompt Management (PRD-58)

Automatos AI features a sophisticated, versioned system prompt registry that allows for A/B testing and evaluation of LLM instructions.

*   **`SystemPrompt`**: The top-level entity identified by a unique `slug` (e.g., `routing-classifier`) used by the code [orchestrator/core/models/system_prompts.py:32-41]().
*   **`SystemPromptVersion`**: Immutable snapshots of prompt content. Only one version per prompt can be `active` at a time [orchestrator/core/models/system_prompts.py:71-93]().
*   **`SystemPromptEvalRun`**: Tracks evaluation scores from FutureAGI for specific prompt versions, covering quality, safety, and optimization metrics [orchestrator/core/models/system_prompts.py:108-135]().

**Sources:** [orchestrator/core/models/system_prompts.py:32-135]()

---

## Natural Language to Code Entity Mapping

This section bridges conceptual data requirements with specific code implementations.

### Prompt Retrieval and Interpolation

When the system needs a specific behavior (e.g., "be friendly"), it resolves a slug to a formatted string.

```mermaid
graph LR
    subgraph "Natural Language Space"
        ["'I need the friendly persona'"]
        ["'Format with agent name Atlas'"]
    end

    subgraph "Code Entity Space"
        ["PromptRegistry.get(slug)"]
        ["SystemPrompt.slug = 'chatbot-friendly'"]
        ["CachedPrompt (TTL 60s)"]
        ["_HARDCODED_DEFAULTS"]
    end

    ["'I need the friendly persona'"] --> ["PromptRegistry.get(slug)"]
    ["PromptRegistry.get(slug)"] -.-> ["SystemPrompt.slug = 'chatbot-friendly'"]
    ["SystemPrompt.slug = 'chatbot-friendly'"] -.-> ["CachedPrompt (TTL 60s)"]
    ["CachedPrompt (TTL 60s)"] -- "Fallback" --> ["_HARDCODED_DEFAULTS"]
```
**Sources:** [orchestrator/core/services/prompt_registry.py:35-76, 93-115](), [orchestrator/core/seeds/seed_system_prompts.py:25-43]()

### Admin Configuration Mapping

System-wide settings are transitioned from `.env` files to database-backed `SystemSetting` models for real-time updates.

```mermaid
graph TD
    subgraph "Natural Language Space"
        ["'Update the API rate limit'"]
        ["'Change system-wide LLM'"]
    end

    subgraph "Code Entity Space"
        ["SystemSetting (key, value)"]
        ["SystemSettingUpdate (Pydantic)"]
        ["update_system_setting()"]
        ["_require_admin(ctx)"]
    end

    ["'Update the API rate limit'"] --> ["update_system_setting()"]
    ["'Change system-wide LLM'"] --> ["update_system_setting()"]
    ["update_system_setting()"] -.-> ["_require_admin(ctx)"]
    ["_require_admin(ctx)"] --> ["SystemSetting (key, value)"]
```
**Sources:** [orchestrator/api/system_settings.py:41-47, 162-186](), [orchestrator/core/models/system_settings.py:23-27]()

---

## Seed Data and Defaults

The database is populated with essential operational data via seeders:

### System Prompt Manifest
The `PROMPT_MANIFEST` defines the initial set of instructions for core platform functions:
*   **`routing-classifier`**: Instructions for the Universal Router to select agents [orchestrator/core/seeds/seed_system_prompts.py:85-102]().
*   **`chatbot-technical`**: Persona guidelines for developer-focused interactions [orchestrator/core/seeds/seed_system_prompts.py:64-81]().
*   **`task-decomposer`**: Logic for breaking complex goals into agent sub-tasks [orchestrator/core/seeds/seed_system_prompts.py:104-123]().

### Hardcoded Fallbacks
The `PromptRegistry` maintains a set of `_HARDCODED_DEFAULTS`. These ensure the system remains functional even if the database is temporarily unreachable during the bootstrap phase [orchestrator/core/services/prompt_registry.py:149-199]().

**Sources:** [orchestrator/core/seeds/seed_system_prompts.py:23-123](), [orchestrator/core/services/prompt_registry.py:145-203]()

---

## Key Database Functions and Services

| Function / Class | File Path | Purpose |
|----------|-----------|---------|
| `get_db` | `core/database/database.py` | FastAPI dependency for thread-safe session management with automatic rollback [orchestrator/core/database/database.py:105-116](). |
| `PromptRegistry` | `core/services/prompt_registry.py` | Singleton service managing prompt caching (60s TTL) and DB resolution [orchestrator/core/services/prompt_registry.py:35-53](). |
| `_assert_admin` | `api/admin_prompts.py` | Security gate ensuring only admin users can modify system-level prompts [orchestrator/api/admin_prompts.py:49-62](). |
| `AuditService` | `core/services/audit_service.py` | Logs security-relevant database and setting changes to the `audit` logger [orchestrator/core/services/audit_service.py:55-90](). |
| `list_system_settings` | `api/system_settings.py` | Endpoint for retrieving system-wide configuration with admin-only access [orchestrator/api/system_settings.py:50-67](). |

**Sources:** [orchestrator/core/database/database.py:105-116](), [orchestrator/core/services/prompt_registry.py:35-53](), [orchestrator/api/admin_prompts.py:49-62](), [orchestrator/core/services/audit_service.py:55-122](), [orchestrator/api/system_settings.py:50-67]()

---