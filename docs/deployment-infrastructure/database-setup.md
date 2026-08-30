# Database Setup

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/hooks/use-database-knowledge.ts](frontend/hooks/use-database-knowledge.ts)
- [orchestrator/api/admin_prompts.py](orchestrator/api/admin_prompts.py)
- [orchestrator/api/credentials.py](orchestrator/api/credentials.py)
- [orchestrator/api/database_knowledge.py](orchestrator/api/database_knowledge.py)
- [orchestrator/api/document_generation.py](orchestrator/api/document_generation.py)
- [orchestrator/api/generated_images.py](orchestrator/api/generated_images.py)
- [orchestrator/api/system_settings.py](orchestrator/api/system_settings.py)
- [orchestrator/core/database/database.py](orchestrator/core/database/database.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/models/system_prompts.py](orchestrator/core/models/system_prompts.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/seeds/seed_system_prompts.py](orchestrator/core/seeds/seed_system_prompts.py)
- [orchestrator/core/services/audit_service.py](orchestrator/core/services/audit_service.py)
- [orchestrator/core/services/prompt_registry.py](orchestrator/core/services/prompt_registry.py)
- [orchestrator/modules/documents/generation_service.py](orchestrator/modules/documents/generation_service.py)
- [orchestrator/modules/nl2sql/service.py](orchestrator/modules/nl2sql/service.py)

</details>



## Purpose and Scope

This document covers the PostgreSQL database configuration, initialization, and management for Automatos AI. It details the connection management using SQLAlchemy, schema initialization via Alembic migrations, and the extensive seeding process for system defaults, credential types, and LLM-facing prompts.

For application-level data models and ORM patterns, see [Backend Architecture](18.3). For complete environment variable reference, see [Environment Variables](20.3).

---

## PostgreSQL with pgvector

Automatos AI uses **PostgreSQL** as its primary relational store, augmented with the **pgvector** extension for vector similarity search. The system relies on native vector operations for semantic routing, RAG (Retrieval-Augmented Generation), and memory retrieval.

### Key Features

| Feature | Purpose | Implementation |
|---------|---------|----------------|
| **pgvector Extension** | Vector embeddings for RAG and Semantic Routing | Enabled via initialization scripts to support `vector` types in models. |
| **SQLAlchemy ORM** | Python database abstraction | Centralized `SessionLocal` and `get_db` dependency [orchestrator/core/database/database.py:94-111](). |
| **Connection Pooling** | Performance and resource management | Configured with `pool_size=10` and `max_overflow=20` [orchestrator/core/database/database.py:83-91](). |
| **Credential-Based URL** | Secure connection resolution | Tries to resolve DB params via the internal `CredentialResolver` before falling back to env vars [orchestrator/core/database/database.py:23-40](). |

**Sources:** [orchestrator/core/database/database.py:23-111]()

---

## Schema Initialization and Migrations

### Database Bootstrapping Flow

The system initializes in three phases: structural creation, version control via Alembic, and comprehensive seeding of platform defaults.

**Fresh databases (compose, CI).** An empty database has no `alembic_version` table, and the migration history cannot replay from empty on its own. The backend entrypoint (`docker-entrypoint.sh`) therefore runs `python -m scripts.init_fresh_db` first: it builds the schema from the SQLAlchemy models plus a tolerant replay of the migration forest and stamps Alembic at heads, after which `alembic upgrade heads` is a no-op. No SQL snapshot is committed — the former `init_complete_schema.sql` is retired. Existing databases (anything with an `alembic_version` row) skip straight to incremental migrations. The `schema-drift` CI lane (`scripts/ci/schema_drift_check.py`) fails when a migration alters a table no writer creates. See [Self-hosting](../getting-started/self-hosting.md#4-first-boot--what-happens-and-how-long-it-takes) for the full boot order.

```mermaid
flowchart TD
    subgraph "Phase 1: Engine & Tables"
        [get_database_url] --> [create_engine]
        [create_engine] --> [create_tables]
        [create_tables] --> [Base.metadata.create_all]
    end

    subgraph "Phase 2: Migrations (Alembic)"
        [Base.metadata.create_all] --> [Alembic_Upgrade]
        [Alembic_Upgrade] --> [Schema_Versioning]
    end

    subgraph "Phase 3: Seeding"
        [Schema_Versioning] --> [load_seed_data.py]
        [load_seed_data.py] --> [seed_system_settings]
        [load_seed_data.py] --> [seed_models]
        [load_seed_data.py] --> [seed_system_prompts]
    end
```
**Sources:** [orchestrator/core/database/database.py:69-130](), [orchestrator/core/database/load_seed_data.py:25-54](), [orchestrator/core/database/load_seed_data.py:121-154]()

---

## Natural Language to Code Entity Mapping

This section bridges conceptual data requirements with specific code implementations.

### Credential and Database Knowledge Mapping

When a user adds a "Database Source" via natural language, the system maps this to encrypted credentials and introspected schema metadata.

```mermaid
graph LR
    subgraph "Natural Language Space"
        ["'Connect my production Postgres DB'"]
        ["'What is the schema of our sales table?'"]
    end

    subgraph "Code Entity Space"
        ["DatabaseKnowledgeSource"]
        ["DatabaseKnowledgeService.add_database_source()"]
        ["CredentialStore.create_credential()"]
        ["DatabaseIntrospectionService"]
    end

    ["'Connect my production Postgres DB'"] --> ["DatabaseKnowledgeService.add_database_source()"]
    ["DatabaseKnowledgeService.add_database_source()"] -.-> ["CredentialStore.create_credential()"]
    ["'What is the schema of our sales table?'"] --> ["DatabaseIntrospectionService"]
    ["DatabaseIntrospectionService"] -.-> ["DatabaseKnowledgeSource"]
```
**Sources:** [orchestrator/api/database_knowledge.py:118-141](), [orchestrator/modules/nl2sql/service.py:112-140](), [orchestrator/core/credentials/service.py:130-131]()

### Prompt Registry Mapping

System prompts are resolved from a hierarchy of sources to ensure the platform can bootstrap even without a fully populated database.

```mermaid
graph TD
    subgraph "Natural Language Space"
        ["'The AI should be technical'"]
        ["'Use the standard routing logic'"]
    end

    subgraph "Code Entity Space"
        ["PromptRegistry.get('chatbot-technical')"]
        ["SystemPromptVersion (status='active')"]
        ["_HARDCODED_DEFAULTS"]
        ["seed_system_prompts.py"]
    end

    ["'The AI should be technical'"] --> ["PromptRegistry.get('chatbot-technical')"]
    ["PromptRegistry.get('chatbot-technical')"] -.-> ["SystemPromptVersion (status='active')"]
    ["SystemPromptVersion (status='active')"] -- "Fallback" --> ["_HARDCODED_DEFAULTS"]
    ["'Use the standard routing logic'"] --> ["seed_system_prompts.py"]
```
**Sources:** [orchestrator/core/services/prompt_registry.py:59-76](), [orchestrator/core/services/prompt_registry.py:93-115](), [orchestrator/core/seeds/seed_system_prompts.py:23-43]()

---

## Seeding and Defaults

The platform uses an idempotent seeding strategy to ensure essential data is present across all environments.

### 1. System Settings (PRD-25)
Replaces static `.env` files with database-backed settings. This allows admins to modify platform behavior (e.g., LLM timeouts, feature flags) via the `SystemSettingsAPI` without restarting services [orchestrator/api/system_settings.py:1-18]().

### 2. Credential Types
The `load_seed_data.py` script populates over 400 credential types from `credential_types_seed.json`. This enables dynamic form generation in the UI for connecting external services [orchestrator/core/database/load_seed_data.py:60-108]().

### 3. System Prompts (PRD-58)
The `seed_system_prompts.py` manifest defines the core "personalities" and "orchestrator" logic. These are stored in `SystemPrompt` and `SystemPromptVersion` tables, allowing for versioning, rollback, and A/B testing via the Admin Prompts API [orchestrator/api/admin_prompts.py:1-12](), [orchestrator/core/seeds/seed_system_prompts.py:23-102]().

---

## Key Database Functions and Services

| Function / Service | File Path | Purpose |
|----------|-----------|---------|
| `get_db` | `core/database/database.py` | FastAPI dependency for yielding database sessions [orchestrator/core/database/database.py:105-111](). |
| `DatabaseKnowledgeService` | `modules/nl2sql/service.py` | Manages schema introspection and SQL generation for external DB sources [orchestrator/modules/nl2sql/service.py:75-96](). |
| `CredentialStore` | `core/credentials/service.py` | Handles encrypted storage and retrieval of sensitive keys [orchestrator/api/credentials.py:52-54](). |
| `PromptRegistry` | `core/services/prompt_registry.py` | Singleton service for resolving system prompts with a 60-second TTL cache [orchestrator/core/services/prompt_registry.py:35-53](). |
| `DocumentTemplateService` | `modules/documents/template_service.py` | Manages Jinja2 templates for PDF/DOCX generation [orchestrator/api/document_generation.py:83-85](). |

**Sources:** [orchestrator/core/database/database.py:105-111](), [orchestrator/modules/nl2sql/service.py:75-96](), [orchestrator/core/services/prompt_registry.py:35-53](), [orchestrator/api/document_generation.py:83-85]()

---