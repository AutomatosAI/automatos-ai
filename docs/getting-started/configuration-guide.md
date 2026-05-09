# Configuration Guide

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/settings/GeneralSettingsTab.tsx](frontend/components/settings/GeneralSettingsTab.tsx)
- [frontend/components/settings/OnboardingAgentsTab.tsx](frontend/components/settings/OnboardingAgentsTab.tsx)
- [frontend/components/settings/SettingsPanel.tsx](frontend/components/settings/SettingsPanel.tsx)
- [frontend/components/settings/SystemSettingsTab.tsx](frontend/components/settings/SystemSettingsTab.tsx)
- [frontend/tsconfig.tsbuildinfo](frontend/tsconfig.tsbuildinfo)
- [orchestrator/api/chatbot_llm.py](orchestrator/api/chatbot_llm.py)
- [orchestrator/api/onboarding_agents.py](orchestrator/api/onboarding_agents.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/llm/manager.py](orchestrator/core/llm/manager.py)
- [orchestrator/core/models/system_settings.py](orchestrator/core/models/system_settings.py)
- [orchestrator/core/seeds/seed_system_settings.py](orchestrator/core/seeds/seed_system_settings.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/modules/memory/context_router.py](orchestrator/modules/memory/context_router.py)
- [orchestrator/modules/memory/unified_memory_service.py](orchestrator/modules/memory/unified_memory_service.py)
- [orchestrator/scripts/create_test_workspace.py](orchestrator/scripts/create_test_workspace.py)
- [orchestrator/tests/test_unified_memory.py](orchestrator/tests/test_unified_memory.py)
- [scripts/ralph/IMPLEMENTATION_PLAN.md](scripts/ralph/IMPLEMENTATION_PLAN.md)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)
- [scripts/ralph/progress.txt](scripts/ralph/progress.txt)

</details>



This document covers the technical configuration of Automatos AI, including environment variables, service-specific settings (LLM, Redis, Postgres), memory layer parameters, and the database-backed system settings architecture that replaces traditional `.env` files for runtime configuration.

---

## Configuration Architecture

Automatos AI has migrated from a static environment-based configuration to a dynamic, database-backed system. While core infrastructure (DB/Redis) still uses environment variables for bootstrapping, LLM tiers and feature flags are managed via the `SystemSetting` model.

### Configuration Loading Flow

```mermaid
graph TB
    subgraph "BootstrapSpace"
        EnvFile[".env File"]
        ConfigClass["'Config' (config.py)"]
    end

    subgraph "CodeEntitySpace"
        AppStartup["FastAPI 'main.py'"]
        LLMManager["'LLMManager' (manager.py)"]
        SysSeed["'seed_system_settings' (seed_system_settings.py)"]
        DB_Settings["'SystemSetting' Table (models/system_settings.py)"]
    end

    subgraph "FrontendSpace"
        SettingsUI["'SystemSettingsTab' (SystemSettingsTab.tsx)"]
    end

    EnvFile --> ConfigClass
    ConfigClass --> AppStartup
    AppStartup -->|"Check/Seed"| SysSeed
    SysSeed --> DB_Settings
    
    LLMManager -->|"get_system_setting"| DB_Settings
    SettingsUI -->|"bulkUpdateSettings"| DB_Settings
```

**Sources:** [orchestrator/config.py:28-150](), [orchestrator/core/llm/manager.py:56-96](), [orchestrator/core/seeds/seed_system_settings.py:161-180](), [frontend/components/settings/SystemSettingsTab.tsx:72-107]()

---

## Core Infrastructure

### PostgreSQL with pgvector
The system requires PostgreSQL with the `pgvector` extension. Production environments enforce SSL for non-local hosts via `Config.get_database_url()`.

| Variable | Required | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | **Yes** | Primary connection string. Overrides individual params. [orchestrator/config.py:42]() |
| `POSTGRES_DB` | **Yes** | Database name. [orchestrator/config.py:37]() |
| `SQL_DEBUG` | No | Enables SQLAlchemy echo mode. [orchestrator/config.py:43]() |

### Redis Configuration
Redis is the backbone for L1 memory, task queues, and result caching. The `Config` class constructs the `REDIS_URL` from parts if not explicitly provided.

| Variable | Required | Description |
| :--- | :--- | :--- |
| `REDIS_HOST` | **Yes** | Redis server host. [orchestrator/config.py:63]() |
| `REDIS_URL` | No | Full connection string (e.g., `redis://user:pass@host:port/0`). [orchestrator/config.py:69-79]() |

**Sources:** [orchestrator/config.py:34-79]()

---

## LLM Tier Configuration (PRD-136)

The system has collapsed fragmented LLM configurations into three canonical tiers. These are managed via the `SystemSettingsTab` in the UI and stored in the `system_settings` table.

| Tier | Category | Purpose |
| :--- | :--- | :--- |
| **Auto** | `orchestrator_llm` | The "Brain". Premium reasoning, planning, and user chat. [orchestrator/core/llm/manager.py:35-36]() |
| **System** | `system_llm` | Internal high-volume calls (RAG, summarization, tool routing). [orchestrator/core/llm/manager.py:39-49]() |
| **Embeddings** | `embeddings` | Vectorization and semantic search. [orchestrator/core/llm/manager.py:52]() |

### LLM Resolution Logic
The `LLMManager` resolves configurations using a tiered strategy:
1. **Tier Settings**: Fetches `provider` and `model` from the category defined in `SERVICE_CATEGORY_MAP`. [orchestrator/core/llm/manager.py:33-53]()
2. **Credential Resolution**: Maps the provider to a secret in the `CredentialStore` using the pattern `credential_name_{provider}`. [orchestrator/core/llm/manager.py:163-180]()
3. **Fallback**: Defaults to environment variables (e.g., `OPENAI_API_KEY`) if no database credential is found. [orchestrator/core/llm/manager.py:145-146]()

**Sources:** [orchestrator/core/llm/manager.py:1-185](), [orchestrator/core/seeds/seed_system_settings.py:29-158]()

---

## Memory System Parameters

The `UnifiedMemoryService` manages a 5-layer stack. Configuration for these layers is defined in `Config` and can be overridden via system settings.

### Memory Layer Configuration (L1-L3)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MEMORY_SESSION_TTL_SECONDS` | `86400` | L1 Working Memory (Redis) retention (24h). [orchestrator/config.py:85]() |
| `MEMORY_DECAY_RATE` | `0.1` | Ebbinghaus decay rate for L2 (Short-term). [orchestrator/config.py:99]() |
| `MEMORY_PROMOTION_MIN_IMPORTANCE` | `0.7` | Threshold for L2 $\rightarrow$ L3 (Long-term) promotion. [orchestrator/config.py:105]() |
| `MEMORY_CACHE_TTL_SECONDS` | `300` | TTL for Mem0 search result caching in Redis. [orchestrator/config.py:89]() |

### Memory Namespacing
All memory operations use the `MemoryNamespace` class to ensure consistent key formatting across Redis and Mem0.

```python
# Standardized namespacing implementation
namespace = MemoryNamespace(workspace_id="ws_123")
redis_key = namespace.session("conv_456") # mem:session:ws_123:conv_456
mem0_user_id = namespace.agent(agent_id=7) # mem:ws_123:agent:7
```

**Sources:** [orchestrator/config.py:82-124](), [orchestrator/modules/memory/unified_memory_service.py:38-117]()

---

## Credential Management System

The `CredentialStore` provides secure storage for API keys and sensitive tokens, using the `EncryptionService` for AES-256 encryption.

### Credential Resolution Strategy
When a service (like the LLM Manager) requests a credential, the `resolver.py` follows this priority:
1. **Explicit Mapping**: Check `system_settings` for a key like `orchestrator_llm.credential_name_openai`. [orchestrator/core/llm/manager.py:163-170]()
2. **Standard Pattern**: Search for `{environment}_{provider}_api`. [orchestrator/core/llm/manager.py:141]()
3. **Environment Fallback**: Last resort check for standard env vars. [orchestrator/core/llm/manager.py:145]()

### Monitoring and Audit
Every credential access is logged to the `CredentialAuditLog` table, tracking the `workspace_id` and the identity of the requester to ensure multi-tenant isolation.

**Sources:** [orchestrator/core/llm/manager.py:135-185](), [frontend/components/settings/SettingsPanel.tsx:71-74](), [orchestrator/main.py:61]()

---

## Onboarding & Coordination Settings

Specialized settings for the **Mission Pipeline** (PRD-130) and **Onboarding Agents** are managed through the `OnboardingAgentsTab`.

- **Planner Configuration**: Controls `planner_max_tokens` and `planner_temperature` for the Mission decomposition stage. [frontend/components/settings/OnboardingAgentsTab.tsx:45]()
- **Verifier Configuration**: Sets the `verification_pass_threshold` (e.g., 0.8) and `catastrophic_threshold` for the LLM-as-judge pipeline. [frontend/components/settings/OnboardingAgentsTab.tsx:47]()
- **Agent Personas**: Allows runtime editing of `custom_persona_prompt` for onboarding agents without code changes. [frontend/components/settings/OnboardingAgentsTab.tsx:32]()

**Sources:** [frontend/components/settings/OnboardingAgentsTab.tsx:20-176](), [orchestrator/api/onboarding_agents.py]()

---