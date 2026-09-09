# Configuration Guide

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitleaksignore](.gitleaksignore)
- [Makefile](Makefile)
- [docs/PRDS/PRD-234-SESSION-MODE-SUBSCRIPTION-RUNTIME.md](docs/PRDS/PRD-234-SESSION-MODE-SUBSCRIPTION-RUNTIME.md)
- [docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md)
- [frontend/components/__tests__/prd197-substrate-tile.test.tsx](frontend/components/__tests__/prd197-substrate-tile.test.tsx)
- [frontend/components/command-center/is-it-working-strip.tsx](frontend/components/command-center/is-it-working-strip.tsx)
- [frontend/components/settings/GeneralSettingsTab.tsx](frontend/components/settings/GeneralSettingsTab.tsx)
- [frontend/components/settings/SessionModeTab.tsx](frontend/components/settings/SessionModeTab.tsx)
- [frontend/hooks/use-analytics-api.ts](frontend/hooks/use-analytics-api.ts)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/api/workflows.py](orchestrator/api/workflows.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/models/substrate_metrics.py](orchestrator/core/models/substrate_metrics.py)
- [orchestrator/core/models/system_settings.py](orchestrator/core/models/system_settings.py)
- [orchestrator/core/observability/substrate_metrics.py](orchestrator/core/observability/substrate_metrics.py)
- [orchestrator/core/seeds/seed_system_settings.py](orchestrator/core/seeds/seed_system_settings.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/modules/memory/write_contract.py](orchestrator/modules/memory/write_contract.py)
- [orchestrator/reports/route-manifest.json](orchestrator/reports/route-manifest.json)
- [orchestrator/router_manifest.py](orchestrator/router_manifest.py)
- [orchestrator/services/cli_host_service.py](orchestrator/services/cli_host_service.py)
- [orchestrator/tests/authz_sweep_probe.py](orchestrator/tests/authz_sweep_probe.py)
- [orchestrator/tests/test_p2w2_authz_boundary_sweep.py](orchestrator/tests/test_p2w2_authz_boundary_sweep.py)
- [orchestrator/tests/test_prd154_s5_missions.py](orchestrator/tests/test_prd154_s5_missions.py)
- [orchestrator/tests/test_prd206_write_contract.py](orchestrator/tests/test_prd206_write_contract.py)
- [orchestrator/tests/test_prd222_w2s1_plan_tiers.py](orchestrator/tests/test_prd222_w2s1_plan_tiers.py)
- [orchestrator/tests/test_prd234_s1a_cli_hosts_realdb.py](orchestrator/tests/test_prd234_s1a_cli_hosts_realdb.py)
- [orchestrator/tests/test_system_settings_null_flags.py](orchestrator/tests/test_system_settings_null_flags.py)
- [services/cli-host/automatos_cli_host/allowlist.py](services/cli-host/automatos_cli_host/allowlist.py)
- [services/cli-host/automatos_cli_host/hook_server.py](services/cli-host/automatos_cli_host/hook_server.py)

</details>



## Purpose & Scope

This page covers the technical configuration subsystem of Automatos AI. It details how environment variables, database-backed system settings (`SystemSetting`), LLM tiers (`SettingCategory`), core infrastructure services (PostgreSQL, Redis, S3), and service features (such as Auto Live voice and memory lifecycles) are managed via `orchestrator/config.py` and system settings seeds.

---

## Configuration Architecture

Automatos AI employs a hybrid configuration model. While core infrastructure connection parameters (database URLs, Redis hosts) are ingested via environment variables through a centralized `Config` class, application-level parameters—including LLM provider choices, model hyper-parameters, memory thresholds, and feature flags—are stored in the database-backed **System Settings** framework (`orchestrator/core/models/system_settings.py`).

### Configuration Loading Flow

```mermaid
graph TB
    subgraph "InputSources"
        EnvFile[".env File"]
        ShellVars["Shell Env Vars"]
        DBCfg["'system_settings' Table"]
    end

    subgraph "CodeEntitySpace"
        ConfigClass[""Config" (config.py)"]
        SysSettings[""SystemSetting" (models/system_settings.py)"]
        LLMManager[""LLMManager""]
        AutoAgent[""Auto" Agent Row"]
    end

    EnvFile --> ConfigClass
    ShellVars --> ConfigClass
    
    ConfigClass -->|"Init DB"| SysSettings
    DBCfg --> SysSettings
    
    SysSettings --> LLMManager
    LLMManager -->|"Runtime Config"| Orchestrator["Orchestrator Logic"]
    
    DBCfg -->|"Workspace Overrides"| AutoAgent
    AutoAgent -->|"Brain Config"| ChatOrchestrator[""SmartChatOrchestrator""]
```

**Sources:** [orchestrator/config.py:28-32](), [orchestrator/core/models/system_settings.py:59-83](), [orchestrator/core/seeds/seed_system_settings.py:161-171]()

---

## Core Infrastructure Services

### PostgreSQL with pgvector
The persistence layer requires PostgreSQL with the `pgvector` extension enabled. In non-local deployments, SSL mode (`sslmode=require`) is programmatically enforced in `Config.get_database_url()` [orchestrator/config.py:47-60]().

| Parameter / Variable | Required | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | **Yes** | Full connection string. When provided, overrides individual host/port parameters [orchestrator/config.py:44]() |
| `POSTGRES_DB` | **Yes** | Target database name [orchestrator/config.py:39]() |
| `POSTGRES_USER` | **Yes** | Database username [orchestrator/config.py:40]() |
| `POSTGRES_PASSWORD` | **Yes** | Database password [orchestrator/config.py:41]() |
| `POSTGRES_HOST` | **Yes** | Database host address [orchestrator/config.py:42]() |
| `POSTGRES_PORT` | **Yes** | Database port [orchestrator/config.py:43]() |
| `SQL_DEBUG` | No | Toggles SQLAlchemy echo logging (`false` by default) [orchestrator/config.py:45]() |

### Redis Configuration
Redis functions as the L1 memory cache backbone, asynchronous Pub/Sub broker, and backend job store for `APScheduler` [orchestrator/config.py:65-81]().

| Parameter / Variable | Required | Description |
| :--- | :--- | :--- |
| `REDIS_URL` | No | Explicit URL string. If absent, constructed from `REDIS_HOST` and `REDIS_PORT` [orchestrator/config.py:73-81]() |
| `REDIS_HOST` | **Yes** | Redis server hostname [orchestrator/config.py:65]() |
| `REDIS_PORT` | **Yes** | Redis server port [orchestrator/config.py:66]() |
| `REDIS_PASSWORD` | No | Auth password if cluster authentication is enabled [orchestrator/config.py:67]() |
| `REDIS_DB` | No | Target database index (defaults to `0`) [orchestrator/config.py:68]() |

**Sources:** [orchestrator/config.py:30-81]()

---

## LLM Tier Management (PRD-136)

Per PRD-136, LLM configuration across the platform is collapsed into three clean functional tiers defined in `SettingCategory` (`orchestrator/core/models/system_settings.py`) and populated via `seed_system_settings` (`orchestrator/core/seeds/seed_system_settings.py`):

1. **Orchestrator LLM (`orchestrator_llm` / Auto)**: The high-reasoning "Brain" model (e.g., GPT-4o, Claude 3.5 Sonnet) used for user-facing chat responses and complex workflow orchestration planning [orchestrator/core/models/system_settings.py:31]().
2. **System LLM (`system_llm`)**: A fast, economical model (e.g., GPT-4o-mini, Gemini Flash) dedicated to internal background operations: complexity assessment, memory extraction, router classification, and agent deliberation loops [orchestrator/core/models/system_settings.py:32]().
3. **Embeddings (`embeddings`)**: The designated model configuration for vectorization, knowledge ingestion, and semantic retrieval [orchestrator/core/models/system_settings.py:33]().

**Sources:** [orchestrator/core/models/system_settings.py:19-34](), [orchestrator/core/seeds/seed_system_settings.py:35-158]()

---

## Memory System Configuration

The multi-layer memory architecture is governed by configurable parameters in `Config` (`orchestrator/config.py`) that balance retrieval precision against token overhead.

### Key Memory Lifecycle Constants

| Constant | Default | Description |
| :--- | :--- | :--- |
| `MEMORY_SESSION_TTL_SECONDS` | `86400` | L1 active session time-to-live (24 hours) [orchestrator/config.py:87]() |
| `MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS` | `3600` | Grace window following `end_session()` for final consolidation (1 hour) [orchestrator/config.py:89]() |
| `MEMORY_CACHE_TTL_SECONDS` | `300` | Redis caching window for durable L3 search results [orchestrator/config.py:91]() |
| `MEMORY_DECAY_RATE` | `0.004` | Hourly Ebbinghaus decay multiplier for L2 short-term memories [orchestrator/config.py:108]() |
| `MEMORY_DECAY_ARCHIVE_THRESHOLD` | `0.3` | Score threshold below which decaying items are archived [orchestrator/config.py:110]() |
| `MEMORY_PROMOTION_MIN_IMPORTANCE` | `0.7` | Importance gate required for automatic promotion from L2 to L3 [orchestrator/config.py:119]() |

**Sources:** [orchestrator/config.py:84-134]()

---

## Auto Live & Voice Configuration (PRD-207)

Auto Live powers real-time bidirectional voice communication via Retell AI. It is governed by a two-stage authorization mechanism to protect platform resources.

```mermaid
graph LR
    subgraph "GlobalSettings"
        AdminSwitch[""live_enabled" (SystemSetting)"]
        RetellCreds[""retell_api_key""]
    end

    subgraph "WorkspaceSettings"
        WSEnabled[""voice_live.enabled" (JSONB)"]
        WSCap[""monthly_cap_minutes""]
    end

    AdminSwitch --> Gate{{"Voice Gate"}}
    RetellCreds --> Gate
    WSEnabled --> Gate
    WSCap --> Gate
    
    Gate -->|"Pass"| VoiceCall["Mint Retell Call"]
    Gate -->|"Fail"| Blocked["Silent/Error"]
```

### Configuration Gates
- **Platform Arming**: Super-administrators configure global Retell API integration and master enable flags under the `voice` settings category [orchestrator/core/models/system_settings.py:52-53]().
- **Workspace Cap**: Individual workspaces opt in and configure `monthly_cap_minutes` ceilings to govern operational expenditure.

**Sources:** [orchestrator/core/models/system_settings.py:52-53](), [frontend/components/settings/VoiceProfilesSettingsTab.tsx:146-210]()

---

## Onboarding & Wizard Configuration

New workspaces (`is_new_workspace: true`) trigger the Business Intake Wizard [orchestrator/api/workspaces.py:54-56](). The initialization logic detects whether a workspace has non-system agents and automatically boots the onboarding state machine, setting up Firecrawl scraping and RAG ingestion pipelines to seed the initial knowledge graph and profile.

**Sources:** [orchestrator/api/workspaces.py:43-118](), [frontend/components/onboarding/welcome-modal.tsx:1-185]()

---