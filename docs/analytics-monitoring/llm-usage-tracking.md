# LLM Usage Tracking

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md](docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md)
- [frontend/components/activity/memory-card.tsx](frontend/components/activity/memory-card.tsx)
- [frontend/components/activity/memory/health-banner.tsx](frontend/components/activity/memory/health-banner.tsx)
- [frontend/components/activity/memory/index.ts](frontend/components/activity/memory/index.ts)
- [frontend/components/activity/memory/memory-sidebar.tsx](frontend/components/activity/memory/memory-sidebar.tsx)
- [frontend/components/activity/projects/index.ts](frontend/components/activity/projects/index.ts)
- [frontend/components/marketplace/llm-model-card.tsx](frontend/components/marketplace/llm-model-card.tsx)
- [frontend/components/marketplace/llm-model-detail-modal.tsx](frontend/components/marketplace/llm-model-detail-modal.tsx)
- [frontend/components/marketplace/marketplace-agents-tab.tsx](frontend/components/marketplace/marketplace-agents-tab.tsx)
- [frontend/components/marketplace/marketplace-llms-tab.tsx](frontend/components/marketplace/marketplace-llms-tab.tsx)
- [frontend/components/marketplace/marketplace-plugin-detail-modal.tsx](frontend/components/marketplace/marketplace-plugin-detail-modal.tsx)
- [frontend/components/marketplace/marketplace-plugins-tab.tsx](frontend/components/marketplace/marketplace-plugins-tab.tsx)
- [frontend/components/marketplace/marketplace-skills-tab.tsx](frontend/components/marketplace/marketplace-skills-tab.tsx)
- [frontend/components/marketplace/marketplace-tools-tab.tsx](frontend/components/marketplace/marketplace-tools-tab.tsx)
- [frontend/components/settings/ApiKeysSettingsTab.tsx](frontend/components/settings/ApiKeysSettingsTab.tsx)
- [frontend/hooks/use-memory-explorer-api.ts](frontend/hooks/use-memory-explorer-api.ts)
- [frontend/hooks/use-openrouter-api.ts](frontend/hooks/use-openrouter-api.ts)
- [orchestrator/api/composio_analytics.py](orchestrator/api/composio_analytics.py)
- [orchestrator/api/database_analytics.py](orchestrator/api/database_analytics.py)
- [orchestrator/api/llm_analytics.py](orchestrator/api/llm_analytics.py)
- [orchestrator/api/llm_marketplace.py](orchestrator/api/llm_marketplace.py)
- [orchestrator/api/marketplace.py](orchestrator/api/marketplace.py)
- [orchestrator/api/marketplace_plugins.py](orchestrator/api/marketplace_plugins.py)
- [orchestrator/api/memory_stats.py](orchestrator/api/memory_stats.py)
- [orchestrator/api/openrouter_marketplace.py](orchestrator/api/openrouter_marketplace.py)
- [orchestrator/api/user_api_keys.py](orchestrator/api/user_api_keys.py)
- [orchestrator/core/database/migrations/042_openrouter_models_cache.sql](orchestrator/core/database/migrations/042_openrouter_models_cache.sql)
- [orchestrator/core/llm/clients/__init__.py](orchestrator/core/llm/clients/__init__.py)
- [orchestrator/core/llm/openrouter_analytics.py](orchestrator/core/llm/openrouter_analytics.py)
- [orchestrator/core/llm/usage_tracker.py](orchestrator/core/llm/usage_tracker.py)
- [orchestrator/scripts/seed_llm_marketplace.py](orchestrator/scripts/seed_llm_marketplace.py)
- [orchestrator/tests/test_prd143_obs_routers_batch2.py](orchestrator/tests/test_prd143_obs_routers_batch2.py)

</details>



## Purpose and Scope

This document describes the LLM usage tracking system that records every LLM API call for cost calculation, analytics, and optimization. The system captures token counts, latency, model information, and calculates costs based on a model pricing registry. Usage data is workspace-scoped and powers the unified analytics dashboard.

The tracking system integrates with multiple LLM providers (OpenAI, Anthropic, OpenRouter, Google, Azure OpenAI, xAI, etc.) and supports both platform-provided keys and user-provided BYOK (Bring Your Own Key) credentials. All tracked usage is attributed to workspaces and optionally to specific agents or workflow executions.

**Key Capabilities:**
- Per-request token and cost tracking for all LLM providers [orchestrator/api/llm_analytics.py:113-164]()
- Dual-source cost calculation: preference for `llm_usage` table with fallback to `agent_statistics` [orchestrator/api/llm_analytics.py:223-235]()
- OpenRouter management API integration for credits, key limits, and activity sync [orchestrator/core/llm/openrouter_analytics.py:5-11]()
- Workspace-scoped analytics with strict tenant isolation [orchestrator/api/llm_analytics.py:31-41]()
- Super-admin oversight for platform-wide aggregate usage and infrastructure health [orchestrator/api/llm_analytics.py:42-46]()

Sources: [orchestrator/api/llm_analytics.py:1-55](), [orchestrator/core/llm/openrouter_analytics.py:1-25](), [orchestrator/api/statistics.py:1-33]()

---

## LLM Provider Tracking Flow

Every LLM request follows a lifecycle where usage is captured from the provider's response and persisted for analytics.

### Credential Resolution and Tracking Logic

Title: "LLM Request and Usage Capture Flow"
```mermaid
graph TD
    "AgentRequest"["Agent Execution Request"]
    "LLMManager"["LLMManager.generate_response"]
    
    subgraph "Provider_Execution"["Provider Execution"]
        "OpenAI"["OpenAIProvider"]
        "OpenRouter"["OpenRouterProvider"]
        "Grok"["GrokProvider"]
    end
    
    "LLMUsage"["LLMUsage (SQLAlchemy Model)"]
    "UsageTable"[("LLMUsage Table")]
    "AnalyticsAPI"["LLM Analytics API"]

    "AgentRequest" --> "LLMManager"
    "LLMManager" --> "OpenAI"
    "LLMManager" --> "OpenRouter"
    "LLMManager" --> "Grok"
    
    "OpenAI" -->|"usage metadata"| "LLMUsage"
    "OpenRouter" -->|"usage metadata"| "LLMUsage"
    "Grok" -->|"usage metadata"| "LLMUsage"
    
    "LLMUsage" -->|"db.add()"| "UsageTable"
    "UsageTable" -->|"SQL Query"| "AnalyticsAPI"
    "AnalyticsAPI" -->|"JSON Response"| "FrontendUI"["Analytics Dashboard"]
```

### Response Capture
The system captures usage metrics during the execution of LLM requests. These metrics are persisted into the `LLMUsage` model which includes `input_tokens`, `output_tokens`, and `total_cost` [orchestrator/api/llm_analytics.py:60-80](). 

Sources: [orchestrator/api/llm_analytics.py:58-86](), [orchestrator/core/llm/openrouter_analytics.py:118-137]()

---

## Database Schema & Analytics API

### The LLMUsage Model
The `LLMUsage` table is the source of truth for granular tracking. It records:
- **Identity**: `workspace_id`, `agent_id`, `execution_id` [orchestrator/api/llm_analytics.py:129-132]()
- **Metrics**: `input_tokens`, `output_tokens`, `total_tokens`, `latency_ms` [orchestrator/api/llm_analytics.py:140-143]()
- **Economics**: `input_cost`, `output_cost`, `total_cost`, `is_byok` [orchestrator/api/llm_analytics.py:143-144]()
- **Metadata**: `model_id`, `provider`, `tier`, `status` [orchestrator/api/llm_analytics.py:127-132]()

### Analytics Endpoints
The `llm_analytics.py` module provides a suite of REST endpoints:
- `GET /api/analytics/llm/usage`: Token usage grouped by dimension (model, provider, agent, tier) [orchestrator/api/llm_analytics.py:113-164]().
- `GET /api/analytics/llm/costs`: Financial breakdown by model or provider, including daily trends [orchestrator/api/llm_analytics.py:167-216]().
- `GET /api/analytics/llm/summary`: High-level dashboard metrics, including total tokens, cost, and error rates [orchestrator/api/llm_analytics.py:219-255]().

Sources: [orchestrator/api/llm_analytics.py:58-86](), [orchestrator/api/llm_analytics.py:113-164](), [orchestrator/api/llm_analytics.py:167-216]()

---

## OpenRouter Sync Strategy

OpenRouter requires a dual-tracking strategy because it acts as an aggregator for numerous models.

1.  **Activity Sync**: The `OpenRouterAnalyticsService` periodically fetches data from OpenRouter's `/activity` endpoint [orchestrator/core/llm/openrouter_analytics.py:44-58]().
2.  **Deduplication**: The service uses a composite check of `workspace_id`, `model_id`, and `created_at` (date) to avoid double-counting synced rows versus real-time captured rows [orchestrator/core/llm/openrouter_analytics.py:97-106]().
3.  **Manual Trigger**: Super-admins can manually trigger a sync via `POST /api/analytics/llm/openrouter/sync` [orchestrator/api/llm_analytics.py:48-55]().

### Credit Monitoring
The system monitors OpenRouter credit balances and key limits:
- `get_credits`: Fetches account balance [orchestrator/core/llm/openrouter_analytics.py:154-171]().
- `get_key_info`: Fetches specific API key limits and remaining quota [orchestrator/core/llm/openrouter_analytics.py:185-195]().

Sources: [orchestrator/core/llm/openrouter_analytics.py:27-148](), [orchestrator/core/llm/openrouter_analytics.py:154-195](), [orchestrator/api/llm_analytics.py:47-55]()

---

## Access Control & Multi-Tenancy

The tracking system enforces strict access boundaries:
- **Workspace Owners/Admins**: Can view usage and costs strictly filtered by their `workspace_id` [orchestrator/api/llm_analytics.py:31-41]().
- **Super Admins**: Can access platform-wide aggregate analytics via the `admin_router` [orchestrator/api/llm_analytics.py:42-46]().
- **Mutation Lock**: Destructive operations (like clearing usage logs) and management tasks (OpenRouter sync) are restricted to the `super_admin` role [orchestrator/api/llm_analytics.py:51-55]().

### Verification
Access controls are verified by a router-wide `require_super_admin` dependency for admin routes [orchestrator/api/llm_analytics.py:45](). Workspace-level access is audited to ensure no cross-workspace data leakage [orchestrator/tests/test_prd143_obs_routers_batch2.py:147-152]().

Sources: [orchestrator/api/llm_analytics.py:31-55](), [orchestrator/tests/test_prd143_obs_routers_batch2.py:1-21](), [docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md:51-62]()

---

## Statistics & Metrics

Beyond LLM tokens, the system tracks broader operational metrics:
- **Agent Statistics**: Counts of active/inactive agents and execution success rates [orchestrator/api/statistics.py:38-82]().
- **System Metrics**: Host-level health including CPU and memory usage [orchestrator/api/statistics.py:88-116]().
- **Database Analytics**: Real-time tracking of query execution times and row counts via `database_query_audit` [orchestrator/api/database_analytics.py:27-69]().

Title: "System Metrics and Analytics Architecture"
```mermaid
graph TD
    "RequestContext"["RequestContext (Hybrid Auth)"]
    "DB_Session"["SQLAlchemy Session"]

    subgraph "Analytics_Routers"["Analytics Routers"]
        "LLM_Router"["llm_analytics.py"]
        "Stats_Router"["statistics.py"]
        "DB_Router"["database_analytics.py"]
    end

    subgraph "Storage"["Storage Layer"]
        "LLMUsage_Table"[("LLMUsage")]
        "WorkflowExec_Table"[("WorkflowExecution")]
        "DBAudit_Table"[("database_query_audit")]
    end

    "RequestContext" --> "LLM_Router"
    "RequestContext" --> "Stats_Router"
    "RequestContext" --> "DB_Router"

    "LLM_Router" -->|"Query"| "LLMUsage_Table"
    "Stats_Router" -->|"Query"| "WorkflowExec_Table"
    "DB_Router" -->|"Query"| "DBAudit_Table"
```

Sources: [orchestrator/api/statistics.py:1-33](), [orchestrator/api/database_analytics.py:1-25](), [orchestrator/api/llm_analytics.py:113-119]()

---