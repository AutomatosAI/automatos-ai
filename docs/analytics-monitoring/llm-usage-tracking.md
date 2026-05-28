# LLM Usage Tracking

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/52-UNIFIED-ANALYTICS.md](docs/PRDS/52-UNIFIED-ANALYTICS.md)
- [frontend/app/analytics/page.tsx](frontend/app/analytics/page.tsx)
- [frontend/components/analytics/analytics-admin.tsx](frontend/components/analytics/analytics-admin.tsx)
- [frontend/components/analytics/analytics-agents.tsx](frontend/components/analytics/analytics-agents.tsx)
- [frontend/components/analytics/analytics-costs.tsx](frontend/components/analytics/analytics-costs.tsx)
- [frontend/components/analytics/analytics-documents.tsx](frontend/components/analytics/analytics-documents.tsx)
- [frontend/components/analytics/analytics-memory.tsx](frontend/components/analytics/analytics-memory.tsx)
- [frontend/components/analytics/analytics-openrouter-credits.tsx](frontend/components/analytics/analytics-openrouter-credits.tsx)
- [frontend/components/analytics/analytics-overview.tsx](frontend/components/analytics/analytics-overview.tsx)
- [frontend/components/analytics/analytics-page.tsx](frontend/components/analytics/analytics-page.tsx)
- [frontend/components/analytics/analytics-pandas-chart.tsx](frontend/components/analytics/analytics-pandas-chart.tsx)
- [frontend/components/analytics/analytics-plan-usage.tsx](frontend/components/analytics/analytics-plan-usage.tsx)
- [frontend/components/analytics/analytics-recommendations.tsx](frontend/components/analytics/analytics-recommendations.tsx)
- [frontend/components/analytics/analytics-workflows.tsx](frontend/components/analytics/analytics-workflows.tsx)
- [frontend/components/dashboard/widgets/system-health-widget.tsx](frontend/components/dashboard/widgets/system-health-widget.tsx)
- [frontend/components/knowledge/QueryTemplatesGrid.tsx](frontend/components/knowledge/QueryTemplatesGrid.tsx)
- [frontend/components/system/rag-configuration.tsx](frontend/components/system/rag-configuration.tsx)
- [frontend/hooks/use-unified-analytics.ts](frontend/hooks/use-unified-analytics.ts)
- [orchestrator/api/llm_analytics.py](orchestrator/api/llm_analytics.py)
- [orchestrator/core/llm/openrouter_analytics.py](orchestrator/core/llm/openrouter_analytics.py)

</details>



## Purpose and Scope

This document describes the LLM usage tracking system that records every LLM API call for cost calculation, analytics, and optimization. The system captures token counts, latency, model information, and calculates costs based on a model pricing registry. Usage data is workspace-scoped and powers the unified analytics dashboard.

The tracking system integrates with multiple LLM providers (OpenAI, Anthropic, OpenRouter, Google, Azure OpenAI, xAI, etc.) and supports both platform-provided keys and user-provided BYOK (Bring Your Own Key) credentials. All tracked usage is attributed to workspaces and optionally to specific agents or workflow executions.

**Key Capabilities:**
- Per-request token and cost tracking for all LLM providers [orchestrator/api/llm_analytics.py:87-138]()
- Dual-source cost calculation: preference for `llm_usage` table with fallback to `agent_statistics` [frontend/hooks/use-unified-analytics.ts:70-73]()
- OpenRouter management API integration for credits, key info, and activity sync [orchestrator/core/llm/openrouter_analytics.py:5-11]()
- Workspace-scoped analytics with admin override for platform-wide views [frontend/hooks/use-unified-analytics.ts:12-14]()
- Automated cost optimization recommendations based on usage patterns [orchestrator/api/llm_analytics.py:254-268]()

Sources: [orchestrator/api/llm_analytics.py:1-28](), [frontend/hooks/use-unified-analytics.ts:1-43](), [orchestrator/core/llm/openrouter_analytics.py:1-11]()

---

## LLM Provider Tracking Flow

Every LLM request follows a lifecycle where usage is captured from the provider's response and persisted for analytics.

### Credential Resolution and Tracking Logic

Title: "LLM Request and Usage Capture Flow"
```mermaid
graph TB
    "Request"["Agent/Workflow Request"]
    "LLMManager"["LLMManager.generate_response"]
    
    subgraph "Provider_Execution"["Provider Execution"]
        "OpenAI"["OpenAIProvider"]
        "OpenRouter"["OpenRouterProvider"]
        "Grok"["GrokProvider"]
    end
    
    "LLMResponse"["LLMResponse Object"]
    "UsageTable"[("LLMUsage Table")]
    "StreamHandler"["StreamingChatService"]

    "Request" --> "LLMManager"
    "LLMManager" --> "OpenAI"
    "LLMManager" --> "OpenRouter"
    "LLMManager" --> "Grok"
    
    "OpenAI" -->|"usage metadata"| "LLMResponse"
    "OpenRouter" -->|"usage metadata"| "LLMResponse"
    "Grok" -->|"usage metadata"| "LLMResponse"
    
    "LLMResponse" -->|"async write"| "UsageTable"
    "LLMResponse" -->|"format_aisdk_usage"| "StreamHandler"
    "StreamHandler" -->|"d:type:usage"| "UserUI"["Frontend Analytics"]
```

### Response Capture
Providers implement `generate_response` to return a standardized `LLMResponse` object containing a `usage` dictionary with `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- **OpenAI**: Extracts usage directly from the response object [orchestrator/api/llm_analytics.py:21-25]().
- **OpenRouter**: Handles usage extraction from the aggregator response [orchestrator/core/llm/openrouter_analytics.py:111-115]().
- **Sync Logic**: For OpenRouter, the `OpenRouterAnalyticsService` fetches historical activity and upserts it into `LLMUsage` [orchestrator/core/llm/openrouter_analytics.py:44-50]().

Sources: [orchestrator/api/llm_analytics.py:110-126](), [orchestrator/core/llm/openrouter_analytics.py:77-89]()

---

## Database Schema & Analytics API

### The LLMUsage Model
The `LLMUsage` table is the source of truth for granular tracking. It records:
- **Identity**: `workspace_id`, `agent_id`, `execution_id` [orchestrator/api/llm_analytics.py:101-103]()
- **Metrics**: `input_tokens`, `output_tokens`, `total_tokens`, `latency_ms` [orchestrator/api/llm_analytics.py:114-117]()
- **Economics**: `input_cost`, `output_cost`, `total_cost`, `is_byok` [orchestrator/api/llm_analytics.py:117-118]()
- **Metadata**: `model_id`, `provider`, `tier`, `status` [orchestrator/api/llm_analytics.py:101-104]()

### Analytics Endpoints
The `llm_analytics.py` module provides a suite of REST endpoints for the frontend:
- `GET /api/analytics/llm/usage`: Grouped token usage (by model, provider, agent, etc.) [orchestrator/api/llm_analytics.py:87-108]().
- `GET /api/analytics/llm/costs`: Financial breakdown, including daily cost trends [orchestrator/api/llm_analytics.py:141-164]().
- `GET /api/analytics/llm/summary`: High-level dashboard metrics, including error rates and top models [orchestrator/api/llm_analytics.py:194-221]().

Sources: [orchestrator/api/llm_analytics.py:34-60](), [orchestrator/api/llm_analytics.py:87-138](), [orchestrator/api/llm_analytics.py:141-191]()

---

## OpenRouter Sync Strategy

OpenRouter requires a dual-tracking strategy because it acts as an aggregator for 200+ models.

1.  **Direct Tracking**: Captured during real-time inference.
2.  **Activity Sync**: The `OpenRouterAnalyticsService` periodically fetches data from OpenRouter's `/activity` endpoint to ensure consistency [orchestrator/core/llm/openrouter_analytics.py:44-58]().
3.  **Deduplication**: The service uses a composite check of `workspace_id`, `model_id`, and `created_at` (date) to avoid double-counting synced rows versus real-time captured rows [orchestrator/core/llm/openrouter_analytics.py:97-109]().

### Credit Monitoring
The system monitors OpenRouter credit balances and key limits to provide proactive warnings [orchestrator/core/llm/openrouter_analytics.py:154-171]().

Sources: [orchestrator/core/llm/openrouter_analytics.py:27-38](), [orchestrator/core/llm/openrouter_analytics.py:44-75](), [orchestrator/core/llm/openrouter_analytics.py:185-200]()

---

## Dual-Source Strategy

The frontend analytics dashboard employs a robust "prefer-granular-fallback-to-aggregate" logic to calculate costs.

Title: "Unified Analytics Data Consolidation"
```mermaid
graph LR
    subgraph "Data_Sources"
        "UsageTable"[("LLMUsage (llm_usage)")]
        "AgentStats"[("Agent.model_usage_stats")]
        "OR_API"["OpenRouter Management API"]
    end

    "useAnalyticsOverview"["useAnalyticsOverview Hook"]
    "UI_Display"["Analytics Dashboard StatsBar"]

    "UsageTable" -->|"llmSummary.total_cost"| "useAnalyticsOverview"
    "AgentStats" -->|"sum(agent.cost)"| "useAnalyticsOverview"
    "OR_API" -->|"credits/activity"| "OpenRouterSync"
    
    "useAnalyticsOverview" -->|"totalCost logic"| "UI_Display"
```

In `useAnalyticsOverview`, the system attempts to fetch the `llm_summary`. If `llm_summary.total_cost` is greater than 0, it is used as the primary metric. Otherwise, it falls back to summing the `model_usage_stats` field from the list of all agents [frontend/hooks/use-unified-analytics.ts:70-73]().

Sources: [frontend/hooks/use-unified-analytics.ts:46-75](), [frontend/components/analytics/analytics-costs.tsx:144-152]()

---

## UI Components and Visualization

### Analytics Dashboard
The dashboard uses `StatsBar` to display real-time metrics and `Recharts` for trend visualization.
- **Cost Analytics**: Visualizes daily costs by model and model comparisons using `AreaChart` and `BarChart` [frontend/components/analytics/analytics-costs.tsx:206-230]().
- **Agent Analytics**: Provides an expanded panel showing memory importance vs. token usage per agent [frontend/components/analytics/analytics-agents.tsx:47-72]().
- **Admin View**: Provides a platform-wide view of top spenders (workspaces) and plan distribution [frontend/components/analytics/analytics-admin.tsx:164-198]().
- **AI Charts**: Uses `AnalyticsPandasChart` to generate dynamic visualizations from natural language queries [frontend/components/analytics/analytics-pandas-chart.tsx:15-39]().

### Recommendations
The system provides actionable insights via `AnalyticsRecommendations`, categorized by `cost`, `performance`, `document`, and `quota` [frontend/components/analytics/analytics-recommendations.tsx:46-71]().

Sources: [frontend/components/analytics/analytics-costs.tsx:1-50](), [frontend/components/analytics/analytics-agents.tsx:162-180](), [frontend/components/analytics/analytics-admin.tsx:1-42](), [frontend/components/analytics/analytics-recommendations.tsx:19-31]()

---