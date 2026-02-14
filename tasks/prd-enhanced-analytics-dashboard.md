# PRD: Enhanced Analytics Dashboard

## Introduction

Upgrade the unified analytics page (`/analytics`) with comprehensive real-data dashboards covering LLM provider usage (OpenRouter activity sync, per-model metrics, cost projections), Composio tool/action analytics, and PandasAI-powered dynamic chart generation. Two audience views: Users get cost optimization intelligence; Admins get cross-workspace billing and API volume metrics. All dashboards connect to real database data — no mocks.

## Goals

- Sync OpenRouter activity/credits data into local DB for rich LLM analytics
- Surface Composio tool/action usage analytics from existing `AgentAppFeature` data
- Wire PandasAI chart generation into the analytics page for dynamic visualizations
- Give users actionable cost optimization intelligence (model comparison, projections, recommendations)
- Give admins cross-workspace cost, API volume, and billing-ready metrics
- Use Automatos brand colors and glass-card design system throughout

## User Stories

### US-001: Create OpenRouter activity sync service
**Description:** As a developer, I need a backend service that fetches usage data from OpenRouter's `/api/v1/activity` endpoint and stores it locally for analytics queries.

**Acceptance Criteria:**
- [ ] Create `orchestrator/core/llm/openrouter_analytics.py` with `OpenRouterAnalyticsService` class
- [ ] `sync_activity(api_key: str, workspace_id: UUID)` method fetches from `GET https://openrouter.ai/api/v1/activity` with Bearer auth
- [ ] Parses response: date, model, provider_name, usage (cost $), requests, prompt_tokens, completion_tokens, reasoning_tokens, byok_usage_inference
- [ ] Upserts rows into `llm_usage` table matching by workspace_id + model_id + date (avoids duplicates on re-sync)
- [ ] `get_credits(api_key: str)` method fetches from `GET https://openrouter.ai/api/v1/credits` returning total_credits and total_usage
- [ ] `get_key_info(api_key: str)` method fetches from `GET https://openrouter.ai/api/v1/key` returning limit, limit_remaining, usage_daily/weekly/monthly
- [ ] Graceful error handling: logs warning on 401/403/500, does not crash
- [ ] Typecheck passes

### US-002: Create OpenRouter analytics API endpoints
**Description:** As a frontend developer, I need API endpoints to access OpenRouter activity data and credits for the analytics dashboard.

**Acceptance Criteria:**
- [ ] Add endpoints to `orchestrator/api/llm_analytics.py`
- [ ] `POST /api/analytics/openrouter/sync` triggers activity sync for current workspace (requires OpenRouter API key from UserApiKey or system settings)
- [ ] `GET /api/analytics/openrouter/credits` returns credits data (total_credits, total_usage, limit_remaining)
- [ ] `GET /api/analytics/openrouter/key-info` returns key usage stats (daily/weekly/monthly usage, BYOK usage)
- [ ] All endpoints require workspace context and hybrid auth
- [ ] Returns 404 with helpful message if no OpenRouter API key configured
- [ ] Typecheck passes

### US-003: Create Composio analytics API endpoints
**Description:** As a frontend developer, I need API endpoints that aggregate Composio tool/action usage from existing database tables.

**Acceptance Criteria:**
- [ ] Create `orchestrator/api/composio_analytics.py` with router prefix `/api/analytics/composio`
- [ ] `GET /api/analytics/composio/apps` returns connected apps with: app_name, status, total_actions_used (sum of usage_count from AgentAppFeature), agent_count (distinct agents using this app), documents_synced (from ComposioConnection), last_used_at
- [ ] `GET /api/analytics/composio/actions` returns action leaderboard: action_name, app_name, total_usage_count (sum across agents), agent_count, last_used_at — sorted by usage_count desc
- [ ] `GET /api/analytics/composio/agent-tools` returns per-agent tool mapping: agent_id, agent_name, tools (list of {app_name, action_name, usage_count, last_used_at})
- [ ] All endpoints filter by workspace_id from request context
- [ ] All endpoints accept optional `?days=7|30|90` parameter
- [ ] Register router in main app
- [ ] Typecheck passes

### US-004: Create PandasAI analytics chart generation endpoint
**Description:** As a frontend developer, I need an endpoint that generates charts from analytics data using PandasAI.

**Acceptance Criteria:**
- [ ] Create `orchestrator/api/analytics_charts.py` with router prefix `/api/analytics/charts`
- [ ] `POST /api/analytics/charts/generate` accepts JSON body: `{"query": "cost by model last 30 days", "chart_type": "bar|line|pie|area|scatter|auto"}`
- [ ] Endpoint queries `llm_usage` table based on parsed query intent, passes result rows to `PandasAIService.generate_insight()`
- [ ] Returns `{"summary": "text insight", "charts": ["base64-png-string"], "data": [...rows]}`
- [ ] `GET /api/analytics/charts/presets` returns list of pre-built chart configs: `[{"id": "cost-by-model", "title": "Cost by Model", "query": "...", "chart_type": "pie"}, ...]` with at least 6 presets covering: cost by model, tokens over time, cost by provider, agent cost breakdown, model latency comparison, error rates
- [ ] Requires workspace context
- [ ] Typecheck passes

### US-005: Add enhanced LLM model comparison endpoint
**Description:** As a user, I need an endpoint that provides model comparison data for side-by-side analysis of models I'm using.

**Acceptance Criteria:**
- [ ] Add `GET /api/analytics/llm/comparison` to `orchestrator/api/llm_analytics.py`
- [ ] Accepts `?model_ids=model1,model2,model3` (comma-separated, max 4)
- [ ] Returns per-model: display_name, provider, input_cost_per_1k, output_cost_per_1k, context_window, capabilities JSON, total_requests (from llm_usage), total_tokens, total_cost, avg_latency_ms, error_rate, success_rate
- [ ] Data scoped to workspace and optional `?period=7d|30d|90d`
- [ ] Falls back to model registry data if no usage data exists yet
- [ ] Typecheck passes

### US-006: Add cost projection endpoint
**Description:** As a user, I need to see projected monthly costs based on current usage trajectory.

**Acceptance Criteria:**
- [ ] Add `GET /api/analytics/llm/projections` to `orchestrator/api/llm_analytics.py`
- [ ] Calculates daily average cost from llm_usage for the period
- [ ] Projects to 30-day monthly cost: `daily_avg * 30`
- [ ] Returns: current_period_cost, daily_average, projected_monthly, projected_by_model (list of {model_id, projected_monthly_cost}), projected_by_provider
- [ ] Includes comparison to previous period: `change_percent`
- [ ] Scoped to workspace
- [ ] Typecheck passes

### US-007: Create frontend hook for OpenRouter analytics
**Description:** As a frontend developer, I need React Query hooks to fetch OpenRouter credits and activity data.

**Acceptance Criteria:**
- [ ] Add to `frontend/hooks/use-unified-analytics.ts`: `useOpenRouterCredits()` hook calling `GET /api/analytics/openrouter/credits`
- [ ] Add `useOpenRouterKeyInfo()` hook calling `GET /api/analytics/openrouter/key-info`
- [ ] Add `useTriggerOpenRouterSync()` mutation hook calling `POST /api/analytics/openrouter/sync`
- [ ] All hooks use `unifiedAnalyticsKeys` pattern for cache keys
- [ ] Hooks handle 404 gracefully (no OpenRouter key configured) by returning null data
- [ ] `staleTime: 300000` (5 minutes) for credits, `60000` for key info
- [ ] Typecheck passes

### US-008: Create frontend hooks for Composio analytics
**Description:** As a frontend developer, I need React Query hooks for Composio tool usage data.

**Acceptance Criteria:**
- [ ] Add to `frontend/hooks/use-unified-analytics.ts`: `useComposioApps(days)` hook calling `GET /api/analytics/composio/apps?days=X`
- [ ] Add `useComposioActions(days)` hook calling `GET /api/analytics/composio/actions?days=X`
- [ ] Add `useComposioAgentTools(days)` hook calling `GET /api/analytics/composio/agent-tools?days=X`
- [ ] Cache keys follow `unifiedAnalyticsKeys` pattern with `composio` namespace
- [ ] `staleTime: 60000` (1 minute)
- [ ] Typecheck passes

### US-009: Create frontend hooks for charts, comparison, and projections
**Description:** As a frontend developer, I need React Query hooks for PandasAI charts, model comparison, and cost projections.

**Acceptance Criteria:**
- [ ] Add `useAnalyticsChart(query, chartType)` mutation hook calling `POST /api/analytics/charts/generate`
- [ ] Add `useChartPresets()` query hook calling `GET /api/analytics/charts/presets`
- [ ] Add `useModelComparison(modelIds, period)` hook calling `GET /api/analytics/llm/comparison`
- [ ] Add `useCostProjections(period)` hook calling `GET /api/analytics/llm/projections`
- [ ] All hooks follow existing pattern and cache key structure
- [ ] Typecheck passes

### US-010: Build OpenRouter Credits card component
**Description:** As a user, I want to see my OpenRouter credits balance and usage on the analytics page.

**Acceptance Criteria:**
- [ ] Create `frontend/components/analytics/analytics-openrouter-credits.tsx`
- [ ] Shows card with: total credits, total usage, remaining balance, daily/weekly/monthly usage
- [ ] Progress bar showing credits used vs total (color-coded: green <70%, yellow 70-90%, red >90%)
- [ ] "Sync Activity" button that triggers OpenRouter sync and shows loading state
- [ ] Uses `glass-card` styling, Automatos brand chart colors
- [ ] Shows "No OpenRouter key configured" message with link to settings if 404
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-011: Enhance LLM & Costs tab with model comparison section
**Description:** As a user, I want to compare models I'm using side-by-side with cost, performance, and capability data.

**Acceptance Criteria:**
- [ ] Add "Model Comparison" section to `frontend/components/analytics/analytics-costs.tsx`
- [ ] Multi-select dropdown (max 4 models) populated from models in user's llm_usage data
- [ ] Comparison table showing per-model: cost/1K tokens (input+output), total requests, total tokens, total cost, avg latency, error rate, capabilities badges
- [ ] Uses Recharts `RadarChart` to visualize capability ratings across selected models (reasoning, coding, analysis, creativity)
- [ ] Uses `glass-card` styling with brand colors
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-012: Add cost projection section to LLM & Costs tab
**Description:** As a user, I want to see projected monthly costs and period-over-period comparison.

**Acceptance Criteria:**
- [ ] Add "Cost Projections" section to `analytics-costs.tsx` below the existing cost trend chart
- [ ] Shows: current period cost, daily average, projected monthly cost (large number), change % vs previous period (up/down arrow with color)
- [ ] Shows projected cost per model as horizontal bar chart using Recharts `BarChart`
- [ ] Green/red color coding: if projected cost decreased vs last period show green, if increased show red
- [ ] Uses `glass-card` styling
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-013: Build Composio Tools analytics tab
**Description:** As a user, I want a dedicated tab showing which Composio apps/actions my agents use and how often.

**Acceptance Criteria:**
- [ ] Create `frontend/components/analytics/analytics-composio.tsx`
- [ ] StatsBar with 4 cards: Connected Apps (count), Total Actions Used (sum), Most Used App (name + count), Active Integrations (connected count)
- [ ] "Connected Apps" section: grid of app cards showing app_name, status badge (active/disconnected/error), total actions used, agent count, documents synced, last used
- [ ] "Action Leaderboard" section: sortable table with columns: Action Name, App, Usage Count, Agents Using, Last Used — sorted by usage count descending
- [ ] "Agent Tool Mapping" section: expandable rows showing each agent and its assigned Composio tools with usage counts
- [ ] Uses `glass-card` styling, existing Badge component for status
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-014: Add Composio tab to analytics page navigation
**Description:** As a user, I want to access Composio analytics from the main analytics page tabs.

**Acceptance Criteria:**
- [ ] In `frontend/components/analytics/analytics-page.tsx`, add new tab "Tools & Integrations" between "LLM & Costs" and "Admin"
- [ ] Tab renders `<AnalyticsComposio days={days} />` component
- [ ] Tab icon uses `Wrench` from lucide-react
- [ ] Tab visible to all users (not admin-only)
- [ ] Lazy-loads data only when tab is selected (matches existing pattern)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-015: Build PandasAI chart widget component
**Description:** As a user, I want to see AI-generated charts on the analytics page that visualize my data in rich ways.

**Acceptance Criteria:**
- [ ] Create `frontend/components/analytics/analytics-pandas-chart.tsx`
- [ ] Component accepts props: `presetId?: string`, `query?: string`, `chartType?: string`
- [ ] Fetches chart from `POST /api/analytics/charts/generate` or uses preset config
- [ ] Renders base64 chart image with NL summary text below
- [ ] Loading skeleton while generating (shows "Generating chart..." message)
- [ ] Error state if chart generation fails
- [ ] Uses `glass-card` wrapper with `CardHeader` title
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-016: Add PandasAI preset charts to Overview tab
**Description:** As a user, I want to see pre-generated AI charts on the analytics overview tab.

**Acceptance Criteria:**
- [ ] In `frontend/components/analytics/analytics-overview.tsx`, add "AI Insights" section below recommendations
- [ ] Renders 2-3 `AnalyticsPandasChart` components using presets: "cost-by-model" and "tokens-over-time"
- [ ] Charts display in a 2-column grid on desktop, single column on tablet
- [ ] Each chart has a title from the preset config
- [ ] Section has header "AI-Generated Insights" with sparkle icon
- [ ] Only renders if PandasAI endpoint returns data (graceful fallback if service unavailable)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-017: Enhance Admin tab with cross-workspace cost aggregation
**Description:** As an admin, I need to see total platform costs, per-workspace cost breakdown, and API call volumes for billing.

**Acceptance Criteria:**
- [ ] Add `GET /api/admin/analytics/costs` endpoint in `orchestrator/api/llm_analytics.py` (admin-only auth check)
- [ ] Returns: total_platform_cost, total_tokens, total_requests, cost_by_workspace (list of {workspace_id, workspace_name, total_cost, total_tokens, total_requests}), cost_by_provider, daily_cost_trend
- [ ] Queries `llm_usage` table across ALL workspaces (no workspace filter)
- [ ] Accepts `?period=7d|30d|90d`
- [ ] Typecheck passes

### US-018: Build admin cost dashboard frontend section
**Description:** As an admin, I want to see platform-wide cost analytics with per-workspace breakdown in the Admin tab.

**Acceptance Criteria:**
- [ ] Update `frontend/components/analytics/analytics-admin.tsx` to add "Platform Costs" section
- [ ] StatsBar cards: Total Platform Cost ($), Total Tokens (formatted), Total API Requests, Active Providers (count)
- [ ] Stacked area chart showing daily cost trend by provider using Recharts `AreaChart` with brand chart colors
- [ ] Per-workspace cost table: Workspace Name, Plan, Total Cost, Total Tokens, Requests, Top Model — sortable by cost
- [ ] Provider distribution pie chart using `PieChart` from Recharts
- [ ] Add `useAdminCostAnalytics(days)` hook in `use-unified-analytics.ts`
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-019: Register all new API routers in main app
**Description:** As a developer, I need all new analytics routers registered in the FastAPI app so endpoints are accessible.

**Acceptance Criteria:**
- [ ] In the main app setup (wherever routers are registered), import and include: `composio_analytics.router`, `analytics_charts.router`
- [ ] Verify all new endpoints appear in API docs at `/docs`
- [ ] No import errors or circular dependencies
- [ ] Typecheck passes

## Functional Requirements

- FR-1: OpenRouter activity data syncs into local `llm_usage` table with deduplication
- FR-2: OpenRouter credits endpoint returns real balance/usage data
- FR-3: Composio analytics aggregate from existing `AgentAppFeature` and `ComposioConnection` tables
- FR-4: PandasAI charts generate from real `llm_usage` DB data, not mocks
- FR-5: Model comparison shows real usage metrics alongside registry data
- FR-6: Cost projections extrapolate from actual daily averages
- FR-7: Admin endpoints query across ALL workspaces (no workspace filter)
- FR-8: All user-facing endpoints filter by workspace_id
- FR-9: All charts use Automatos brand colors (--chart-1 through --chart-5)
- FR-10: All cards use glass-card design pattern
- FR-11: New "Tools & Integrations" tab added to analytics navigation
- FR-12: PandasAI charts gracefully degrade if service unavailable

## Non-Goals

- No WebSocket real-time streaming — polling/React Query refresh is sufficient
- No custom date range picker — fixed presets (7d/30d/90d) only
- No billing/payment processing — display usage data only
- No notification/alert system — display only
- No mobile optimization — desktop and tablet only
- No PDF report generation — just dashboard views
- No Composio V3 Logs API integration yet — use existing DB tables only

## Technical Considerations

- All new backend files follow existing FastAPI pattern with `RequestContext` + `get_request_context_hybrid`
- PandasAI service already exists at `modules/tools/services/pandas_ai_service.py` — reuse `get_pandasai_service()`
- OpenRouter API requires management key for `/api/v1/activity` and `/api/v1/credits` — resolve from `UserApiKey` table or `OPENROUTER_API_KEY` env var
- Frontend uses Recharts v2.8.0 for interactive charts, PandasAI for static rich visualizations
- All components use existing shadcn/ui primitives (Card, Badge, Tabs, Skeleton, Progress)
- Brand colors defined in CSS variables: `--chart-1` (orange), `--chart-2` (blue), `--chart-3` (teal), `--chart-4` (purple), `--chart-5` (amber)

## Success Metrics

- Users can see real LLM usage costs within 5 seconds of opening analytics
- Users can compare models they're using with real performance data
- Users can see projected monthly costs based on actual trajectory
- Admins can view per-workspace cost breakdown across the platform
- Composio tool usage visible with agent-to-tool mapping
- PandasAI generates at least 2 meaningful charts on the overview tab
- All data is real — no mock data anywhere in the dashboard
