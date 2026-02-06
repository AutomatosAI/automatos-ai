# PRD: Unified Analytics Page

**Status:** In Progress — Frontend complete, backend & cleanup pending
**Branch:** `analytics-page`
**Created:** February 2026

---

## Completion Status

### DONE — Frontend Implementation

| Item | Files | Status |
|------|-------|--------|
| Analytics page route & tabbed layout | `app/analytics/page.tsx`, `components/analytics/analytics-page.tsx` | Done |
| Overview tab (summary cards, plan usage, memory, recommendations) | `analytics-overview.tsx` | Done |
| Agents tab (summary + sortable table + health badges) | `analytics-agents.tsx` | Done |
| Workflows tab (summary + trend chart + sortable table) | `analytics-workflows.tsx` | Done |
| Documents tab (summary + RAG perf + never-accessed alerts) | `analytics-documents.tsx` | Done |
| LLM & Costs tab (tokens, cost trend, per-model, per-agent) | `analytics-costs.tsx` | Done |
| Admin tab (platform metrics + workspace table + CSV export) | `analytics-admin.tsx` | Done |
| AI recommendations (dismiss/snooze, type-based styling) | `analytics-recommendations.tsx` | Done |
| Plan usage bars (color-coded quotas, pilot-mode nulls) | `analytics-plan-usage.tsx` | Done |
| Automatos Memory card (learned vs user-set) | `analytics-memory.tsx` | Done |
| Unified analytics hooks (8 hooks aggregating existing APIs) | `hooks/use-unified-analytics.ts` | Done |
| Sidebar updated — "Analytics" visible to ALL users | `components/layout/sidebar.tsx` | Done |
| Dashboard redirects to `/analytics` | `app/dashboard/page.tsx` | Done |
| Removed monitoring tab from Workflows | `components/workflows/workflow-management.tsx` | Done |
| Removed performance tab from Agents | `components/agents/agent-management.tsx` | Done |
| Removed analytics tab from Documents | `components/documents/document-management.tsx` | Done |

### PENDING — Post-Merge Tasks

| Item | Notes |
|------|-------|
| **Merge with latest `main`** | Resolve any conflicts from recent PRs |
| **Browser testing** | Verify all 6 tabs render correctly with live backend |
| **TypeScript verification** | Run `tsc` once `node_modules` are installed |
| **Delete old analytics files** | `monitoring-tab.tsx`, `agent-performance.tsx`, `analytics-tab.tsx`, `document-analytics.tsx`, `performance-analytics.tsx` — only after confirming no regressions |
| **Backend: Plan usage endpoint** | `GET /api/analytics/plan-usage` — workspace plan limits + current usage |
| **Backend: Admin cross-workspace endpoint** | `GET /api/admin/analytics/workspaces` — aggregated stats across workspaces |
| **Backend: AI recommendations endpoint** | `GET /api/analytics/recommendations` — LLM-generated optimization suggestions |
| **Database: New tables** | `workspace_plans`, `workspace_usage`, `token_usage_log` (see Data Model section) |
| **Plan tiers configuration** | Define actual limits after pilot monitoring — currently using `null` (no limit) |

---

## Introduction

Replace all scattered analytics (workflow monitoring tab, agent performance tab, document analytics tab, `/analytics` page, `/dashboard` page) with a single unified `/analytics` page that serves two audiences:

1. **Workspace Users** — Actionable intelligence: "What's costing me money? How do I fix it? Are my agents/workflows/documents performing well? Is Automatos suggesting smarter choices?"
2. **Super Admins** — Operational monitoring: Cross-workspace costs, API usage, token consumption, plan utilization, and workspace-level reporting.

Both views live on the same `/analytics` route. Super admins see an additional "All Workspaces" tab with elevated data.

### The Problem Today

Analytics is scattered across 5+ locations:
- `/analytics` — Admin-only "Intelligence & Learning" page (AI benchmarking, pattern recognition)
- `/dashboard` — Admin-only system health metrics
- `/workflows` → Monitoring tab (memory, RAG, execution stats)
- `/agents` → Performance tab (resource usage, task history, activity logs)
- `/documents` → Analytics tab (usage patterns, search performance, RAG effectiveness)

None of these are accessible to regular users. There's no cost/token visibility, no LLM optimization suggestions, and no plan/quota tracking.

### What Changes

- **New:** Single `/analytics` page accessible to ALL users (not admin-only)
- **Remove:** Analytics tabs from `/workflows`, `/agents`, `/documents` pages entirely
- **Remove:** Existing `/analytics` (Intelligence & Learning) page — replaced by unified page
- **Remove:** `/dashboard` page — relevant metrics folded into unified analytics
- **Keep:** Individual pages (`/agents`, `/workflows`, `/documents`) retain their management functionality, just without analytics tabs

---

## Goals

- Give workspace users clear, non-technical visibility into what they're spending and how well things are working
- Surface AI-powered recommendations (e.g. "Switch Agent X to a cheaper LLM — it's a simple email task")
- Show plan usage vs. quotas so users understand their consumption
- Provide super admins with cross-workspace cost, usage, and API monitoring
- Consolidate all analytics into one well-organized page, eliminating redundancy
- Make the page accessible to ALL users (not admin-only)

---

## User Stories

### US-001: Analytics page route and role-based layout
**Description:** As a user, I want to access `/analytics` from the sidebar so I can see my workspace analytics.

**Acceptance Criteria:**
- [x] `/analytics` route accessible to ALL authenticated users (remove `requiredRole: 'admin'`)
- [x] Sidebar item renamed from "Intelligence & Learning" to "Analytics" with `BarChart3` icon
- [x] Page renders a tabbed layout: **Overview | Agents | Workflows | Documents | LLM & Costs**
- [x] Super admins see an additional **Admin** tab (6th tab)
- [x] Role detection uses existing `useSystemRole()` hook
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-002: Overview tab — workspace health at a glance
**Description:** As a user, I want a quick summary of my workspace so I know if anything needs attention without diving into details.

**Acceptance Criteria:**
- [x] Top row: 4 summary cards — **Total Agents** (active/total), **Total Workflows** (active/total), **Documents Stored** (count + size), **Monthly Cost** (current period spend)
- [x] Each card shows trend indicator (up/down arrow + percentage vs. previous period)
- [x] Plan usage bar below cards: "Using X of Y agents", "X of Y GB storage", "X of Y API calls" with progress bars color-coded (green < 70%, yellow 70-90%, red > 90%)
- [x] "Recommendations" section below: AI-powered suggestion cards (see US-010)
- [x] Time range selector: 7d / 30d / 90d (default 30d)
- [x] All data fetched from API, loading skeletons while fetching
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-003: Agents tab — per-agent performance
**Description:** As a user, I want to see how each of my agents is performing so I can identify underperformers and optimize them.

**Acceptance Criteria:**
- [x] Agent summary cards: total agents, average success rate, total tokens used, total cost
- [x] Agent performance table with columns: **Name | Status | Success Rate | Avg Run Time | Tokens Used | Cost | LLM Model | Last Run**
- [x] Table is sortable by any column
- [x] Agents flagged with warning badge if: success rate < 80%, cost is top 20% of agents, or using an expensive LLM for simple tasks
- [x] Filter by: status (active/inactive)
- [ ] Clicking an agent row expands to show: execution history (last 20 runs), error breakdown, token usage over time (sparkline)
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-004: Workflows tab — workflow execution analytics
**Description:** As a user, I want to see how my workflows are performing so I can spot failures and bottlenecks.

**Acceptance Criteria:**
- [x] Workflow summary cards: total workflows, executions this period, overall success rate, avg execution time
- [x] Workflow performance table: **Name | Runs | Success Rate | Avg Duration | Tokens Used | Cost | Last Run | Status**
- [x] Table sortable by any column
- [x] Execution trend chart: daily runs (success vs. failure stacked bar) for selected time range
- [ ] Clicking a workflow row expands to show: recent execution timeline (last 20), step-by-step breakdown with per-step timing, error log for failed runs
- [ ] Workflows flagged if: success rate < 80%, avg duration increased > 20% vs previous period
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-005: Documents tab — storage and RAG effectiveness
**Description:** As a user, I want to see which documents are being used and which aren't, so I can improve my knowledge base.

**Acceptance Criteria:**
- [x] Summary cards: total documents, total storage used (+ quota bar), total RAG queries this period, avg retrieval confidence
- [x] Documents table: **Name | Type | Size | Uploaded | Last Accessed | RAG Queries | Confidence Score**
- [x] Sort by any column, default sort by "Last Accessed" (never-accessed documents float to top with "Never used" badge)
- [x] "Unused Documents" alert card: count of documents never retrieved by RAG, with suggestion to improve tags/keywords
- [x] RAG performance section: top search queries, avg response time, retrieval success rate, sources distribution
- [ ] Documents flagged if: never accessed, low confidence scores, or no tags/keywords
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-006: LLM & Costs tab — token usage and model efficiency
**Description:** As a user, I want to see my LLM token usage and costs broken down by model so I can optimize spending.

**Acceptance Criteria:**
- [x] Summary cards: total tokens this period (input + output), total LLM cost, cost per successful task, most expensive agent
- [x] Cost breakdown chart: stacked area chart showing daily cost by LLM model over time range
- [x] Token usage table: **LLM Model | Requests | Input Tokens | Output Tokens | Total Cost | Avg Cost/Request | Used By (agent count)**
- [x] Per-agent cost breakdown: which agents are using which models and costing how much
- [ ] Cost trend: line chart comparing current period to previous period
- [x] Plan quota section: "API calls: 8,423 / 10,000 this month" with progress bar
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-007: Automatos Memory section (within Overview)
**Description:** As a user, I want to review and tweak what Automatos remembers about me so my AI interactions are more personalized and accurate.

**Acceptance Criteria:**
- [x] "What Automatos Knows" card on Overview tab showing key stored preferences
- [x] Each memory item shows: key, value, last updated, source (user-set vs. learned)
- [x] "Automatos learned this from your interactions" vs "You set this" visual distinction
- [ ] Edit button on each item to correct/update values inline
- [ ] Link to full memory management page if one exists, or expand to show all items
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-008: Admin tab — cross-workspace monitoring
**Description:** As a super admin, I want to see usage and costs across all workspaces so I can monitor the platform and identify billing concerns.

**Acceptance Criteria:**
- [x] Tab only visible to users where `isAdmin === true`
- [x] Top-level metrics: total workspaces, total users, total API calls (all workspaces), total token usage, total platform cost
- [x] Workspace table: **Workspace Name | Plan | Users | Agents | Workflows | API Calls | Tokens Used | Cost | Status**
- [x] Export to CSV button for workspace usage data
- [ ] Table sortable, filterable by plan tier
- [ ] Clicking a workspace row expands to show per-workspace breakdown
- [ ] Cost trend chart: stacked area by workspace over time
- [ ] "Over quota" alert section: workspaces approaching or exceeding plan limits
- [ ] API call monitoring: top API endpoints by call volume, error rates per endpoint
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-009: Remove existing scattered analytics
**Description:** As a developer, I need to remove analytics tabs from individual pages so users have one source of truth.

**Acceptance Criteria:**
- [x] Remove `monitoring-tab.tsx` content from `/workflows` page (memory, RAG, tools tabs)
- [x] Remove `agent-performance.tsx` content from `/agents` page (resource usage, task history, performance trends, activity logs)
- [x] Remove `analytics-tab.tsx` and `document-analytics.tsx` from `/documents` page
- [x] Remove existing `/dashboard` page and route (redirect to `/analytics`)
- [x] Update sidebar: remove "System Dashboard" item, rename "Intelligence & Learning" to "Analytics"
- [x] Existing management functionality on each page (create/edit/delete agents/workflows/documents) remains untouched
- [ ] No broken imports or dead code left behind (old files still on disk — delete after testing)
- [ ] Typecheck/lint passes

### US-010: AI-powered optimization recommendations
**Description:** As a user, I want Automatos to tell me how to save money and improve performance without me having to analyze data myself.

**Acceptance Criteria:**
- [x] Recommendation card format: icon + title + description + action button
- [x] Recommendation types: cost, performance, document, quota — with type-based styling
- [x] Each recommendation has a dismiss/snooze option
- [x] Max 5 recommendations shown, sorted by estimated impact
- [ ] Recommendations engine that analyzes: agent LLM usage vs. task complexity, token consumption patterns, document retrieval effectiveness, workflow failure patterns (requires backend)
- [ ] Recommendations use LLM to generate natural language explanations from structured data (requires backend)
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-011: Plan usage and quota display
**Description:** As a user, I want to see how much of my plan I've used so I know if I'm approaching limits.

**Acceptance Criteria:**
- [x] Plan usage component showing current plan name and tier
- [x] Usage bars for each tracked resource: agents (count), workflows (count), documents (storage GB), API calls (monthly count), token usage (monthly tokens)
- [x] Color coding: green (< 70%), yellow (70-90%), red (> 90%)
- [x] Shows "No limit set (pilot)" when limit is null
- [ ] "Upgrade" link/button when any resource exceeds 80%
- [ ] Plan comparison tooltip or link
- [ ] Quota data fetched from real workspace plan configuration (requires backend)
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

---

## Functional Requirements

- **FR-1:** The `/analytics` page must be accessible to ALL authenticated users (not admin-only) — **DONE**
- **FR-2:** The page must use a tabbed layout: Overview, Agents, Workflows, Documents, LLM & Costs, and (admin-only) Admin — **DONE**
- **FR-3:** All data must be fetched from backend APIs via React Query hooks — no hardcoded mock data — **DONE** (hooks aggregate from existing endpoints; new dedicated endpoints pending)
- **FR-4:** Time range selector (7d/30d/90d) must apply globally across all tabs — **DONE**
- **FR-5:** Tables must support sorting by clicking column headers — **DONE**
- **FR-6:** Plan quota bars must reflect real workspace plan limits from the backend — **PENDING** (using null/pilot placeholders)
- **FR-7:** AI recommendations must be generated by calling an LLM endpoint with structured usage data as context — **PENDING** (frontend cards built, backend endpoint needed)
- **FR-8:** Admin tab must query cross-workspace data and only render for `isAdmin === true` users — **DONE** (frontend built, backend endpoint needed)
- **FR-9:** All metrics must show loading skeletons (not spinners) while data fetches — **DONE**
- **FR-10:** Export functionality: Admin tab supports CSV export of workspace usage data — **DONE**
- **FR-11:** Responsive layout: works on desktop (1024px+) and tablet (768px+). Mobile is not required. — **DONE**
- **FR-12:** The page must reuse existing shadcn/ui components and Recharts for charts — **DONE**
- **FR-13:** Existing analytics components on `/workflows`, `/agents`, `/documents` pages must be removed entirely — **DONE** (imports removed; old files still on disk for safety)
- **FR-14:** The `/dashboard` route must redirect to `/analytics` — **DONE**
- **FR-15:** Memory preferences section must allow inline editing of stored user preferences — **PENDING**
- **FR-16:** Document analytics must surface "never accessed" documents with actionable suggestions — **DONE**
- **FR-17:** Agent analytics must flag agents using expensive LLMs for simple tasks — **DONE**
- **FR-18:** Cost data must be broken down by LLM model, by agent, and by time period — **DONE**

---

## Non-Goals (Out of Scope)

- **No billing/payment UI** — No credit card forms, invoices, or payment processing. Just display usage vs. plan limits.
- **No plan management** — Users cannot upgrade/downgrade plans from this page (link to external billing page is fine)
- **No real-time WebSocket streaming** — Polling on page load + time range change is sufficient. Real-time can come later.
- **No report generation** — PDF/scheduled reports will be a separate PRD (Panda integration mentioned by user)
- **No mobile optimization** — Tablet and desktop only
- **No notification system** — No email/Slack alerts when quotas are hit. Display-only for now.
- **No custom date ranges** — Fixed presets (7d/30d/90d) only. Custom date picker is a future enhancement.
- **No A/B testing of recommendations** — Ship recommendations v1, measure later

---

## Design Considerations

### Layout Structure

```
/analytics
├── Header: "Analytics" + time range selector (7d/30d/90d) + Refresh + Export
├── Tab Bar: Overview | Agents | Workflows | Documents | LLM & Costs | [Admin]
│
├── Overview Tab
│   ├── 4x Summary Cards (Agents, Workflows, Documents, Monthly Cost)
│   ├── Plan Usage Bars (agents, storage, API calls, tokens)
│   ├── Automatos Memory Card (editable preferences)
│   └── AI Recommendations (up to 5 cards)
│
├── Agents Tab
│   ├── 4x Summary Cards
│   ├── Filter Bar (status, model, health)
│   └── Sortable Agent Table (expandable rows)
│
├── Workflows Tab
│   ├── 4x Summary Cards
│   ├── Execution Trend Chart (success/fail stacked bars)
│   └── Sortable Workflow Table (expandable rows)
│
├── Documents Tab
│   ├── 3x Summary Cards (docs, storage quota, RAG queries)
│   ├── "Unused Documents" Alert
│   ├── RAG Performance Section
│   └── Sortable Document Table
│
├── LLM & Costs Tab
│   ├── 4x Summary Cards (tokens, cost, cost/task, most expensive agent)
│   ├── Cost Breakdown Chart (stacked area by model)
│   ├── Token Usage Table (by model)
│   ├── Per-Agent Cost Table
│   └── Plan Quota Section
│
└── Admin Tab (super admin only)
    ├── Platform-level Metrics (5x cards)
    ├── Workspace Table (expandable rows)
    ├── Cost Trend Chart (by workspace)
    ├── Over-Quota Alerts
    └── Export CSV Button
```

### Existing Components Reused
- `Card`, `CardContent`, `CardHeader`, `CardTitle` — from shadcn/ui
- `Tabs`, `TabsContent`, `TabsList`, `TabsTrigger` — from shadcn/ui
- `Badge`, `Progress`, `Button`, `Select` — from shadcn/ui
- `Recharts` — `AreaChart`, `BarChart`, `LineChart`, `PieChart` for all visualizations
- `motion` from framer-motion — for entry animations (match existing page patterns)
- `useSystemRole()` — for admin detection
- `useAnalyticsSuccessRate`, `useAllMetrics`, `useCostAnalysis` — existing hooks extended in `use-unified-analytics.ts`

### Tone for Users
- Non-technical language: "Your email agent is using an expensive AI model" not "GPT-4 token consumption exceeds threshold"
- Actionable: Every metric should answer "so what?" — surface the insight, not just the number
- Friendly: Recommendations should feel like a helpful assistant, not a warning system

---

## Technical Considerations

### Frontend Architecture
All components live in `/frontend/components/analytics/`:
- `analytics-page.tsx` — main layout, tabs, time range state
- `analytics-overview.tsx` — Overview tab content
- `analytics-agents.tsx` — Agents tab content
- `analytics-workflows.tsx` — Workflows tab content
- `analytics-documents.tsx` — Documents tab content
- `analytics-costs.tsx` — LLM & Costs tab content
- `analytics-admin.tsx` — Admin tab content
- `analytics-recommendations.tsx` — AI recommendation cards
- `analytics-plan-usage.tsx` — Plan quota bars
- `analytics-memory.tsx` — Automatos memory preferences

Hooks in `/frontend/hooks/use-unified-analytics.ts` (8 hooks):
- `useAnalyticsOverview` — aggregates agent/workflow/document counts + cost
- `useAgentAnalytics` — per-agent performance data
- `useWorkflowAnalytics` — per-workflow execution data
- `useDocumentAnalyticsUnified` — document usage + RAG effectiveness
- `useCostAnalyticsUnified` — token/cost breakdown by model and agent
- `usePlanUsage` — plan limits vs current usage (pilot: null limits)
- `useRecommendations` — AI suggestions (staleTime: 24h daily cache)
- `useWorkspaceMemory` — stored user/agent preferences
- `useAdminWorkspaceAnalytics` — cross-workspace stats (admin only)

### Backend API Requirements (PENDING)
New or extended endpoints needed:

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `GET /api/analytics/overview` | Overview summary (agent/workflow/doc counts, period cost) | Pending |
| `GET /api/analytics/agents` | Per-agent performance table data | Pending (using existing agent APIs) |
| `GET /api/analytics/workflows` | Per-workflow execution data | Pending (using existing workflow APIs) |
| `GET /api/analytics/documents` | Document usage + RAG effectiveness | Pending (using existing doc APIs) |
| `GET /api/analytics/costs` | Token usage + cost by model, by agent | Pending |
| `GET /api/analytics/recommendations` | AI-generated optimization suggestions | Pending |
| `GET /api/analytics/plan-usage` | Current plan + usage vs. quota | Pending |
| `GET /api/analytics/memory` | User preference memory items | Pending (using existing memory APIs) |
| `PUT /api/analytics/memory/:id` | Update a memory preference | Pending |
| `GET /api/admin/analytics/workspaces` | Cross-workspace usage (admin only) | Pending |
| `GET /api/admin/analytics/export` | CSV export of workspace data (admin only) | Pending |

All endpoints must accept `?days=7|30|90` and `?workspace_id=<uuid>` query params.

### Data Model Additions (PENDING)

```sql
-- Plan/quota tracking (new table)
CREATE TABLE workspace_plans (
  id UUID PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  plan_tier VARCHAR(50) NOT NULL,  -- 'starter', 'small_business', 'business', 'enterprise'
  max_agents INTEGER,
  max_workflows INTEGER,
  max_storage_gb FLOAT,
  max_api_calls_monthly INTEGER,
  max_tokens_monthly BIGINT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking (new table, updated daily)
CREATE TABLE workspace_usage (
  id UUID PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,
  agent_count INTEGER DEFAULT 0,
  workflow_count INTEGER DEFAULT 0,
  document_count INTEGER DEFAULT 0,
  storage_used_gb FLOAT DEFAULT 0,
  api_calls INTEGER DEFAULT 0,
  tokens_used BIGINT DEFAULT 0,
  total_cost FLOAT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Token usage per model per agent (new table)
CREATE TABLE token_usage_log (
  id UUID PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  agent_id INTEGER REFERENCES agents(id),
  workflow_execution_id INTEGER REFERENCES workflow_executions(id) NULL,
  llm_model_id INTEGER REFERENCES llm_models(id),
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost FLOAT NOT NULL,
  task_type VARCHAR(100),  -- for recommendation engine
  task_complexity VARCHAR(20),  -- 'simple', 'moderate', 'complex'
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Existing Infrastructure Leveraged
- `LLMModel` table already stores `input_cost_per_1k_tokens` and `output_cost_per_1k_tokens`
- `Agent.model_usage_stats` JSON field tracks `total_tokens`, `total_cost`, `total_requests`
- `WorkflowExecution` tracks `started_at`, `completed_at`, `status`, `duration`
- `AgentStatistics` tracks `success_rate`, `avg_execution_time`, `total_executions`
- Existing React Query hooks (`use-analytics-api.ts`, `use-agent-api.ts`, etc.) aggregated in unified hooks

### Performance
- Tab content lazy-loads (only fetch data when tab is selected)
- Agent/workflow/document tables: paginate at 25 rows, load more on scroll
- Admin workspace table: paginate at 20 rows with server-side pagination
- Cache analytics data for 60 seconds (React Query `staleTime`)
- Recommendations cached for 24 hours (`staleTime: 86400000`)

---

## Code Reuse Audit

### Hooks Reused & Extended
- `use-analytics-api.ts` — `useAllMetrics()`, `useCostAnalysis()`, `useAnalyticsSuccessRate()`
- `use-agent-api.ts` — `useAgents()`, `useAgentPerformance()`, `useAgentLogs()`
- `use-workflow-api.ts` — `useWorkflows()`, `useWorkflowStatsDashboard()`
- `use-document-api.ts` — `useDocumentAnalytics()`, `useUsageAnalytics()`, `useProcessingAnalytics()`
- `use-usage-analytics-api.ts` — `useUsageAnalytics()` for document RAG tracking
- `use-orchestration-data.ts` — `useAggregatedOrchestrationData()`

### Patterns Adapted
- **Metric card pattern** from `monitoring-tab.tsx` — glass-card with icon, value, trend
- **Sortable table pattern** from `analytics-tab.tsx` — document type breakdown
- **Chart tooltip style** — consistent dark tooltip from `performance-analytics.tsx`
- **Skeleton loading** from `agent-performance.tsx` — loading states
- **Time range selector** pattern — Select with 7d/30d/90d
- **Badge status indicators** — color-coded status badges

### Design Tokens (from globals.css)
- `glass-card` class for all cards
- `card-glow` for hover effects
- `gradient-text` for accent headings
- Chart palette: `--chart-1` through `--chart-5`
- Orange brand color: `hsl(16 100% 55%/60%)`

---

## Files to Delete After Testing

These files are no longer imported but remain on disk until testing confirms no regressions:

- `frontend/components/workflows/monitoring-tab.tsx`
- `frontend/components/agents/agent-performance.tsx`
- `frontend/components/documents/analytics-tab.tsx`
- `frontend/components/documents/document-analytics.tsx`
- `frontend/components/analytics/performance-analytics.tsx` (old pre-unified analytics)

---

## Success Metrics

- All users can access `/analytics` (not just admins)
- Users can identify their most expensive agent within 10 seconds of opening the page
- Users can see plan usage vs. quota without navigating away
- AI recommendations surface at least 1 actionable suggestion per workspace with active usage
- Admin can view cross-workspace costs sorted by highest spend in under 5 seconds
- Zero analytics-related components remain on `/agents`, `/workflows`, `/documents` pages after migration
- Page loads with data in < 2 seconds on broadband

---

## Open Questions — RESOLVED

1. **Plan tiers definition** — TBD. Running a pilot first to monitor real usage, then setting limits. For now, use placeholder limits in the UI that can be configured later.

2. **Recommendation frequency** — Cache daily (not per page load). Recommendations are expensive to generate and usage patterns don't change that fast. Refresh once per day or on manual trigger.

3. **Memory scope** — Per workspace. Also includes agent-specific memory (e.g., communication agent remembers John's email, your Slack channel preferences).

4. **Document access tracking** — RAG tracking already exists via `/api/documents/analytics/usage` endpoint and `useUsageAnalytics` hook. Reuse this.

5. **CSV export** — Simple initial implementation matching workspace table columns. Can iterate later.

6. **Dashboard removal** — Remove `/dashboard` entirely. Audit all dashboard widgets first — move useful patterns into unified analytics. Don't rewrite from scratch where existing code works.
