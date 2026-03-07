# Implementation Plan: PRD-72 Activity Command Centre

> **Scope**: Cross-cutting (frontend + backend) | **Risk**: Balanced | **Branch**: `ralph/72-activity-command-centre`

## Summary

Replace `/workflows` with `/activity` — a unified Activity Command Centre showing chats, routines, recipes, and missions across four tabs. Backend: register heartbeat router, create activity service + API endpoints. Frontend: new page shell, feed items, routine cards, execution detail drill-down, live stats.

## Reference

- **PRD**: `docs/PRDS/72-ACTIVITY-COMMAND-CENTRE.md`
- **Design system**: `frontend/app/globals.css` (glass-card, card-glow, stage-*, log-entry-*, semantic colour tokens)
- **Shared components**: `frontend/components/shared/` (PageHeader, StatsBar, FilterTabs, SearchInput)
- **Hook pattern**: `frontend/hooks/use-workflow-api.ts` (React Query key factory + useQuery + useMutation)
- **Backend pattern**: `orchestrator/api/execution_history.py` (FastAPI router + hybrid auth)
- **Existing heartbeat API**: `orchestrator/api/heartbeat.py` (exists but NOT registered in main.py)

## Tasks

- [x] **US-001: Create /activity route shell page** — Create `frontend/app/activity/page.tsx` + `frontend/components/activity/activity-page.tsx` with PageHeader, StatsBar (hardcoded), FilterTabs (4 tabs). Update sidebar.tsx nav item. Add `/workflows` → `/activity` redirect. NOTE: `CookingPot` icon doesn't exist in this lucide-react version — use `ChefHat` instead.
- [x] **US-002: Build Missions placeholder + wire RecipesTab** — Created `activity-missions.tsx` Coming Soon card with glass-card styling, gradient-text accent, capability bullets, and "Request Early Access" button routing to /chat. Wired existing RecipesTab into Recipes tab and ActivityMissions into Missions tab of activity-page.tsx.
- [x] **US-003: Register heartbeat router + add endpoints** — Registered heartbeat_router in main.py (import + include_router). Added GET /api/heartbeat/workspace (lists all agent heartbeat configs with last_run and next_run), PATCH /api/heartbeat/{id}/toggle (toggles enabled flag + updates APScheduler), GET /api/heartbeat/{id}/executions (returns last N results with error extraction from findings JSONB). NOTE: heartbeat_id = agent_id since heartbeat config lives in agent.configuration.heartbeat JSON, not a separate table.
- [x] **US-004: Create activity_service.py** — Created `orchestrator/services/activity_service.py` with `ActivityService` class (request-scoped, accepts `db: Session` + `workspace_id`). `get_feed()` merges chats (via messages JOIN for workspace scoping), heartbeat_results (raw SQL with agent JOIN), and recipe_executions (ORM). Batch-fetches recipe names. `get_stats()` returns working_now (running recipes), channels_live (connected channel_connections), completed_today, needs_attention. NOTE: chats table has no workspace_id — workspace scoping uses EXISTS subquery through messages table. heartbeat_results has no ORM model — uses raw SQL. Agent avatar uses `marketplace_icon` field (no `avatar_url` column).
- [x] **US-005: Create /api/activity/feed + /api/activity/stats endpoints** — Created `orchestrator/api/activity.py` with GET /api/activity/feed (query params: type CSV, status, period, limit, offset) and GET /api/activity/stats (query param: period). Both use get_request_context_hybrid() auth. Delegates to ActivityService. Registered activity_router in main.py. NOTE: Local import test fails on DB credentials (expected — same as all other routers), syntax validation passes.
- [x] **US-006: Create use-activity-api.ts hooks** — Created `frontend/hooks/use-activity-api.ts` with TypeScript interfaces (ActivityFeedItem, ActivityStats, ActivityFeedResponse, ActivityFeedFilters, ActivityChannel, ActivityStepProgress, ActivityAgent), query key factory (activityQueryKeys.feed/stats), useActivityFeed(filters) with 15s polling + param serialization, useActivityStats(period) with 15s polling. Uses apiClient.request<T>() directly (same pattern as use-database-knowledge.ts).
- [ ] **US-007: Create use-heartbeats-api.ts hook** — React Query hooks for heartbeat list, toggle mutation, execution history.
- [ ] **US-008: Build routine-card.tsx** — Glass-card component showing agent, schedule, status badge, pause/edit actions.
- [ ] **US-009: Build activity-routines.tsx** — Routines tab with card grid, New Routine CTA, empty state, loading skeletons. Wire into activity-page.
- [ ] **US-010: Add execution history to routine-card** — Expandable section showing last 10 runs with log-entry CSS classes.
- [ ] **US-011: Build activity-feed-item.tsx** — Feed card with type-coloured border, status badge, channel badge, context line, View/Configure actions.
- [ ] **US-012: Build activity-feed.tsx + wire live stats** — Feed tab with filter chips, status dropdown, renders feed items. Replace hardcoded StatsBar with live data.
- [ ] **US-013: Build execution-detail.tsx** — Drill-down view with step pipeline (stage-* classes), output panel, execution log, Re-run/Edit actions.
- [ ] **US-014: Add run history dots to recipe cards** — Modify RecipesTab to show last 3 runs as coloured status dots per recipe card.
- [ ] **US-015: Loading skeletons + animations** — Skeleton states for all tabs. stage-active pulse on running items. log-slide-in for new feed items. framer-motion stagger. prefers-reduced-motion.
- [ ] **US-016: Mobile responsive pass** — 2x2 stats grid, single-column cards, scrollable filter chips, reduced blur, 44px touch targets.
- [ ] **US-017: Update SHEPHERD tour for /activity** — Create activity-tour.ts. Update tour-registry.ts. Add data-tour attributes.
