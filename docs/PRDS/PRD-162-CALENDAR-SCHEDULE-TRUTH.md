# PRD-162 — Calendar & Schedule Truth (WS-7)

**Chain:** Block A lead (first after Night-1 — owner-facing pain). Branch `ralph/prd-162-calendar-truth` from main. Size **S**.
**Source:** report §2.8; root cause: APScheduler in-process state exists in 1 of 4 uvicorn workers → 75% of loads return empty 200s and the UI looks like it loads for minutes.

## Overview

The calendar becomes a stateless read of configured schedules from the DB — identical on every worker, seconds not minutes — and one unified feed of everything time-based.

## Binding amendments

D8 (studio CalendarTab is canonical; classic ActivityCalendar deleted here per delete-what-you-replace — its unique features port over), Q46 default: feed includes heartbeats + playbook crons + scheduled_tasks + one-shots + mission SLAs, Q48 default: render browser-local with per-item source timezone on hover; active-hours windows mask non-firing occurrences, Q49: configured crons always render, scheduler health is a separate banner, Q50: 30–60s cache TTL acceptable, Q51 default: scheduler stays in-API with thread-offloaded calls (no new Railway service this PRD).

## User Stories

### S1: DB-first `get_schedule`
Rewrite to read `agents.configuration['heartbeat']` + `workflow_templates.schedule_config` (+ `scheduled_tasks`) and compute `next_run` statelessly with one shared cron util; return structured `{cron_expression, interval_minutes, timezone, active_hours}`. Collapse the heartbeat-workspace N+1 to one `DISTINCT ON` query. Delete the three regex schedule parsers for the one `cron-parser`-equivalent util (backend: croniter — already a transitive dep; verify, else add explicitly).
**Acceptance:**
- [ ] Same response from all 4 workers (test boots app twice, compares)
- [ ] p95 endpoint latency < 500ms with 200 schedules seeded (benchmark test)
- [ ] One SQL round-trip for the list (query-count assertion)
- [ ] Regex parsers deleted; structured fields validated by schema test

### S2: Unified feed + month view
Merge heartbeats, playbook crons, scheduled_tasks, one-shot runs, and mission SLA deadlines into typed feed items; month-view renders states (scheduled/paused/missed); scheduler-health banner sourced from a real health endpoint, never blocking render.
**Acceptance:**
- [ ] Feed test: all five source types present with type discriminators
- [ ] Calendar paints < 2s after navigation on seeded data — dev-browser verify
- [ ] Paused scheduler shows banner + still renders configured crons — dev-browser verify

### S3: `platform_get_schedule` tool
Same DB-first service exposed via the 3-file tool pattern so Auto can answer "what's scheduled?" without screenshots.
**Acceptance:**
- [ ] Tool returns the unified feed scoped to workspace (test); reachability gate green

### S4: Client truth
React Query keyed cache with 30–60s TTL + background refetch; the PRD-154 S11 stopgap (retry-on-inactive) removed as moot; classic `components/activity/calendar/*` deleted after porting next-up/always-running widgets into studio CalendarTab.
**Acceptance:**
- [ ] No retry-loop code remains (grep gate)
- [ ] Classic calendar tree deleted; contract tests green; nav has no orphan link
- [ ] Refresh feels instant (cache) with background update — dev-browser verify

## Non-Goals

Dispatcher/claim loop (161), dedicated scheduler worker container (revisit only if in-API + to_thread proves insufficient — record metrics first).

## Success Metrics

- Calendar first-paint < 2s on production-shaped data (vs minutes).
- Zero empty-calendar incidents across 20 consecutive loads on 4 workers (soak test).

## Testing

New `test_schedule_endpoint.py` (stateless/N+1/latency), feed-merge tests, vitest for the query/cache layer. Deleted-component test references cleaned up. Full suite + contract green.
