# PRD-128: Unified Notification System — Implementation Plan

## Overview

A single notification pipeline that captures completion events from every source (heartbeats, tasks, missions, playbooks, triggers, reports, agent errors), routes them via per-workspace preferences (in_app / telegram / slack / webhook / silent), and surfaces in-app notifications via a bell-icon dropdown. Reuses the existing `notification_service.py` fan-out and `channel_connections` table.

## Architecture

```
<event source>
  └─ NotificationDispatcher(db, workspace_id).dispatch(event_type, ...)
       ├─ read notification_preferences (workspace defaults + user overrides)
       ├─ for each enabled pref:
       │    ├─ silent   → skip
       │    ├─ in_app   → INSERT notifications row (no commit; caller owns txn)
       │    ├─ telegram/slack/webhook → notification_service.send_workspace_notification
       │    └─ channel  → send to specific channel_connection_id
       └─ return {dispatched_to: [...]}

Frontend:
  navbar <NotificationBell/>
    ├─ poll /api/notifications/unread-count (30s)
    ├─ open → GET /api/notifications?limit=20
    └─ click row → POST /{id}/read + next/navigation router.push(linkFor(row))

Settings:
  /settings/notifications → GET /api/notification-preferences / PUT bulk upsert
```

## Key Files

| Area | File | Purpose |
|------|------|---------|
| DB schema | `orchestrator/alembic/versions/prd128_notifications.py` | Tables + indexes (US-001) |
| Provisioning | `orchestrator/core/auth/hybrid.py` | `_provision_new_user_workspace()` — seed 9 prefs (US-002) |
| Service | `orchestrator/core/services/notification_dispatcher.py` | NEW dispatcher class (US-003) |
| Fan-out reuse | `orchestrator/services/notification_service.py` | `send_workspace_notification` entry point |
| API | `orchestrator/api/notifications.py` | NEW — notifications + preferences routers (US-004, US-005) |
| Main | `orchestrator/main.py` | Include new routers |
| Heartbeats | `orchestrator/services/heartbeat_service.py` | Replace `_deliver_notification` (US-006) |
| Tasks | `orchestrator/api/tasks.py` | Dispatch task_complete (US-007) |
| Missions | `orchestrator/services/coordinator_service.py` | mission_step_complete / mission_complete (US-007) |
| Playbooks | `orchestrator/services/playbook_executor.py` | playbook_step_complete / playbook_complete (US-007) |
| Reports | `orchestrator/services/report_service.py` | report_submitted (US-007) |
| Bell UI | `frontend/components/notifications/notification-bell.tsx` | NEW popover (US-008) |
| Navbar | `frontend/components/navbar.tsx` (or equivalent) | Mount bell component (US-008) |
| Settings UI | `frontend/app/settings/notifications/page.tsx` | NEW settings page (US-009) |
| Sidebar | `frontend/components/settings/*` | Add link (US-009) |
| E2E | `tests/integration/test_notification_pipeline.py` | Smoke test (US-010) |

## Event Types (9)

| Event | Default | Description |
|-------|---------|-------------|
| `heartbeat_complete` | in_app | Heartbeat cycle finished |
| `task_complete` | in_app | Board task marked complete |
| `mission_step_complete` | silent | Per-step mission progress (noisy) |
| `mission_complete` | in_app | Mission terminal state |
| `playbook_step_complete` | silent | Per-step playbook progress (noisy) |
| `playbook_complete` | in_app | Playbook finished |
| `trigger_fired` | in_app | Composio trigger fired |
| `report_submitted` | in_app | Agent submitted a report |
| `agent_error` | in_app | Agent raised an error |

## Tasks

### Phase 1: Schema & Seed

- [x] **US-001**: Alembic migration — `notification_preferences` + `notifications` tables + all indexes (file: `orchestrator/alembic/versions/prd128_notifications.py`, down_revision=`prd127_attachment_ids`)
- [ ] **US-002**: Seed 9 default prefs on workspace provisioning (idempotent)

### Phase 2: Dispatcher & API

- [ ] **US-003**: `NotificationDispatcher` service with full fan-out + unit tests
- [ ] **US-004**: Notifications API — list, unread-count, read, read-all, dismiss
- [ ] **US-005**: Preferences API — GET merged list, PUT bulk upsert

### Phase 3: Wire Sources

- [ ] **US-006**: Migrate `HeartbeatService` to dispatcher, delete `_deliver_notification`
- [ ] **US-007**: Wire tasks, missions (step+complete), playbooks (step+complete), reports

### Phase 4: Frontend

- [ ] **US-008**: `NotificationBell` component + navbar mount
- [ ] **US-009**: `/settings/notifications` page + sidebar link

### Phase 5: Verification

- [ ] **US-010**: End-to-end smoke test of notification pipeline

## Constraints

- **Dispatcher never commits** — caller owns the transaction so notification inserts roll back with the main work on failure.
- **Multi-destination fan-out** — one event_type row may have multiple preference rows; all enabled rows fire (silent skips silently).
- **User overrides workspace default** — when both exist for the same `(event_type, destination)`, user-specific wins.
- **Non-blocking** — every new `dispatcher.dispatch` call is wrapped in try/except log-only so notification bugs never break the primary flow.
- **workspace + user scoping** — all API queries enforce `workspace_id = ctx.workspace_id AND (user_id = ctx.user_id OR user_id IS NULL)`.
- **react-query v4** — `isLoading` not `isPending`, `useRouter().push()` not `window.location.href`.

## Quality Bar

- Every DB query workspace-scoped
- Dispatcher covered by unit tests: silent, in_app, multi-destination, user override, no-prefs default
- Migration up/down clean against local postgres
- Typecheck passes on every story
- No silent failures — all exceptions logged with `exc_info=True`
