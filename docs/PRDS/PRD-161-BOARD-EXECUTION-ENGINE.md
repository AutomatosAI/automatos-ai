# PRD-161 — Board Execution Engine (WS-6)

**Chain:** Block A, branch `ralph/prd-161-board-engine` from main after Night-1. Size **M**. Feeds PRD-162 (shared dispatch), PRD-164.
**Source:** report §2.7, §3.2 (dispatch spine); PRD-154 S4 landed dispatch-on-assign + priority fix.

## Overview

One dispatch spine: Postgres-native claim/lease/requeue so assigned work ALWAYS executes exactly once, fails honestly, and retries deliberately — replacing the opt-in heartbeat fold-in (3-tasks-batched-into-1) that made pickup look random. Same Postgres, no new services.

## Binding amendments

D8 (board UI fixes land in studio board-tab), D11 (per-task execution accepted — that's the point; per-agent concurrency slots cap spend), Q39 default: assignment = immediate dispatch with per-workspace `auto_start` override, Q40: board dispatcher independent of heartbeats (heartbeat demotes to monitoring/recurring-only), Q41 retries: 2 attempts then `failed`, Q44: rejection returns task to same agent as `assigned` with feedback context, Q45 default: finish SLA on the existing `sla_deadline` column, Q42: external SDK-agent writes (Slice 3 `tasks:write`) stay OUT — PRD-09 narrow-dep pattern noted for a future PRD.

## User Stories

### S1: Claim loop — `FOR UPDATE SKIP LOCKED` + LISTEN/NOTIFY
Worker claim loop over `board_tasks` with lease + attempts columns (migration extends the existing table — no new table); `pg_notify` on assign/create wakes claimants; poll fallback. No fcntl leader election for pickup.
**Acceptance:**
- [ ] Concurrency test: 4 workers, 50 tasks → each executed exactly once (no double-claim)
- [ ] Crash test: lease expiry requeues with attempts+1 instead of closing done
- [ ] NOTIFY latency test: assign→claim under 1s in-process
- [ ] Migration reversible; alembic single-head preserved

### S2: One real execution per task
Delete the heartbeat 3-task fold-in; each claimed task runs the existing execution path individually with guardrail validation; results land on the task (typed activity entries, not blob text).
**Acceptance:**
- [ ] Batch fold-in code deleted (no shim)
- [ ] Per-task result + typed activity feed entries (schema test)
- [ ] `test_board_task_handlers.py` updated; `test_board_sdk_auth.py` green (PRD-09 untouched)

### S3: Honest lifecycle — `failed`, rejection feedback, sweeper
Terminal `failed` state + `task_failed` notification (today: done+error); reject → back to same agent as `assigned` with reviewer feedback in context (Q44); sweeper requeues stale leases and flags `unresponsive` agents (ack-deadline badge).
**Acceptance:**
- [ ] State-machine test covers assigned→in_progress→{done|failed|rejected→assigned}
- [ ] Studio board shows failed column + unresponsive badge — dev-browser verify
- [ ] Notification emitted on failure (test)

### S4: Throughput controls
Per-agent concurrency slots (config, default 2) + double-texting policy (queue, don't drop); `asyncio.to_thread` on sync Composio calls inside task execution so the loop never blocks; OpenHands-style StuckDetector port (same-action-loop breaker).
**Acceptance:**
- [ ] Slot test: 5 tasks to one agent → ≤2 concurrent, rest queued
- [ ] Event-loop block test: sync tool call doesn't stall other claims
- [ ] Stuck test: repeated identical failing action breaks the loop with a task_learning memory write (hooks into PRD-159 S2 if merged; otherwise logs)

### S5: SLA + board polish
Wire the dead `sla_deadline`: set on agent-created tasks, breach scan in the sweeper, indicator chips in studio board; Run-Now button; no-heartbeat warning removed (dispatcher makes it moot); archive done > N days (config).
**Acceptance:**
- [ ] Breach test: overdue task flagged + notification
- [ ] Run Now dispatches immediately — dev-browser verify
- [ ] SSE board events behind the PRD-09 TASKS_READ narrow dep (read-only) — auth tests green

## Non-Goals

Calendar (162), mission sub-task orchestration (163), SDK `tasks:write` Slice 3, classic board page (sunset in 169 per D8).

## Success Metrics

- 100% of assigned tasks reach a terminal state without human nudging in the 50-task soak test.
- Zero double-executions across 4 workers (soak).
- Median assign→start < 2s.

## Testing

New `test_board_dispatch.py` (claim/lease/concurrency/sweeper — DB-backed, follows `test.yml` Postgres pattern), state-machine tests, SSE auth tests. Updated: `test_board_task_handlers.py`. Full suite + contract green.
