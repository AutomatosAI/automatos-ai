# PRD-227 — Light Up the Board: Live Agent Activity & Orchestration Narration

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled.
> **Origin:** Munder Difflin deep review (2026-08-27) — visibility is what makes their demo read as "a managed floor"; ours runs dark. Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Wiring/extension. One-seam diffs on existing event paths; no schema, no new routes.

## 1. Overview

Agent activity must be as live as human activity, or PRD-224's management never reads as management. Today the board's SSE lane pushes human moves in real time while agent moves land silently in the DB and appear on the next stale refetch; mission progress reaches chat only via watch verdicts; and two notification link types navigate nowhere. This PRD is the small set of wiring diffs that make the existing loop visible.

## 2. Current reality (grounded)

- **Human board moves push, agent moves don't.** The human PATCH path calls `notify_board_event` (`api/board_tasks.py:912-916`); the agent-side handler `update_board_task_status` does **not**, and accepts only `{inbox, assigned, in_progress, review, done}` — no `blocked`/`failed` (`modules/tools/discovery/handlers_board_tasks.py:258-320`, enum at `:270`). SSE infra itself is solid: Postgres LISTEN/NOTIFY on `board_events`, workspace-scoped frames (`services/board_events.py:34-231`), consumed via fetch-stream with auth headers (`frontend/hooks/use-board-event-stream.ts`).
- **Only two SSE event names exist:** `board_changed` and `chat_changed` (`services/board_events.py:229-231`).
- **Orchestration never narrates.** The PRD-205 background→chat seam (`services/chat_messenger.py:96-202`) is generic and battle-tested, with exactly two producers: watch notifications (`services/watch_notifications.py:87-100`) and scheduled tasks (`services/scheduled_task_service.py:446-465`). Mission lifecycle events (`RUN_*`, `TASK_*` in `core/models/orchestration_enums.py`) never reach the thread that launched the mission.
- **The bell can't link what matters.** `linkFor()` handles `task, mission, playbook, heartbeat, report, trigger, agent` — not `approval_grant`, not `watch` (`frontend/components/notifications/notification-bell.tsx:42-64`) — yet those are exactly the `link_type`s written by `modules/tools/execution/tool_grants.py:412` and `services/watch_notifications.py:75`. Those notifications mark-read and navigate nowhere.
- Mission detail UI polls (10s) and renders a full state vocabulary already (`frontend/types/missions.ts`); no change needed there for this PRD.

## 3. Goals

- G1: An agent moving a card is visible on the open Command Center within the SSE latency budget, exactly like a human move.
- G2: Agent-side status vocabulary reaches parity where it matters: agents can set `blocked` (with reason) and `failed`.
- G3: A mission launched from a chat thread narrates its lifecycle back into that thread: approved → started → each task done/failed → completed/failed, throttled to stay readable.
- G4: Every notification the platform writes is navigable from the bell.

## 4. Non-goals

- No new SSE event names beyond the existing two (a `board_changed` frame already triggers the right refetches; `notification_created` push is deliberately deferred — the 30s bell poll stays).
- No mission UI rework; no WebSockets; no changes to the dispatcher/leases.
- Not the ticket-lane behavior itself (PRD-224) — this PRD makes it visible.

## 5. Design

- **Component A — agent-move SSE parity.** `handlers_board_tasks.py:258-320`: after a successful status write, call `notify_board_event` with the same payload shape as `api/board_tasks.py:912`; extend the handler's allowed statuses with `blocked` (requires `blocked_reason`, sets `blocked_at`) and `failed`, mirroring the API-side transitions (`api/board_tasks.py:548-553, 898-902`). Same for the create/assign handlers (create already notifies on the HTTP path at `:389`).
- **Component B — mission narration producers.** In the coordinator's state-transition points (approve at `coordinator_service.py:3129`, task terminal recording at `:2272 _record_task_result`, run terminal states), call `deliver_background_message` targeting the mission's originating chat (origin capture already exists — `watches.origin_chat_id` pattern at `platform_executor.py:1111`; missions carry `created_by` and the creating chat via the same executor context). Throttle: narrate run-level events always; task-level events collapse to one line per task terminal state; suppress task lines for runs with >N tasks (config `MISSION_NARRATION_TASK_CAP` in `config.py`). Source label `"Auto · mission"` with `link_type="mission"` so the badge deep-links.
- **Component C — bell links.** Add `case 'approval_grant'` (→ Command Center governance/approvals; → Questions tab for `kind=question` once PRD-225 lands) and `case 'watch'` (→ watchlist tab, `?tab=watchlist`) to `linkFor()`.

## 6. Waves & acceptance criteria

Single wave — the three components are independent one-seam diffs and can ship as one PR or three.

- [ ] Agent-driven status change (via `platform_update_task_status`) produces a `board_changed` SSE frame observed by the existing stream test pattern; UI refetch happens without waiting for staleness (component/behavior test mirroring the human-move path).
- [ ] Agent can set `blocked` with a reason (renders in the existing `board-card.tsx:152-155` blocked strip) and `failed`; invalid transitions rejected identically to the HTTP path.
- [ ] A chat-launched mission produces narration messages in the launching thread with the provenance badge; a >N-task mission stays under the throttle cap; a wizard-launched mission (no originating chat) narrates to the Auto thread (existing `find_or_create_auto_chat` fallback).
- [ ] Bell: approval and watch notifications navigate to their surfaces; no `link_type` written anywhere in the backend lacks a `linkFor` case (add a unit test enumerating producers vs cases so this can't drift again).
- [ ] No new routes; route manifest untouched; CI green.

## 7. Technical considerations

- `notify_board_event` runs on its own connection semantics — follow the existing fail-soft pattern (`board_events.py:38-70`): a NOTIFY failure must never fail the tool call.
- Narration writes through `ChatService.save_message` via the messenger — rate limits and message ordering are inherited; the throttle cap is the only new knob.
- Interaction: PRD-224's watch verdicts also narrate; run-level narration and watch verdicts are distinct events (start/progress vs judged outcome) — wording must keep them distinguishable.

## 8. Open questions (Gerard)

1. Narration default: on for all missions, or only chat-launched ones? Proposal: all (wizard/scheduled runs narrate to the Auto thread) — visibility is the point.
2. `MISSION_NARRATION_TASK_CAP` default (proposal: task-level lines suppressed above 8 tasks; run-level always).
